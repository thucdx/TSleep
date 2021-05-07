import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2


class ModifiedSleepEEGNet(Model):

    def __init__(self, num_classes=5, input_length=3000, name="ModifiedSleepEEGNet"):
        super(ModifiedSleepEEGNet, self).__init__(name=name)
        self.num_classes = num_classes
        self.conv1 = Conv1D(filters=20, kernel_size=200, strides=1, activation="relu", input_shape=(input_length, 1))
        self.pool1 = MaxPooling1D(pool_size=20, strides=10)

        self.reshape = Reshape(target_shape=(-1, 20, 1))
        self.conv2 = Conv2D(filters=400, kernel_size=(30, 20), strides=(1, 1), activation="relu")
        self.pool2 = MaxPooling2D(pool_size=(10, 1), strides=(2, 1))

        # self.flatten = Flatten()
        self.fc1 = Dense(50, activation="relu")
        self.fc2 = Dense(50, activation="relu")
        self.fc3 = Dense(self.num_classes, activation="softmax", kernel_regularizer=l2(0.01))

    def call(self, inputs, **kwargs):
        # print("Input shape: ", inputs.shape)
        conv1 = self.conv1(inputs)
        # print("conv1 shape: ", conv1.shape)
        pool1 = self.pool1(conv1)
        # print("pool1 shape: ", pool1.shape)

        reshaped = self.reshape(pool1)
        # print("reshaped shape: ", reshaped.shape)
        conv2 = self.conv2(reshaped)
        # print('Conv2 shape: ', conv2.shape)

        pool2 = self.pool2(conv2)
        # print("Pool2 shape: ", pool2.shape)

        flatten = Flatten()(pool2)
        # print("Flatten ", flatten.shape)
        fc1 = self.fc1(flatten)
        fc2 = self.fc2(fc1)
        return self.fc3(fc2)


class FeatureNet(tf.keras.Model):

    def __init__(self, fs=100):
        super(FeatureNet, self).__init__()
        self.fs = fs
        self.small_width = tf.keras.Sequential(
            [
                Conv1D(filters=64, kernel_size=self.fs // 2, strides=self.fs // 16, activation='relu'),
                # weight decay = 1e-3
                MaxPooling1D(pool_size=8, strides=8),
                Dropout(0.5),
                Conv1D(filters=128, kernel_size=8, strides=1, activation='relu'),
                Conv1D(filters=128, kernel_size=8, strides=1, activation='relu'),
                Conv1D(filters=128, kernel_size=8, strides=1, activation='relu'),
                MaxPooling1D(pool_size=4, strides=4),
                Flatten()
            ]
        )

        self.large_width = tf.keras.Sequential(
            [
                Conv1D(filters=64, kernel_size=self.fs * 4, strides=self.fs // 2, activation='relu'),
                MaxPooling1D(pool_size=4, strides=4),
                Dropout(0.5),
                #         layers.Reshape(target_shape=(-1,1)), # ????
                Conv1D(filters=128, kernel_size=6, strides=1, activation='relu'),
                Conv1D(filters=128, kernel_size=6, strides=1, activation='relu'),
                #         layers.Conv1D(filters=128, kernel_size=6, strides=1, activation='relu'), # TODO: if add this layer -> the size would be negative -> not works!
                MaxPooling1D(pool_size=2, strides=2),
                Flatten()
            ]
        )

        self.dropout = Dropout(0.5)

        #         Using for pre-training
        self.fc = Dense(5, activation='softmax')

    def call(self, inputs, pretraining=True, **kwargs):
        out1 = self.small_width(inputs)
        out2 = self.large_width(inputs)
        concat = tf.concat([out1, out2], axis=1)
        out = self.dropout(concat)

        if pretraining:
            out = self.fc(out)

        return out


class SequenceResidualNet(tf.keras.Model):
   
    def __init__(self, representation_model, lstm_size=512):
        super(SequenceResidualNet, self).__init__()
        self.rep_model = representation_model
        self.reshape = Reshape((-1, 1))
        self.two_blstm = tf.keras.Sequential(
            [
                Bidirectional(LSTM(lstm_size, return_sequences=True)),
                Dropout(0.5),
                Bidirectional(LSTM(lstm_size)),
                Dropout(0.5)
            ]
        )
        self.fc1 = Dense(2 * lstm_size, activation='relu')
        self.dropout = Dropout(0.5)
        self.fc2 = Dense(5, activation='softmax')

    def call(self, inputs, **kwargs):
        rep_out = self.rep_model(inputs, pretraining=False)
        rep_reshaped = self.reshape(rep_out)
        lstm_out = self.two_blstm(rep_reshaped)
        rep_fc_out = self.fc1(rep_out)
        sum_out = lstm_out + rep_fc_out
        sum_dropout = self.dropout(sum_out)
        final_out = self.fc2(sum_dropout)

        return final_out
