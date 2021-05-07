import tensorflow as tf
import math
import numpy as np

class ConsecutiveSampleGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size, width=2):
        self.x = x
        self.y = y
        self.width = width
        self.raw_len = len(x)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil((self.raw_len - 2 * self.width - 1) / self.batch_size)

    def __getitem__(self, batch_id):
        from_idx = self.batch_size * batch_id
        to_idx = min(self.raw_len - 2 * self.width - 1, from_idx + self.batch_size)
        return self.__data_generation(range(from_idx, to_idx))

    def __data_generation(self, list_IDs):
        x_batch = []
        y_batch = []
        for idx in list_IDs:
            item = self.x[idx:idx + 2 * self.width + 1].reshape(-1, 1)
            x_batch += [item]
            y_batch += [self.y[idx + self.width]]

        return np.array(x_batch), np.array(y_batch)