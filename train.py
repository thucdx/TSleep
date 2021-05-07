import argparse
import os
import logging
from models import ModifiedSleepEEGNet, FeatureNet, SequenceResidualNet
import tensorflow as tf
from preprocessing import *
from metrics import *

DEBUG = True

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    print("Could not use GPU!!!")


def debug_f(f, *args):
    if DEBUG:
        f(*args)


def print_debug(text):
    debug_f(print, text)


def train_modified_deep_sleep_net(fold_ids, data_files, total_fold, output_dir):
    # model_name = "modified_sleep_eeg"
    if len(fold_ids) == 0:
        # all folds
        fold_ids = range(0, total_fold)

    all_fold_result = []

    for fold_idx in fold_ids:
        if 0 <= fold_idx < total_fold:
            print("### FOLD %d ###" % fold_idx)

            print_debug("Extract dataset, testset for fold %d" % fold_idx)
            train, test = split_train_val(fold_idx, data_files, total_fold)
            data_train, label_train = train
            data_test, label_test = test

            # Oversampling
            print_debug("Oversampling")
            os_train, os_label = get_balance_class_oversample(data_train, label_train)

            # Shuffling training set
            print_debug("Shuffle training data")
            os_train, os_label = shuffle(os_train, os_label)
            # debug_f(info, os_train, os_label)

            print_debug("First phase training")
            fn_fold = FeatureNet()
            fn_fold.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )
            fn_fold_cp = tf.keras.callbacks.ModelCheckpoint(
                filepath="%s/1_cp_fold_%d" % (output_dir, fold_idx),
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
                save_best_only=True
            )

            fn_fold_cbs = [
                tf.keras.callbacks.TensorBoard(log_dir="%s/1_fold_%d" % (output_dir, fold_idx)),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', min_delta=0, patience=3,
                    mode='auto', baseline=None
                ),
                fn_fold_cp
            ]

            # Fit
            if len(os_train.shape) > 3:
                os_train = os_train.squeeze(-1)
            if len(data_test.shape) > 3:
                data_test = data_test.squeeze(-1)

            fn_fold.fit(os_train, os_label,
                        epochs=30, # FIXME: change back to 30 before submit!!!
                        validation_split=0.4,
                        batch_size=128,
                        callbacks=fn_fold_cbs)

            # Save first-phase model
            first_phase_save_path = "%s/1_model_fold_%d" % (output_dir ,fold_idx)
            print_debug("Save first phase model to %s" % first_phase_save_path)
            fn_fold.save(first_phase_save_path)

            # Second-phase training
            fn_fold.trainable = False
            srn_fold = SequenceResidualNet(fn_fold, lstm_size=128)
            srn_fold.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )

            srn_fold_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath="%s/2_cp_fold_%d" % (output_dir, fold_idx),
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
                save_best_only=True
            )

            srn_fold_cbs = [
                tf.keras.callbacks.TensorBoard(log_dir="%s/2_fold_%d" % (output_dir, fold_idx)),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', min_delta=0, patience=2,
                    mode='auto', baseline=None
                ),
                srn_fold_checkpoint
            ]

            print_debug("Second phase training")
            srn_fold.fit(data_train, # train on sequential data, not the oversampled one
                         label_train,
                         epochs=3, # FIXME: change to 3 before submit
                         validation_split=0.4,
                         shuffle=False,
                         batch_size=128,
                         callbacks=srn_fold_cbs)

            # Save learn model
            save_path = "%s/model_fold_%d" % (output_dir, fold_idx)
            print_debug("Saving model to %s" % save_path)
            srn_fold.save(save_path)

            # Evaluate
            confusion, acc, prec, f1 = check_model(srn_fold, data_test, label_test)
            kappa_score = cohen_kappa_score_from_confusion_matrix(confusion)

            print_debug("EVALUATE FOR FOLD %d" % fold_idx)
            print_single_result((confusion, acc, f1, kappa_score))

            # Store result
            all_fold_result += [(confusion, acc, prec, f1, kappa_score)]
            try:
                with open("%s/result_fold_%d" % (output_dir, fold_idx), "wb") as f:
                    result_tup = np.array([confusion, acc, prec, f1, kappa_score])
                    np.save(f, result_tup)
            except:
                print("Error during saving result for fold %d" % fold_idx)
    # Save final result
    final_result = np.array(all_fold_result)
    try:
        with open("%s/final_result" % output_dir, "wb") as f:
            np.save(f, final_result)
    except:
        print("Error during saving final result")

    print_summary_result(all_fold_result)


def train_modified_sleep_eeg_net(fold_ids, data_files, total_fold, output_dir):
    # model_name = "modified_sleep_eeg"
    if len(fold_ids) == 0:
        # all folds
        fold_ids = range(0, total_fold)

    all_fold_result = []

    for fold_idx in fold_ids:
        if 0 <= fold_idx < total_fold:
            print("### FOLD %d ###" % fold_idx)

            print_debug("Extract dataset, testset for fold %d" % fold_idx)
            train, test = split_train_val(fold_idx, data_files, total_fold)
            data_train, label_train = train
            data_test, label_test = test

            # Oversampling
            print_debug("Oversampling")
            os_train, os_label = get_balance_class_oversample(data_train, label_train)

            # Shuffling training set
            print_debug("Shuffle training data")
            os_train, os_label = shuffle(os_train, os_label)
            # debug_f(info, os_train, os_label)

            if len(os_train.shape) > 3:
                os_train = os_train.squeeze(-1)
            if len(data_test.shape) > 3:
                data_test = data_test.squeeze(-1)

            debug_f(info, os_train, os_label)

            # Initialize model
            modifiedSleepEEGModel = ModifiedSleepEEGNet()
            opt = tf.keras.optimizers.Adam(1e-3)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
            modifiedSleepEEGModel.compile(opt, loss, metrics)

            callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir="%s/fold_%d" % (output_dir, fold_idx)),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", min_delta=0, patience=3,
                    mode="auto", baseline=None
                )
            ]

            # Training
            print_debug("Training")
            modifiedSleepEEGModel.fit(os_train,
                                      os_label,
                                      epochs=15,  # FIXME: Change to 15 before submit
                                      batch_size=256, # Consider change it to lower batch_size if the GPU has litle memory
                                      validation_split=0.4,
                                      callbacks=callbacks)

            # Save learn model for this fold
            save_path = "%s/model_fold_%d" % (output_dir, fold_idx)
            print_debug("Saving model to %s" % save_path)
            modifiedSleepEEGModel.save(save_path)

            # Evaluate
            confusion, acc, prec, f1 = check_model(modifiedSleepEEGModel, data_test, label_test)
            kappa_score = cohen_kappa_score_from_confusion_matrix(confusion)

            print_debug("EVALUATE FOR FOLD %d" % fold_idx)
            print_single_result((confusion, acc, f1, kappa_score))

            # Store result
            all_fold_result += [(confusion, acc, prec, f1, kappa_score)]
            try:
                with open("%s/result_fold_%d" % (output_dir, fold_idx), "wb") as f:
                    result_tup = np.array([confusion, acc, prec, f1, kappa_score])
                    np.save(f, result_tup)
            except:
                print("Error during saving result for fold %d" % fold_idx)

    # Save final result
    final_result = np.array(all_fold_result)
    try:
        with open("%s/final_result" % output_dir, "wb") as f:
            np.save(f, final_result)
    except:
        print("Error during saving final result")

    print_summary_result(all_fold_result)


def print_summary_result(all_fold_result):
    # Display overall stats
    acc_all = [a[1] for a in all_fold_result]
    f1_all = [a[3] for a in all_fold_result]
    mean_acc = sum(acc_all) / len(acc_all)
    mean_f1 = sum(f1_all) / len(f1_all)

    # Confusion
    all_confusion = None
    for a in all_fold_result:
        confusion = a[0]
        if all_confusion is None:
            all_confusion = confusion
        else:
            all_confusion += confusion

    kappa_all = cohen_kappa_score_from_confusion_matrix(all_confusion)
    print("####################################")
    print("RESULT OVER %d FOLDS" % len(acc_all))
    print_single_result((all_confusion, mean_acc, mean_f1, kappa_all))


def print_single_result(single_result):
    confusion, acc, f1, kappa = single_result
    print("ACCURACY: %.3f" % acc)
    print("F1: %.3f" % f1)
    print("COHEN'S KAPPA SCORE: %.3f" % kappa)
    print("CONFUSION MATRIX")
    print(confusion)

def load_and_summarize(result_dir):
    result = np.load("%s/final_result" % result_dir, allow_pickle=True)
    print_summary_result(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="data/sc_eeg_fpz_cz",
                        help="Directory contain extracted channels")
    parser.add_argument("--output_dir", type=str,
                        default="output/sc_eeg_fpz_cz",
                        help="Directory to store model, progress")
    parser.add_argument("--fold_ids", type=str,
                        default="0",
                        help="Fold/Folds to train, each valid fold from [0 to total_fold-1]."
                             "\n- Can contain multiple folds, separate by comma `,`, for example: 0,1,2."
                             "\n- -1 to train on all folds")
    parser.add_argument("--model", type=str,
                        default="mod_sleep_eeg",
                        help="mod_sleep_eeg: Modified SleepEEG - using CNNS"
                             "\nmod_deep_sleep: Modified DeepSleep - using CNNs + Bi-LSTMs")
    parser.add_argument("--total_fold", type=int,
                        default=20,
                        help="Number of fold")

    args = parser.parse_args()
    all_files = list_files(args.data_dir)
    print("args: ", args)
    fold_ids = [int(x) for x in args.fold_ids.split(",")]
    fold_ids = list(set(fold_ids))
    if -1 in fold_ids:
        fold_ids = []
    if args.model == "mod_sleep_eeg":
        print("Training and evaluate result for ModifiedSleepEEG Net")
        train_modified_sleep_eeg_net(fold_ids, all_files, args.total_fold, args.output_dir + "/modified_sleep_eeg")
    elif args.model == "mod_deep_sleep":
        print("Training and evaluate result for ModifiedDeepSleep")
        train_modified_deep_sleep_net(fold_ids, all_files, args.total_fold, args.output_dir + "/modified_deep_sleep")
    else:
        print("Unknown model. Valid model is : mod_sleep_eeg / mod_deep_sleep")


    # load_and_summarize(args.output_dir + "/modified_sleep_eeg")
    # load_and_summarize(args.output_dir + "/modified_deep_sleep")
    # train_modified_sleep_eeg_net([17], all_files, 20, args.output_dir + "/modified_sleep_eeg")
    # train_modified_deep_sleep_net([17], all_files, 20, args.output_dir + "/modified_deep_sleep")
