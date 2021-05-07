import numpy as np
from prepare_physionet import class_dict
import os

DEBUG = None


def list_files(data_dir):
    """
    List all files in `data_dir`
    """
    try:
        all_files = os.listdir(data_dir)
        all_files = ["{}/{}".format(data_dir, file) for file in all_files]
        return all_files
    except:
        print("Cannot list directory. Check if the input '%s' is correct" % data_dir)
        return []


def load_npz_file(npz_file):
    """Load data and labels from a npz file."""
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
    return data, labels, sampling_rate


# Reference: NonSeqDataLoader of DeepSleepNet
def load_npz_list_files(npz_files):
    """Load data and labels from list of npz files."""
    data = []
    labels = []
    fs = None
    for npz_f in npz_files:
        if DEBUG:
            print("Loading {} ...".format(npz_f))
        if not npz_f.endswith(".npz"):
            continue
        tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")

        # Reshape the data to match the input of the model - conv2d
        tmp_data = np.squeeze(tmp_data)
        tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

        # # Reshape the data to match the input of the model - conv1d
        # tmp_data = tmp_data[:, :, np.newaxis]

        # Casting
        tmp_data = tmp_data.astype(np.float32)
        tmp_labels = tmp_labels.astype(np.int32)

        data.append(tmp_data)
        labels.append(tmp_labels)
    data = np.vstack(data)
    labels = np.hstack(labels)

    return data, labels


def split_train_val(fold_idx, all_files, total_fold=20):
    """Extract training set and test set
    fold_idx: id of current fold, start fom 0
    all_files: list of all files containing data
    total_fold: number of fold
    """
    all_files.sort()
    total_file = len(all_files)
    fold_size = total_file // total_fold
    from_id = fold_size * fold_idx
    to_id = from_id + fold_size if fold_idx < total_fold - 1 else total_file

    # print("Fold #{} from {} to {}".format(fold_idx, from_id, to_id))

    validation_files = all_files[from_id:to_id]
    train_files = list(set(all_files) - set(validation_files))

    data_train, label_train = load_npz_list_files(train_files)
    data_val, label_val = load_npz_list_files(validation_files)

    return (data_train, label_train), (data_val, label_val)


def get_balance_class_oversample(x, y):
    """
    Balance the number of samples of all classes by (oversampling):
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    Reference: Utils.py of DeepSleepNet
    """
    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y


def shuffle(data, label):
    """
    Shuffle data and label
    """
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx]
    label = label[idx]
    return data, label

def print_n_samples_each_class(labels):
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print("{}: {}".format(class_dict[c], n_samples))


def info(data, label):
    print("Data set: {}, {}".format(data.shape, label.shape))
    print_n_samples_each_class(label)
    print(" ")