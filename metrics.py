import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

import pandas as pd
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, \
    f1_score

from prepare_physionet import known_class_dict


def report(predicted, actual):
    TP = tf.math.count_nonzero(predicted * actual)
    TN = tf.math.count_nonzero((predicted - 1) * (actual - 1))
    FP = tf.math.count_nonzero(predicted * (actual - 1))
    FN = tf.math.count_nonzero((predicted - 1) * actual)
    accuracy = (TP + TN) / len(actual)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)


def check_model(model, data_val, label_val):
    # loss_val, acc_val = model.evaluate(data_val, label_val)
    pred_prob_val = model.predict(data_val)
    pred_val = np.argmax(pred_prob_val, axis=1)

    confusion = pd.DataFrame(tf.math.confusion_matrix(label_val, pred_val).numpy())
    confusion["class"] = list(known_class_dict.values())
    confusion.set_index("class", inplace=True)
    confusion.rename(columns=known_class_dict, inplace=True)

    acc = accuracy_score(label_val, pred_val)
    precision = precision_score(label_val, pred_val, average='macro')
    f1 = f1_score(label_val, pred_val, average='macro')

    return confusion, acc, precision, f1


def cohen_kappa_score_from_confusion_matrix(confusion_matrix):
    """
    Calculate Cohen Kappa score from confusion matrix
    """
    total_element = sum(confusion_matrix.sum(axis=0).values)
    total_class = len(confusion_matrix)
    actual = []
    pred = []

    for actual_label in range(total_class):
        for pred_label in range(total_class):
            n_ele = confusion_matrix.iloc[actual_label, pred_label]
            actual += [actual_label] * n_ele
            pred += [pred_label] * n_ele

    assert len(actual) == total_element
    assert len(pred) == total_element

    return cohen_kappa_score(actual, pred)
