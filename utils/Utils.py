from __future__ import absolute_import

import runpy
import os
import tensorflow as tf
from nltk.metrics import distance
from configs import DefaultConfig

conf_required = ["base_model",
                 "decoder",
                 "batch_size",
                 "num_epochs",
                 "vocabulary_file_path",
                 "learning_rate",
                 "min_lr",
                 "sample_rate",
                 "frame_ms",
                 "stride_ms",
                 "num_feature_bins",
                 "checkpoint_dir"]

conf_paths = ["train_data_transcript_paths",
              "test_data_transcript_paths",
              "eval_data_transcript_paths",
              "vocabulary_file_path",
              "checkpoint_dir",
              "log_dir"]


def check_key_in_dict(dictionary, keys):
    for key in keys:
        if dictionary.get(key, None) is None:
            raise ValueError("{} must be defined".format(key))


def preprocess_paths(paths):
    if isinstance(paths, list):
        return list(map(os.path.expanduser, paths))
    return os.path.expanduser(paths)


def get_config(config_path):
    conf_dict = runpy.run_path(config_path)
    check_key_in_dict(dictionary=conf_dict, keys=conf_required)
    # fill missing default optional values
    default_dict = vars(DefaultConfig)
    for key in default_dict.keys():
        if key not in conf_dict.keys():
            conf_dict[key] = default_dict[key]
    # convert paths to take ~/ dir
    for key in conf_paths:
        conf_dict[key] = preprocess_paths(conf_dict[key])

    return conf_dict


def levenshtein(a, b):
    """
    Calculate Levenshtein distance between a and b
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n
    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]


def wer(decode, target):
    words = set(decode.split() + target.split())
    word2char = dict(zip(words, range(len(words))))

    new_decode = [chr(word2char[w]) for w in decode.split()]
    new_target = [chr(word2char[w]) for w in target.split()]

    return distance.edit_distance(''.join(new_decode), ''.join(new_target))


def cer(decode, target):
    return distance.edit_distance(decode, target)


def mask_nan(x):
    x_zeros = tf.zeros_like(x)
    x_mask = tf.math.is_finite(x)
    y = tf.where(x_mask, x, x_zeros)
    return y
