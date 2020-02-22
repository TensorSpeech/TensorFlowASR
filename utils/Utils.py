from __future__ import absolute_import

import runpy
import tensorflow as tf
from nltk.metrics import distance

conf_options = ["base_model",
                "decoder",
                "batch_size",
                "num_epochs",
                "vocabulary_file_path",
                "learning_rate",
                "sample_rate",
                "frame_ms",
                "stride_ms",
                "num_feature_bins",
                "train_data_transcript_paths",
                "eval_data_transcript_paths",
                "test_data_transcript_paths",
                "checkpoint_dir",
                "log_dir"]


def get_config(config_path):
    conf_dict = runpy.run_path(config_path)
    for option in conf_options:
        if conf_dict.get(option, None) is None:
            raise ValueError(
                '{} has to be defined in the config file'.format(option))
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


def dense_to_sparse(dense_tensor, sequence_length):
    indices = tf.where(tf.sequence_mask(sequence_length))
    values = tf.gather_nd(dense_tensor, indices)
    shape = tf.shape(dense_tensor, out_type=tf.float64)
    return tf.SparseTensor(indices, values, shape)
