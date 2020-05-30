from __future__ import absolute_import

import runpy
import os
import numpy as np
import tensorflow as tf
from configs import DefaultConfig, SeganConfig

asr_conf_required = [
    "base_model", "decoder",
    "vocabulary_file_path",
    "speech_conf", "streaming_size"
]

asr_conf_paths = [
    "train_data_transcript_paths",
    "test_data_transcript_paths",
    "eval_data_transcript_paths",
    "vocabulary_file_path",
    "checkpoint_dir",
    "tfrecords_dir",
    "log_dir"
]

segan_conf_required = [
    "kwidth", "ratio",
    "noise_std", "denoise_epoch",
    "noise_decay", "noise_std_lbound",
    "l1_lambda", "pre_emph",
    "window_size", "sample_rate",
    "stride", "noise_conf"
]

segan_conf_paths = [
    "clean_train_data_dir",
    "noises_dir",
    "clean_test_data_dir",
    "checkpoint_dir",
    "log_dir"
]


def check_key_in_dict(dictionary, keys):
    for key in keys:
        if key not in dictionary.keys():
            raise ValueError("{} must be defined".format(key))


def preprocess_paths(paths):
    if isinstance(paths, list):
        return [os.path.abspath(os.path.expanduser(path)) for path in paths]
    return os.path.abspath(os.path.expanduser(paths))


def get_asr_config(config_path):
    if config_path is None:
        conf_dict = vars(DefaultConfig)
    else:
        conf_dict = runpy.run_path(config_path)
    check_key_in_dict(dictionary=conf_dict, keys=asr_conf_required)
    # fill missing default optional values
    default_dict = vars(DefaultConfig)
    conf_dict = append_default_keys_dict(default_dict, conf_dict)
    # convert paths to take ~/ dir
    for key in asr_conf_paths:
        conf_dict[key] = preprocess_paths(conf_dict[key])

    return conf_dict


def get_segan_config(config_path):
    if config_path is None:
        conf_dict = vars(SeganConfig)
    else:
        conf_dict = runpy.run_path(config_path)
    check_key_in_dict(dictionary=conf_dict, keys=segan_conf_required)
    # fill missing default optional values
    default_dict = vars(SeganConfig)
    conf_dict = append_default_keys_dict(default_dict, conf_dict)
    # convert paths to take ~/ dir
    for key in segan_conf_paths:
        conf_dict[key] = preprocess_paths(conf_dict[key])

    return conf_dict


def append_default_keys_dict(default_dict, dest_dict):
    for key in default_dict.keys():
        if key not in dest_dict.keys():
            dest_dict[key] = default_dict[key]
    return dest_dict


def mask_nan(x):
    x_zeros = tf.zeros_like(x)
    x_mask = tf.math.is_finite(x)
    y = tf.where(x_mask, x, x_zeros)
    return y


def bytes_to_string(array: np.ndarray, encoding: str = "utf-8"):
    return [transcript.decode(encoding) for transcript in array]


def slice_signal(signal, window_size, stride=0.5):
    """ Return windows of the given signal by sweeping in stride fractions
        of window
    """
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    offset = int(window_size * stride)
    slices = []
    for beg_i, end_i in zip(range(0, n_samples, offset),
                            range(window_size, n_samples + offset,
                                  offset)):
        slice_ = signal[beg_i:end_i]
        if slice_.shape[0] < window_size:
            slice_ = np.pad(slice_, (0, window_size - slice_.shape[0]), 'constant', constant_values=0.0)
        if slice_.shape[0] == window_size:
            slices.append(slice_)
    return np.array(slices, dtype=np.float32)


def merge_slices(slices):
    # slices shape = [batch, window_size]
    return tf.keras.backend.flatten(slices)  # return shape = [-1, ]
