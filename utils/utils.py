# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import

import os
import sys
import math
import numpy as np
import tensorflow as tf


def float_feature(list_of_floats):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def int64_feature(list_of_ints):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def append_default_keys_dict(default_dict, dest_dict):
    for key in default_dict.keys():
        if key not in dest_dict.keys():
            dest_dict[key] = default_dict[key]
    return dest_dict


def check_key_in_dict(dictionary, keys):
    for key in keys:
        if key not in dictionary.keys():
            raise ValueError("{} must be defined".format(key))


def preprocess_paths(paths):
    if isinstance(paths, list):
        return [os.path.abspath(os.path.expanduser(path)) for path in paths]
    return os.path.abspath(os.path.expanduser(paths)) if paths else None


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


def get_steps_per_epoch(samples, batch_size):
    return math.ceil(samples / batch_size)


def update_total(tqdm_bar, total):
    tqdm_bar.total = total
    tqdm_bar.refresh()


def print_one_line(*args, **kwargs):
    sys.stdout.write("\033[K")
    tf.print("\r", *args, **kwargs, end=" ")
    sys.stdout.flush()
