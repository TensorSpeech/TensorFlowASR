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

import math
import numpy as np
import tensorflow as tf

from . import shape_util


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def get_num_batches(nsamples, batch_size, drop_remainders=True):
    if nsamples is None or batch_size is None: return None
    if drop_remainders: return math.floor(float(nsamples) / float(batch_size))
    return math.ceil(float(nsamples) / float(batch_size))


def nan_to_zero(input_tensor):
    return tf.where(tf.math.is_nan(input_tensor), tf.zeros_like(input_tensor), input_tensor)


def bytes_to_string(array: np.ndarray, encoding: str = "utf-8"):
    if array is None: return None
    return [transcript.decode(encoding) for transcript in array]


def get_reduced_length(length, reduction_factor):
    return tf.cast(tf.math.ceil(tf.divide(length, tf.cast(reduction_factor, dtype=length.dtype))), dtype=tf.int32)


def count_non_blank(tensor: tf.Tensor, blank: int or tf.Tensor = 0, axis=None):
    return tf.reduce_sum(tf.where(tf.not_equal(tensor, blank), x=tf.ones_like(tensor), y=tf.zeros_like(tensor)), axis=axis)


def merge_two_last_dims(x):
    b, _, f, c = shape_util.shape_list(x)
    return tf.reshape(x, shape=[b, -1, f * c])


def merge_repeated(yseqs, blank=0):
    result = tf.reshape(yseqs[0], [1])

    U = shape_util.shape_list(yseqs)[0]
    i = tf.constant(1, dtype=tf.int32)

    def _cond(i, result, yseqs, U): return tf.less(i, U)

    def _body(i, result, yseqs, U):
        if yseqs[i] != result[-1]:
            result = tf.concat([result, [yseqs[i]]], axis=-1)
        return i + 1, result, yseqs, U

    _, result, _, _ = tf.while_loop(
        _cond,
        _body,
        loop_vars=[i, result, yseqs, U],
        shape_invariants=(
            tf.TensorShape([]),
            tf.TensorShape([None]),
            tf.TensorShape([None]),
            tf.TensorShape([])
        )
    )

    return tf.pad(result, [[U - shape_util.shape_list(result)[0], 0]], constant_values=blank)


def find_max_length_prediction_tfarray(tfarray: tf.TensorArray) -> tf.Tensor:
    with tf.name_scope("find_max_length_prediction_tfarray"):
        index = tf.constant(0, dtype=tf.int32)
        total = tfarray.size()
        max_length = tf.constant(0, dtype=tf.int32)

        def condition(index, _): return tf.less(index, total)

        def body(index, max_length):
            prediction = tfarray.read(index)
            length = tf.shape(prediction)[0]
            max_length = tf.where(tf.greater(length, max_length), length, max_length)
            return index + 1, max_length

        index, max_length = tf.while_loop(condition, body, loop_vars=[index, max_length], swap_memory=False)
        return max_length


def pad_prediction_tfarray(tfarray: tf.TensorArray, blank: int or tf.Tensor) -> tf.TensorArray:
    with tf.name_scope("pad_prediction_tfarray"):
        index = tf.constant(0, dtype=tf.int32)
        total = tfarray.size()
        max_length = find_max_length_prediction_tfarray(tfarray)

        def condition(index, _): return tf.less(index, total)

        def body(index, tfarray):
            prediction = tfarray.read(index)
            prediction = tf.pad(
                prediction, paddings=[[0, max_length - tf.shape(prediction)[0]]],
                mode="CONSTANT", constant_values=blank
            )
            tfarray = tfarray.write(index, prediction)
            return index + 1, tfarray

        index, tfarray = tf.while_loop(condition, body, loop_vars=[index, tfarray], swap_memory=False)
        return tfarray
