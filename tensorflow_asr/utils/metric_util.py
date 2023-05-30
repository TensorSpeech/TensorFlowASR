# Copyright 2020 Huy Le Nguyen (@nglehuy)
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

from typing import Tuple

import tensorflow as tf
from nltk.metrics import distance

from tensorflow_asr.utils import math_util


def execute_wer(
    decode,
    target,
):
    decode = math_util.bytes_to_string(decode)
    target = math_util.bytes_to_string(target)
    dis = 0.0
    length = 0.0
    for dec, tar in zip(decode, target):
        words = set(dec.split() + tar.split())
        word2char = dict(zip(words, range(len(words))))

        new_decode = [chr(word2char[w]) for w in dec.split()]
        new_target = [chr(word2char[w]) for w in tar.split()]

        dis += distance.edit_distance("".join(new_decode), "".join(new_target))
        length += len(tar.split())
    return tf.convert_to_tensor(dis, tf.float32), tf.convert_to_tensor(length, tf.float32)


def wer(
    decode: tf.Tensor,
    target: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Word Error Rate

    Args:
        decode (np.ndarray): array of prediction texts
        target (np.ndarray): array of groundtruth texts

    Returns:
        tuple: a tuple of tf.Tensor of (edit distances, number of words) of each text
    """
    return tf.numpy_function(execute_wer, inp=[decode, target], Tout=[tf.float32, tf.float32])


def execute_cer(decode, target):
    decode = math_util.bytes_to_string(decode)
    target = math_util.bytes_to_string(target)
    dis = 0
    length = 0
    for dec, tar in zip(decode, target):
        dis += distance.edit_distance(dec, tar)
        length += len(tar)
    return tf.convert_to_tensor(dis, tf.float32), tf.convert_to_tensor(length, tf.float32)


def cer(
    decode: tf.Tensor,
    target: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Character Error Rate

    Args:
        decode (np.ndarray): array of prediction texts
        target (np.ndarray): array of groundtruth texts

    Returns:
        tuple: a tuple of tf.Tensor of (edit distances, number of characters) of each text
    """
    return tf.numpy_function(execute_cer, inp=[decode, target], Tout=[tf.float32, tf.float32])


def tf_wer(
    decode: tf.Tensor,
    target: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Tensorflow Word Error Rate

    Args:
        decode (tf.Tensor): tensor shape [B]
        target (tf.Tensor): tensor shape [B]

    Returns:
        tuple: a tuple of tf.Tensor of (edit distances, number of words) of each text
    """
    decode = tf.strings.split(decode)
    target = tf.strings.split(target)
    distances = tf.edit_distance(decode.to_sparse(), target.to_sparse(), normalize=False)  # [B]
    lengths = tf.cast(target.row_lengths(axis=1), dtype=tf.float32)  # [B]
    return tf.reduce_sum(distances), tf.reduce_sum(lengths)


def tf_cer(
    decode: tf.Tensor,
    target: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Tensorflow Charactor Error rate

    Args:
        decoder (tf.Tensor): tensor shape [B]
        target (tf.Tensor): tensor shape [B]

    Returns:
        tuple: a tuple of tf.Tensor of (edit distances, number of characters) of each text
    """
    decode = tf.strings.bytes_split(decode)  # [B, N]
    target = tf.strings.bytes_split(target)  # [B, M]
    distances = tf.edit_distance(decode.to_sparse(), target.to_sparse(), normalize=False)  # [B]
    lengths = tf.cast(target.row_lengths(axis=1), dtype=tf.float32)  # [B]
    return tf.reduce_sum(distances), tf.reduce_sum(lengths)
