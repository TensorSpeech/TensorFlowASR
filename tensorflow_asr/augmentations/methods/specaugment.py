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

from dataclasses import asdict, dataclass

import tensorflow as tf

from tensorflow_asr.augmentations.methods.base_method import AugmentationMethod
from tensorflow_asr.utils import shape_util


@dataclass
class MASK_VALUES:
    MEAN: str = "mean"
    MIN: str = "min"
    MAX: str = "max"
    ZERO: str = "zero"


def get_mask_value(inputs: tf.Tensor, mask_value=MASK_VALUES.ZERO):
    if isinstance(mask_value, (int, float)):
        return tf.constant(mask_value, dtype=inputs.dtype)
    if mask_value == MASK_VALUES.MEAN:
        return tf.reduce_mean(inputs)
    if mask_value == MASK_VALUES.MIN:
        return tf.reduce_min(inputs)
    if mask_value == MASK_VALUES.MAX:
        return tf.reduce_max(inputs)
    return tf.constant(0, dtype=inputs.dtype)  # default zero


class FreqMasking(AugmentationMethod):
    def __init__(
        self,
        num_masks: int = 1,
        mask_factor: float = 27,
        prob: float = 1.0,
        mask_value="zero",
    ):
        super().__init__(prob=prob)
        self.num_masks = num_masks
        self.mask_factor = mask_factor
        self.mask_value = mask_value
        if self.mask_value not in asdict(MASK_VALUES()).values():
            if not isinstance(self.mask_value, (int, float)):
                raise ValueError(f"mask_value must in {asdict(MASK_VALUES()).values()} or a number")

    def augment(self, args):
        """
        Masking the frequency channels (shape[1])

        Parameters
        ----------
        spectrogram : tf.Tensor, shape [T, num_feature_bins] or [T, num_feature_bins, 1]
            Audio features

        Returns
        -------
        tf.Tensor, shape [T, num_feature_bins] or [T, num_feature_bins, 1]
            Masked frequency dim of audio features
        """
        with tf.name_scope("freq_masking_specaugment"):
            spectrogram, spectrogram_length = args
            _, frequency_length, *rest = shape_util.shape_list(spectrogram, out_type=tf.int32)
            indices_shape = (1, -1) + (1,) * len(rest)
            mval = get_mask_value(spectrogram, mask_value=self.mask_value)
            F = tf.convert_to_tensor(self.mask_factor, dtype=tf.int32)
            for _ in range(self.num_masks):
                prob = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
                do_apply = tf.where(tf.less_equal(prob, self.prob), tf.constant(1, tf.int32), tf.constant(0, tf.int32))
                f = tf.random.uniform(shape=[], minval=0, maxval=F, dtype=tf.int32)
                f = do_apply * tf.minimum(f, frequency_length)
                f0 = do_apply * tf.random.uniform(shape=[], minval=0, maxval=(frequency_length - f), dtype=tf.int32)
                indices = tf.reshape(tf.range(frequency_length), indices_shape)
                condition = tf.math.logical_and(tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f))
                spectrogram = tf.where(condition, mval, spectrogram)
            return spectrogram, spectrogram_length


class TimeMasking(AugmentationMethod):
    def __init__(
        self,
        num_masks: int = 1,
        mask_factor: float = 100,
        p_upperbound: float = 1.0,
        prob: float = 1.0,
        mask_value: str = "zero",
    ):
        super().__init__(prob=prob)
        self.num_masks = num_masks
        self.mask_factor = mask_factor
        self.p_upperbound = p_upperbound
        self.mask_value = mask_value
        if self.mask_value not in asdict(MASK_VALUES()).values():
            if not isinstance(self.mask_value, (int, float)):
                raise ValueError(f"mask_value must in {asdict(MASK_VALUES()).values()} or a number")

    def augment(self, args):
        """
        Masking the time channel (shape[0])

        Parameters
        ----------
        spectrogram : tf.Tensor, shape [T, num_feature_bins] or [T, num_feature_bins, 1]
            Audio features

        Returns
        -------
        tf.Tensor, shape [T, num_feature_bins] or [T, num_feature_bins, 1]
            Masked time dim of audio features
        """
        with tf.name_scope("time_masking_specaugment"):
            spectrogram, spectrogram_length = args
            max_length, *rest = shape_util.shape_list(spectrogram, out_type=tf.int32)
            indices_shape = (-1,) + (1,) * len(rest)
            mval = get_mask_value(spectrogram, mask_value=self.mask_value)
            T = tf.cast(tf.floor(tf.cast(spectrogram_length, dtype=spectrogram.dtype) * self.p_upperbound), dtype=tf.int32)
            for _ in range(self.num_masks):
                prob = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
                do_apply = tf.where(tf.less_equal(prob, self.prob), tf.constant(1, tf.int32), tf.constant(0, tf.int32))
                t = tf.random.uniform(shape=[], minval=0, maxval=T, dtype=tf.int32)
                t = do_apply * tf.minimum(t, spectrogram_length)
                t0 = do_apply * tf.random.uniform(shape=[], minval=0, maxval=(spectrogram_length - t), dtype=tf.int32)
                indices = tf.reshape(tf.range(max_length), indices_shape)
                condition = tf.math.logical_and(tf.math.greater_equal(indices, t0), tf.math.less(indices, t0 + t))
                spectrogram = tf.where(condition, mval, spectrogram)
            return spectrogram, spectrogram_length
