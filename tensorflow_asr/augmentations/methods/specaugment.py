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

import tensorflow as tf

from tensorflow_asr.augmentations.methods.base_method import AugmentationMethod
from tensorflow_asr.utils import shape_util

MASK_VALUES = ["mean", "min", "max", "zero"]


class FreqMasking(AugmentationMethod):
    def __init__(
        self,
        num_masks: int = 1,
        mask_factor: float = 27,
        prob: float = 1.0,
        mask_value: str = "mean",
    ):
        super().__init__(prob=prob)
        self.num_masks = num_masks
        self.mask_factor = mask_factor
        self.mask_value = mask_value
        if self.mask_value not in MASK_VALUES:
            raise ValueError(f"mask_value must in {MASK_VALUES}")

    @tf.function
    def augment(self, spectrogram: tf.Tensor):
        """
        Masking the frequency channels (shape[1])
        Args:
            spectrogram: shape (T, num_feature_bins, V)
        Returns:
            frequency masked spectrogram
        """
        _, F, _ = shape_util.shape_list(spectrogram, out_type=tf.int32)
        if self.mask_value == "mean":
            mval = tf.reduce_mean(spectrogram)
        elif self.mask_value == "min":
            mval = tf.reduce_min(spectrogram)
        elif self.mask_value == "max":
            mval = tf.reduce_max(spectrogram)
        elif self.mask_value == "zero":
            mval = tf.constant(0, dtype=spectrogram.dtype)
        for _ in range(self.num_masks):
            f = tf.random.uniform([], minval=0, maxval=self.mask_factor, dtype=tf.dtypes.int32)
            f = tf.minimum(f, F)
            f0 = tf.random.uniform([], minval=0, maxval=F - f, dtype=tf.dtypes.int32)
            indices = tf.reshape(tf.range(F), (1, -1, 1))
            condition = tf.math.logical_and(tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f))
            spectrogram = tf.where(condition, mval, spectrogram)
        return spectrogram


class TimeMasking(AugmentationMethod):
    def __init__(
        self,
        num_masks: int = 1,
        mask_factor: float = 100,
        p_upperbound: float = 1.0,
        prob: float = 1.0,
        mask_value: str = "mean",
    ):
        super().__init__(prob=prob)
        self.num_masks = num_masks
        self.mask_factor = mask_factor
        self.p_upperbound = p_upperbound
        self.mask_value = mask_value
        if self.mask_value not in MASK_VALUES:
            raise ValueError(f"mask_value must in {MASK_VALUES}")

    @tf.function
    def augment(self, spectrogram: tf.Tensor):
        """
        Masking the time channel (shape[0])
        Args:
            spectrogram: shape (T, num_feature_bins, V)
        Returns:
            frequency masked spectrogram
        """
        T, _, _ = shape_util.shape_list(spectrogram, out_type=tf.int32)
        if self.mask_value == "mean":
            mval = tf.reduce_mean(spectrogram)
        elif self.mask_value == "min":
            mval = tf.reduce_min(spectrogram)
        elif self.mask_value == "max":
            mval = tf.reduce_max(spectrogram)
        elif self.mask_value == "zero":
            mval = tf.constant(0, dtype=spectrogram.dtype)
        for _ in range(self.num_masks):
            t = tf.random.uniform([], minval=0, maxval=self.mask_factor, dtype=tf.int32)
            t = tf.minimum(t, tf.cast(tf.cast(T, dtype=tf.float32) * self.p_upperbound, dtype=tf.int32))
            t0 = tf.random.uniform([], minval=0, maxval=(T - t), dtype=tf.int32)
            indices = tf.reshape(tf.range(T), (-1, 1, 1))
            condition = tf.math.logical_and(tf.math.greater_equal(indices, t0), tf.math.less(indices, t0 + t))
            spectrogram = tf.where(condition, mval, spectrogram)
        return spectrogram
