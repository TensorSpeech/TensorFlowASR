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

from typing import List

import tensorflow as tf

from tensorflow_asr.augmentations.methods import gaussnoise, specaugment
from tensorflow_asr.augmentations.methods.base_method import AugmentationMethod

AUGMENTATIONS = {
    "gauss_noise": gaussnoise.GaussNoise,
    "freq_masking": specaugment.FreqMasking,
    "time_masking": specaugment.TimeMasking,
}


class Augmentation:
    def __init__(self, config: dict = None):
        if not config:
            config = {}
        self.signal_augmentations = self.parse(config.pop("signal_augment", {}))
        self.feature_augmentations = self.parse(config.pop("feature_augment", {}))

    def _augment(self, inputs, augmentations: List[AugmentationMethod]):
        outputs = inputs
        for au in augmentations:
            outputs = au.augment(outputs)
            # p = tf.random.uniform(shape=[], dtype=tf.float32)
            # outputs = tf.cond(tf.less(p, au.prob), lambda: au.augment(outputs), lambda: outputs)
        return outputs

    def signal_augment(self, inputs, inputs_length):
        """
        Augment audio signals

        Parameters
        ----------
        inputs : tf.Tensor, shape [B, None]
            Original audio signals
        inputs_length : tf.Tensor, shape [B]
            Original audio signals length

        Returns
        -------
        tf.Tensor, shape [B, None]
            Augmented audio signals
        """
        return tf.map_fn(
            fn=lambda x: self._augment(x, self.signal_augmentations),
            elems=(inputs, inputs_length),
            fn_output_signature=(
                tf.TensorSpec.from_tensor(inputs[0]),
                tf.TensorSpec.from_tensor(inputs_length[0]),
            ),
        )

    def feature_augment(self, inputs, inputs_length):
        """
        Augment audio features

        Parameters
        ----------
        inputs : tf.Tensor, shape [B, T, F]
            Original audio features
        inputs_length : tf.Tensor, shape [B]
            Original audio features length

        Returns
        -------
        tf.Tensor, shape [B, T, F]
            Augmented audio features
        """
        return tf.map_fn(
            fn=lambda x: self._augment(x, self.feature_augmentations),
            elems=(inputs, inputs_length),
            fn_output_signature=(
                tf.TensorSpec.from_tensor(inputs[0]),
                tf.TensorSpec.from_tensor(inputs_length[0]),
            ),
        )

    @staticmethod
    def parse(config: dict) -> list:
        augmentations = []
        for key, value in sorted(config.items(), key=lambda x: x[0]):
            au = AUGMENTATIONS.get(key, None)
            if au is None:
                raise KeyError(f"No tf augmentation named: {key}\n" f"Available tf augmentations: {AUGMENTATIONS.keys()}")
            aug = au(**value) if value is not None else au()
            augmentations.append(aug)
        return augmentations
