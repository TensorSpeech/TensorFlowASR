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

import tensorflow as tf


class Embedding(tf.keras.layers.Embedding):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        initializer="uniform",
        regularizer=None,
        contraint=None,
        **kwargs,
    ):
        super().__init__(
            input_dim=vocab_size,
            output_dim=embed_dim,
            embeddings_initializer=initializer,
            embeddings_regularizer=regularizer,
            embeddings_constraint=contraint,
            mask_zero=False,
            **kwargs,
        )
        self.supports_masking = True

    def recognize_tflite(self, inputs):
        outputs = tf.cast(tf.expand_dims(inputs, axis=-1), dtype=tf.int32)
        return tf.gather_nd(self.embeddings, outputs)  # https://github.com/tensorflow/tensorflow/issues/42410
