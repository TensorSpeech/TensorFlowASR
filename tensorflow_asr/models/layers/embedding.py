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

from tensorflow_asr.models.base_layer import Layer


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

    def call(self, inputs):
        outputs, outputs_length = inputs
        outputs = super().call(outputs)
        return outputs, outputs_length

    def call_next(self, inputs):
        outputs = tf.cast(tf.expand_dims(inputs, axis=-1), dtype=tf.int32)
        return tf.gather_nd(self.embeddings, outputs)  # https://github.com/tensorflow/tensorflow/issues/42410

    def compute_mask(self, inputs, mask=None):
        outputs, outputs_length = inputs
        mask = tf.sequence_mask(outputs_length, maxlen=tf.shape(outputs)[1], dtype=tf.bool)
        return mask, None

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = super().compute_output_shape(output_shape)
        return output_shape, output_length_shape


class OneHotBlank(Layer):
    """
    https://arxiv.org/pdf/1211.3711.pdf
    The inputs are encoded as one-hot vectors;
    that is, if Y consists of K labels and yu = k, then y^u is a length K vector whose elements are all zero
    except the k-th, which is one. ∅ is encoded as a length K vector of zeros
    """

    def __init__(self, blank, depth, name="one_hot_blank", **kwargs):
        super().__init__(name=name, **kwargs)
        self.blank = blank
        self.depth = depth

    def call(self, inputs):
        outputs, outputs_length = inputs
        minus_one_at_blank = tf.where(tf.equal(outputs, self.blank), -1, outputs)
        outputs = tf.one_hot(minus_one_at_blank, depth=self.depth, dtype=self.dtype)
        return outputs, outputs_length

    def call_next(self, inputs):
        outputs, _ = self.call((inputs, None))
        return outputs

    def compute_mask(self, inputs, mask=None):
        outputs, outputs_length = inputs
        mask = tf.sequence_mask(outputs_length, maxlen=tf.shape(outputs)[1], dtype=tf.bool)
        return mask, None

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        output_shape = output_shape + (self.depth,)
        return output_shape, output_length_shape
