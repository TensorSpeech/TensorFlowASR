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


class Embedding(tf.keras.layers.Layer):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 contraint=None,
                 regularizer=None,
                 initializer=None,
                 **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.contraint = tf.keras.constraints.get(contraint)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings", dtype=tf.float32,
            shape=[self.vocab_size, self.embed_dim],
            initializer=self.initializer,
            trainable=True, regularizer=self.regularizer,
            constraint=self.contraint
        )
        self.built = True

    def call(self, inputs):
        outputs = tf.cast(tf.expand_dims(inputs, axis=-1), dtype=tf.int32)
        return tf.gather_nd(self.embeddings, outputs)

    def get_config(self):
        conf = super(Embedding, self).get_config()
        conf.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "contraint": self.contraint,
            "regularizer": self.regularizer,
            "initializer": self.initializer
        })
        return conf
