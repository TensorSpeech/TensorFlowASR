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
from ...utils.utils import shape_list


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,
                 head_size: int,
                 num_heads: int,
                 output_size: int = None,
                 dropout: float = 0.0,
                 name: str = "mha",
                 **kwargs):
        super(MultiHeadAttention, self).__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.all_head_size = self.head_size * self.num_heads
        self.output_size = output_size
        self.dropout = dropout

    def transpose_for_scores(self, x, batch_size):
        """Transpose to calculate attention scores."""
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.head_size])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def build(self, input_shape):
        _, _, value_shape = input_shape
        self.query = tf.keras.layers.Dense(self.all_head_size, name="query")
        self.key = tf.keras.layers.Dense(self.all_head_size, name="key")
        self.value = tf.keras.layers.Dense(self.all_head_size, name="value")
        self.dropout = tf.keras.layers.Dropout(self.dropout, name="dropout")
        if self.output_size is None: self.output_size = value_shape[-1]  # value dim
        self.wo = tf.keras.layers.Dense(self.output_size, name="wo")

    def call(self, inputs, training=False):
        query, key, value = inputs

        batch_size = shape_list(query)[0]
        mixed_query_layer = self.query(query, training=training)
        mixed_key_layer = self.key(key, training=training)
        mixed_value_layer = self.value(value, training=training)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(tf.shape(key_layer)[-1], attention_scores.dtype)  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, [batch_size, -1, self.all_head_size])

        return self.wo(context_layer, training=training)

    def get_config(self):
        conf = super(MultiHeadAttention, self).get_config()
        conf.update({
            "head_size": self.head_size,
            "num_heads": self.num_heads,
            "output_size": self.output_size,
            "dropout": self.dropout
        })
        conf.update(self.query.get_config())
        conf.update(self.key.get_config())
        conf.update(self.value.get_config())
        conf.update(self.dropout.get_config())
        conf.update(self.wo.get_config())
        return conf
