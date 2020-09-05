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

import typing
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,
                 num_heads,
                 head_size,
                 output_size: int = None,
                 dropout: float = 0.0,
                 kernel_initializer: typing.Union[str, typing.Callable] = "glorot_uniform",
                 kernel_regularizer: typing.Union[str, typing.Callable] = None,
                 kernel_constraint: typing.Union[str, typing.Callable] = None,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)

        self.num_heads = num_heads
        self.head_size = head_size
        self.all_head_size = self.head_size * self.num_heads
        self.output_size = output_size
        self.dropout = dropout

    def build_qkvo(self, value_shape):
        self.query = tf.keras.layers.Dense(
            self.all_head_size, name="query", use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_constraint=self.kernel_constraint,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size, name="key", use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_constraint=self.kernel_constraint,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size, name="value", use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_constraint=self.kernel_constraint,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout, name="dropout")
        if self.output_size is None: self.output_size = value_shape[-1]  # value dim
        self.out = tf.keras.layers.Dense(
            self.output_size, name="out", use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_constraint=self.kernel_constraint,
            kernel_regularizer=self.kernel_regularizer,
        )

    def build(self, input_shape):
        _, _, value_shape = input_shape
        self.build_qkvo(value_shape)

    def transpose_for_scores(self, x, batch_size):
        """Transpose to calculate attention scores."""
        # [B, T, H * S] => [B, T, H, S]
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.head_size])
        # => [B, H, T, S]
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call_qkv(self, query, key, value, training=False):
        batch_size = tf.shape(query)[0]
        mixed_query_layer = self.query(query, training=training)
        mixed_key_layer = self.key(key, training=training)
        mixed_value_layer = self.value(value, training=training)

        q = self.transpose_for_scores(mixed_query_layer, batch_size)
        k = self.transpose_for_scores(mixed_key_layer, batch_size)
        v = self.transpose_for_scores(mixed_value_layer, batch_size)

        return q, k, v  # [B, H, T, S]

    def call_attention(self, value, scores, training=False):
        # Normalize the attention scores to probabilities.
        batch_size = tf.shape(value)[0]
        probs = tf.nn.softmax(scores, axis=-1)
        probs = self.dropout(probs, training=training)

        context = tf.matmul(probs, value)  # [B, H, T1, T2] * [B, H, T2, S] = [B, H, T1, S]
        context = tf.transpose(context, perm=[0, 2, 1, 3])  # [B, T, H, S]
        context = tf.reshape(context, [batch_size, -1, self.all_head_size])

        return self.out(context, training=training)

    def call(self, inputs, training=False, **kwargs):
        query, key, value = inputs
        q, k, v = self.call_qkv(query, key, value, training=training)
        # [B, H, T1, S] * [B, H, T2, S] = [B, H, T1, T2]
        scores = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], scores.dtype)  # scale attention scores
        scores = scores / tf.math.sqrt(dk)

        return self.call_attention(v, scores, training=training)

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


class RelPositionMultiHeadAttention(MultiHeadAttention):
    def build_pos(self):
        self.pos = tf.keras.layers.Dense(
            self.all_head_size, name="pos", use_bias=False,
            kernel_regularizer=self.kernel_regularizer,
            kernel_initializer=self.kernel_initializer,
            kernel_constraint=self.kernel_constraint
        )
        self.pos_bias_u = self.add_weight(
            name="pos_bias_u", trainable=True,
            shape=[1, 1, self.num_heads, self.head_size],
            regularizer=self.kernel_regularizer,
            initializer=self.kernel_initializer,
            constraint=self.kernel_constraint
        )
        self.pos_bias_v = self.add_weight(
            name="pos_bias_v", trainable=True,
            shape=[1, 1, self.num_heads, self.head_size],
            regularizer=self.kernel_regularizer,
            initializer=self.kernel_initializer,
            constraint=self.kernel_constraint
        )

    def build(self, input_shape):
        _, _, value_shape, _ = input_shape
        self.build_qkvo(value_shape)
        self.build_pos()

    def rel_shift(self, x):
        x_shape = tf.shape(x)
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2]])
        x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, x_shape)
        return x

    def call(self, inputs, training=False, **kwargs):
        query, key, value, pe = inputs
        q, k, v = self.call_qkv(query, key, value, training=training)
        q = tf.transpose(q, perm=[0, 2, 1, 3])  # [B, T, H, S]

        p = self.pos(pe, training=training)
        p = self.transpose_for_scores(p, tf.shape(pe)[0])  # [B, H, T, S]

        # [B, T, H, S] => [B, H, T, S]
        q_with_bias_u = tf.transpose(q + self.pos_bias_u, perm=[0, 2, 1, 3])
        q_with_bias_v = tf.transpose(q + self.pos_bias_v, perm=[0, 2, 1, 3])

        # => [B, H, T1, T2]
        matrix_ac = tf.matmul(q_with_bias_u, k, transpose_b=True)
        # => [B, H, T1, T2]
        matrix_bd = tf.matmul(q_with_bias_v, p, transpose_b=True)
        matrix_bd = self.rel_shift(matrix_bd)

        scores = matrix_ac + matrix_bd
        dk = tf.cast(tf.shape(k)[-1], scores.dtype)  # scale attention scores
        scores = scores / tf.math.sqrt(dk)

        return self.call_attention(v, scores, training=training)

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
        conf.update(self.pos.get_config())
        return conf
