# pylint: disable=attribute-defined-outside-init
# Copyright 2022 Huy Le Nguyen (@usimarit)
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
from keras.layers import EinsumDense
from keras.layers import MultiHeadAttention as KerasMultiHeadAttention

from tensorflow_asr.utils import math_util

try:
    from keras.layers.multi_head_attention import _build_proj_equation, _get_output_shape
except ImportError:
    from keras.layers.attention.multi_head_attention import _build_proj_equation, _get_output_shape


def _rel_shift(x):
    x = tf.transpose(x, perm=[2, 3, 0, 1])  # BHNM -> NMBH
    x_shape = tf.shape(x)

    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])  # shift on position time dimension M
    x = tf.reshape(x, [x_shape[1] + 1, x_shape[0], x_shape[2], x_shape[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_shape)

    x = tf.transpose(x, perm=[2, 3, 0, 1])  # NMBH -> BHNM
    return x


def compute_causal_mask(query, value=None):
    """Computes a causal mask (e.g., for masked self-attention layers).
    For example, if query and value both contain sequences of length 4,
    this function returns a boolean `Tensor` equal to:
    ```
    [[[True,  False, False, False],
      [True,  True,  False, False],
      [True,  True,  True,  False],
      [True,  True,  True,  True]]]
    ```
    Args:
      query: query `Tensor` of shape `(B, T, ...)`.
      value: value `Tensor` of shape `(B, S, ...)` (optional, defaults to
      query).
    Returns:
      mask: a boolean `Tensor` of shape [1, T, S] containing a lower
            triangular matrix of shape [T, S].
    """
    q_seq_length = tf.shape(query)[1]
    v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
    return tf.linalg.band_part(tf.ones((1, q_seq_length, v_seq_length), tf.bool), -1, 0)  # creates a lower triangular matrix


def compute_self_attention_mask(max_length, inputs_length, use_causal_mask=False):
    """
    Returns
    ```
    [[[True, True, True, False],
      [True, True, True, False],
      [True, True, True, False],
      [False, False, False, False]]]
    ```
    """
    qmask = tf.sequence_mask(inputs_length, maxlen=max_length)
    attention_mask = qmask[:, :, None] & qmask[:, None, :]
    if use_causal_mask:
        attention_mask = attention_mask & compute_causal_mask(qmask, value=qmask)
    return attention_mask


class MultiHeadAttention(KerasMultiHeadAttention):
    def _compute_attention(
        self,
        query,
        key,
        value,
        attention_mask=None,
        training=None,
    ):
        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.
        scale = 1.0 / tf.sqrt(tf.constant(self._key_dim, dtype=query.dtype))

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = tf.einsum(self._dot_product_equation, key, query * scale)

        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)

        # `context_layer` = [B, T, N, H]
        attention_output = tf.einsum(self._combine_equation, attention_scores_dropout, value)
        return attention_output, attention_scores

    # def _masked_softmax(self, attention_scores, attention_mask=None):
    #     if attention_mask is not None:
    #         # The expand dim happens starting from the `num_heads` dimension,
    #         # (<batch_dims>, num_heads, <query_attention_dims,
    #         # key_attention_dims>)
    #         mask_expansion_axis = -len(self._attention_axes) * 2 - 1
    #         for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
    #             attention_mask = tf.expand_dims(attention_mask, axis=mask_expansion_axis)
    #         attention_scores = math_util.masked_fill(
    #             attention_scores, mask=attention_mask, value=math_util.large_compatible_negative(attention_scores.dtype)
    #         )
    #     attention_scores = self._softmax(attention_scores)
    #     return attention_scores


class MultiHeadRelativeAttention(MultiHeadAttention):
    def __init__(
        self,
        kernel_initializer="variance_scaling",
        **kwargs,
    ):
        super().__init__(kernel_initializer=kernel_initializer, **kwargs)

    def _build_from_signature(self, query, value, key=None):
        super()._build_from_signature(query=query, value=value, key=key)
        if hasattr(value, "shape"):
            value_shape = tf.TensorShape(value.shape)
        else:
            value_shape = value
        if key is None:
            key_shape = value_shape
        elif hasattr(key, "shape"):
            key_shape = tf.TensorShape(key.shape)
        else:
            key_shape = key

        common_kwargs = dict(
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )

        with tf.init_scope():  # pylint: disable=not-context-manager
            einsum_equation, _, output_rank = _build_proj_equation(key_shape.rank - 1, bound_dims=1, output_dims=2)
            self._encoding_dense = EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(output_rank - 1, [self._num_heads, self._key_dim]),
                bias_axes=None,
                name="encoding",
                **common_kwargs,
            )
            self.content_attention_bias = self.add_weight(
                name="content_attention_bias",
                shape=[self._num_heads, self._key_dim],
                dtype=self.dtype,
                trainable=True,
                initializer="zeros",
                regularizer=self._bias_regularizer,
            )
            self.positional_attention_bias = self.add_weight(
                name="positional_attention_bias",
                shape=[self._num_heads, self._key_dim],
                dtype=self.dtype,
                trainable=True,
                initializer="zeros",
                regularizer=self._bias_regularizer,
            )

    def _compute_attention(
        self,
        query,
        key,
        value,
        position,
        attention_mask=None,
        training=None,
    ):
        scale = 1.0 / tf.sqrt(tf.constant(self._key_dim, dtype=query.dtype))

        content_attention = tf.einsum(self._dot_product_equation, key, (query + self.content_attention_bias) * scale)  # BSNH,BTNH->BNTS
        positional_attention = tf.einsum(self._dot_product_equation, position, (query + self.positional_attention_bias) * scale)  # BRNH,BTNH->BNTR
        positional_attention = _rel_shift(positional_attention)
        attention_scores = content_attention + positional_attention

        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        attention_output = self._dropout_layer(attention_scores, training=training)

        attention_output = tf.einsum(self._combine_equation, attention_output, value)  # BNTS,BVNH->BTNH
        return attention_output

    def call(
        self,
        query,
        value,
        relative_position_encoding,
        key=None,
        state=None,
        attention_mask=None,
        training=None,
        use_causal_mask=False,
    ):
        if not self._built_from_signature:
            self._build_from_signature(query, value, key=key)
        if key is None:
            key = value
        if state is not None and state.shape.ndims > 1:
            value = tf.concat([state, value], 1)
            key = tf.concat([state, key], 1)

        if hasattr(self, "_compute_attention_mask"):
            attention_mask = self._compute_attention_mask(query, value, key=key, attention_mask=attention_mask, use_causal_mask=use_causal_mask)

        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S + M, N, H]
        key = self._key_dense(key)

        # `value` = [B, S + M, N, H]
        value = self._value_dense(value)

        # `position` = [B, R, N, H]
        position = self._encoding_dense(relative_position_encoding)

        attention_output = self._compute_attention(
            query=query, key=key, value=value, position=position, attention_mask=attention_mask, training=training
        )

        # `attention_output` = [B, S, N, H]
        attention_output = self._output_dense(attention_output)

        return attention_output
