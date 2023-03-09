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

import math

import tensorflow as tf
from keras.layers import EinsumDense
from keras.layers import MultiHeadAttention as KerasMultiHeadAttention
from keras.utils import tf_utils

try:
    from keras.layers.multi_head_attention import _build_proj_equation, _get_output_shape
except ImportError:
    from keras.layers.attention.multi_head_attention import _build_proj_equation, _get_output_shape


def rel_left_shift(x):
    """
    Relative left shift

    Input:
    tf.Tensor(
    [[1 2 3]
    [4 5 6]
    [7 8 9]], shape=(3, 3), dtype=int32)

    Output:
    tf.Tensor(
    [[3 0 0]
    [5 6 0]
    [7 8 9]], shape=(3, 3), dtype=int32)

    Args:
        x (tf.Tensor): shape BNTR

    Returns:
        x: left shifted, shape BNTR
    """
    x = tf.transpose(x, perm=[2, 3, 0, 1])  # BNTR -> TRBN
    x_shape = tf.shape(x)

    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])  # shift on position time dimension R
    x = tf.reshape(x, [x_shape[1] + 1, x_shape[0], x_shape[2], x_shape[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_shape)

    x = tf.transpose(x, perm=[2, 3, 0, 1])  # TRBN -> BNTR
    x *= tf.linalg.band_part(tf.ones((1, 1, x_shape[0], x_shape[1]), x.dtype), -1, 0)
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


def compute_attention_mask(query, value, key=None, attention_mask=None, use_causal_mask=False):
    """Computes the attention mask, using the Keras masks of the inputs.

    * The `query`'s mask is reshaped from [B, T] to [B, T, 1].
    * The `value`'s mask is reshaped from [B, S] to [B, 1, S].
    * The `key`'s mask is reshaped from [B, S] to [B, 1, S]. The `key`'s
      mask is ignored if `key` is `None` or if `key is value`.
    * If `use_causal_mask=True`, then the causal mask is computed. Its shape
      is [1, T, S].

    All defined masks are merged using a logical AND operation (`&`).

    In general, if the `query` and `value` are masked, then there is no need
    to define the `attention_mask`.

    Args:
      query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
      key: Projected key `Tensor` of shape `(B, T, N, key_dim)`.
      value: Projected value `Tensor` of shape `(B, T, N, value_dim)`.
      attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
        attention to certain positions.
      use_causal_mask: A boolean to indicate whether to apply a causal mask
        to prevent tokens from attending to future tokens (e.g., used in a
        decoder Transformer).

    Returns:
      attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
        attention to certain positions, based on the Keras masks of the
        `query`, `key`, `value`, and `attention_mask` tensors, and the
        causal mask if `use_causal_mask=True`.
    """
    query_mask = getattr(query, "_keras_mask", None)
    value_mask = getattr(value, "_keras_mask", None)
    key_mask = getattr(key, "_keras_mask", None)
    auto_mask = None
    if query_mask is not None:
        query_mask = tf.cast(query_mask, tf.bool)  # defensive casting
        # B = batch size, T = max query length
        auto_mask = query_mask[:, :, tf.newaxis]  # shape is [B, T, 1]
    if value_mask is not None:
        value_mask = tf.cast(value_mask, tf.bool)  # defensive casting
        # B = batch size, S == max value length
        mask = value_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
        auto_mask = mask if auto_mask is None else auto_mask & mask
    if key_mask is not None:
        key_mask = tf.cast(key_mask, tf.bool)  # defensive casting
        # B == batch size, S == max key length == max value length
        mask = key_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
        auto_mask = mask if auto_mask is None else auto_mask & mask
    if use_causal_mask:
        # the shape of the causal mask is [1, T, S]
        mask = compute_causal_mask(query, value)
        auto_mask = mask if auto_mask is None else auto_mask & mask
    if auto_mask is not None:
        # merge attention_mask & automatic mask, to shape [B, T, S]
        attention_mask = auto_mask if attention_mask is None else tf.cast(attention_mask, bool) & auto_mask
    return attention_mask


class MultiHeadAttention(KerasMultiHeadAttention):
    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0,
        use_bias=True,
        output_shape=None,
        attention_axes=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            num_heads,
            key_dim,
            value_dim,
            dropout,
            use_bias,
            output_shape,
            attention_axes,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            **kwargs,
        )
        if not hasattr(self, "_compute_attention_mask"):
            self._compute_attention_mask = compute_attention_mask
        if not hasattr(self, "_compute_causal_mask"):
            self._compute_causal_mask = compute_causal_mask

    def call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
        use_auto_mask=True,
    ):
        if use_auto_mask:
            attention_mask = self._compute_attention_mask(query, value, key=key, attention_mask=attention_mask, use_causal_mask=use_causal_mask)

        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value

        query_is_ragged = isinstance(query, tf.RaggedTensor)
        if query_is_ragged:
            query_lengths = query.nested_row_lengths()
            query = query.to_tensor()

        key_is_ragged = isinstance(key, tf.RaggedTensor)
        value_is_ragged = isinstance(value, tf.RaggedTensor)
        if key_is_ragged and value_is_ragged:
            # Ensure they have the same shape.
            bounding_shape = tf.math.maximum(key.bounding_shape(), value.bounding_shape())
            key = key.to_tensor(shape=bounding_shape)
            value = value.to_tensor(shape=bounding_shape)
        elif key_is_ragged:
            key = key.to_tensor(shape=tf.shape(value))
        elif value_is_ragged:
            value = value.to_tensor(shape=tf.shape(key))

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S, N, H]
        key = self._key_dense(key)

        # `value` = [B, S, N, H]
        value = self._value_dense(value)

        attention_output, attention_scores = self._compute_attention(query, key, value, attention_mask, training)
        attention_output = self._output_dense(attention_output)

        if query_is_ragged:
            attention_output = tf.RaggedTensor.from_tensor(attention_output, lengths=query_lengths)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output


class MultiHeadRelativeAttention(MultiHeadAttention):
    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0,
        use_bias=True,
        output_shape=None,
        attention_axes=None,
        kernel_initializer="variance_scaling",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            num_heads,
            key_dim,
            value_dim,
            dropout,
            use_bias,
            output_shape,
            attention_axes,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            **kwargs,
        )
        self._relative_position_encoding_shape = None

    def _build_from_signature(self, query, value, relative_position_encoding, key=None):
        super()._build_from_signature(query=query, value=value, key=key)
        if hasattr(relative_position_encoding, "shape"):
            self._relative_position_encoding_shape = tf.TensorShape(relative_position_encoding.shape)
        else:
            self._relative_position_encoding_shape = tf.TensorShape(relative_position_encoding)

        common_kwargs = dict(
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )

        with tf_utils.maybe_init_scope(self):  # pylint: disable=not-context-manager
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                self._relative_position_encoding_shape.rank - 1, bound_dims=1, output_dims=2
            )
            self._encoding_dense = EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(output_rank - 1, [self._num_heads, self._key_dim]),
                bias_axes=bias_axes if self._use_bias else None,
                name="encoding",
                **common_kwargs,
            )

    def _compute_attention(
        self,
        query,
        key,
        value,
        position,
        content_attention_bias,
        positional_attention_bias,
        attention_mask=None,
        training=None,
    ):
        content_attention = tf.einsum(self._dot_product_equation, key, (query + content_attention_bias))  # BSNH,BTNH->BNTS
        positional_attention = tf.einsum(self._dot_product_equation, position, (query + positional_attention_bias))  # BRNH,BTNH->BNTR
        positional_attention = rel_left_shift(positional_attention)
        attention_scores = content_attention + positional_attention
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self._key_dim)))

        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        attention_output = self._dropout_layer(attention_scores, training=training)

        attention_output = tf.einsum(self._combine_equation, attention_output, value)  # BNTS,BVNH->BTNH
        return attention_output

    def call(
        self,
        query,
        value,
        relative_position_encoding,
        content_attention_bias,
        positional_attention_bias,
        key=None,
        state=None,
        attention_mask=None,
        training=None,
        use_causal_mask=False,
        use_auto_mask=True,
    ):
        if not self._built_from_signature:
            self._build_from_signature(query, value, relative_position_encoding, key=key)
        if key is None:
            key = value
        if state is not None and state.shape.ndims > 1:
            value = tf.concat([state, value], 1)
            key = tf.concat([state, key], 1)

        if use_auto_mask:
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
            query=query,
            key=key,
            value=value,
            position=position,
            content_attention_bias=content_attention_bias,
            positional_attention_bias=positional_attention_bias,
            attention_mask=attention_mask,
            training=training,
        )

        # `attention_output` = [B, S, N, H]
        attention_output = self._output_dense(attention_output)

        return attention_output
