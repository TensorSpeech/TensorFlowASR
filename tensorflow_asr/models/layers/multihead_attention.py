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
    from keras.layers.multi_head_attention import _build_attention_equation, _build_proj_equation, _get_output_shape
except ImportError:
    from keras.layers.attention.multi_head_attention import _build_attention_equation, _build_proj_equation, _get_output_shape

from tensorflow_asr.utils import math_util, shape_util


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

    def _get_common_kwargs_for_sublayer(self):
        common_kwargs = dict(
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            dtype=self.dtype,
        )
        # Create new clone of kernel/bias initializer, so that we don't reuse
        # the initializer instance, which could lead to same init value since
        # initializer is stateless.
        kernel_initializer = self._kernel_initializer.__class__.from_config(self._kernel_initializer.get_config())
        bias_initializer = self._bias_initializer.__class__.from_config(self._bias_initializer.get_config())
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return common_kwargs

    def _build_attention(self, rank):
        """Builds multi-head dot-product attention computations.

        This function builds attributes necessary for `_compute_attention` to
        customize attention computation to replace the default dot-product
        attention.

        Args:
            rank: the rank of query, key, value tensors.
        """
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)
        (
            self._dot_product_equation,
            self._combine_equation,
            attn_scores_rank,
        ) = _build_attention_equation(rank, attn_axes=self._attention_axes)
        norm_axes = tuple(range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
        self._softmax = tf.keras.layers.Softmax(axis=norm_axes, dtype=self.dtype)
        self._dropout_layer = tf.keras.layers.Dropout(rate=self._dropout, dtype=self.dtype)

    def call(
        self,
        inputs,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
        use_auto_mask=True,
    ):
        query, key, value = inputs
        if use_auto_mask:
            attention_mask = self._compute_attention_mask(query, value, key=key, attention_mask=attention_mask, use_causal_mask=use_causal_mask)

        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)

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
            return attention_output, attention_scores, None
        return attention_output, None

    def compute_output_shape(self, input_shape):
        query_shape, key_shape, value_shape = input_shape
        if key_shape is None:
            key_shape = value_shape

        query_shape = tf.TensorShape(query_shape)
        value_shape = tf.TensorShape(value_shape)
        key_shape = tf.TensorShape(key_shape)

        if query_shape[-1] != value_shape[-1]:
            raise ValueError(
                "The last dimension of `query_shape` and `value_shape` "
                f"must be equal, but are {query_shape[-1]}, {value_shape[-1]}. "
                "Received: query_shape={query_shape}, value_shape={value_shape}"
            )

        if value_shape[1:-1] != key_shape[1:-1]:
            raise ValueError(
                "All dimensions of `value` and `key`, except the last one, " f"must be equal. Received {value_shape} and " f"{key_shape}"
            )

        if self._output_shape:
            return query_shape[:-1].concatenate(self._output_shape)

        return query_shape, None


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
        with tf_utils.maybe_init_scope(self):  # pylint: disable=not-context-manager
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                self._relative_position_encoding_shape.rank - 1, bound_dims=1, output_dims=2
            )
            self._encoding_dense = EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(output_rank - 1, [self._num_heads, self._key_dim]),
                bias_axes=bias_axes if self._use_bias else None,
                name="encoding",
                **self._get_common_kwargs_for_sublayer(),
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
        content_attention = tf.einsum(
            self._dot_product_equation,
            key,
            (query + tf.cast(content_attention_bias, query.dtype)),
        )  # BSNH,BTNH->BNTS
        positional_attention = tf.einsum(
            self._dot_product_equation,
            position,
            (query + tf.cast(positional_attention_bias, query.dtype)),
        )  # BRNH,BTNH->BNTR
        positional_attention = rel_left_shift(positional_attention)

        attention_scores = content_attention + tf.slice(positional_attention, begin=[0, 0, 0, 0], size=tf.shape(content_attention))
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self._key_dim)))

        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        attention_output = self._dropout_layer(attention_scores, training=training)

        attention_output = tf.einsum(self._combine_equation, attention_output, value)  # BNTS,BVNH->BTNH
        return attention_output, attention_scores

    def _update_memory(self, query, key, value, memory=None):
        if memory is None:
            return query, key, value, memory

        B, M, D = shape_util.shape_list(memory)  # [B, Mmax, D]
        memory_mask = getattr(memory, "_keras_mask", None)
        if memory_mask is not None:
            memory_mask = tf.cast(memory_mask, memory.dtype)
            memory_lengths = tf.math.count_nonzero(memory_mask, axis=1, keepdims=False, dtype=tf.int32)
            memory_ragged = tf.RaggedTensor.from_tensor(memory, lengths=memory_lengths)  # [B, M, D]
        else:
            memory_ragged = tf.RaggedTensor.from_tensor(memory)
            memory_lengths = tf.repeat(tf.shape(memory, tf.int32)[1][None, ...], B, axis=0)
        memory = tf.stop_gradient(memory)
        memory_ragged = tf.stop_gradient(memory_ragged)

        _, kT, _ = shape_util.shape_list(key)  # [B, Tmax, D]
        key_mask = getattr(key, "_keras_mask", None)
        if key_mask is not None:
            key_mask = tf.cast(key_mask, key.dtype)
            key_lengths = tf.math.count_nonzero(key_mask, axis=1, keepdims=False, dtype=tf.int32)
            key_ragged = tf.RaggedTensor.from_tensor(key, lengths=key_lengths)  # [B, T, D]
        else:
            key_ragged = tf.RaggedTensor.from_tensor(key)
        key_ragged = tf.concat([memory_ragged, key_ragged], 1)  # [B, M + T, D]
        key = key_ragged.to_tensor(shape=(B, M + kT, D))  # [B, Mmax + Tmax, D]
        if key_mask is not None:
            key = math_util.apply_mask(key, mask=tf.sequence_mask(memory_lengths + key_lengths, M + kT), multiply=False)

        _, vT, _ = shape_util.shape_list(value)  # [B, Tmax, D]
        value_mask = getattr(value, "_keras_mask", None)
        if value_mask is not None:
            value_mask = tf.cast(value_mask, value.dtype)
            value_lengths = tf.math.count_nonzero(value_mask, axis=1, keepdims=False, dtype=tf.int32)
            value_ragged = tf.RaggedTensor.from_tensor(value, lengths=value_lengths)  # [B, T, D]
        else:
            value_ragged = tf.RaggedTensor.from_tensor(value)
        value_ragged = tf.concat([memory_ragged, value_ragged], 1)  # [B, M + T, D]
        value = value_ragged.to_tensor(shape=(B, M + vT, D))  # [B, Mmax + Tmax, D]
        if value_mask is not None:
            value = math_util.apply_mask(value, mask=tf.sequence_mask(memory_lengths + value_lengths, M + vT), multiply=False)

        query_mask = getattr(query, "_keras_mask", None)
        if query_mask is not None:
            query_mask = tf.cast(query_mask, query.dtype)
            query_lengths = tf.math.count_nonzero(query_mask, axis=1, keepdims=False, dtype=tf.int32)
            query_ragged = tf.RaggedTensor.from_tensor(query, lengths=query_lengths)
        else:
            query_ragged = tf.RaggedTensor.from_tensor(query)

        # construct new memory, without paddings (move the padding to the end of new memory)
        new_memory = tf.concat([memory_ragged, query_ragged], 1)
        new_memory_row_lengths = tf.cast(new_memory.row_lengths(axis=1), memory_lengths.dtype)
        new_memory_shifts = tf.maximum(0, new_memory_row_lengths - memory_lengths)
        new_memory, new_memory_shifts = tf.map_fn(lambda x: (tf.roll(x[0], shift=-x[1], axis=0), x[1]), (new_memory, new_memory_shifts))
        new_memory = new_memory.to_tensor(shape=tf.shape(memory))
        new_memory = math_util.apply_mask(
            new_memory,
            mask=tf.sequence_mask(new_memory_row_lengths - new_memory_shifts, tf.shape(memory, tf.int32)[1]),
            multiply=True,
        )
        new_memory = tf.stop_gradient(new_memory)
        return query, key, value, new_memory

    def call(
        self,
        inputs,
        content_attention_bias,
        positional_attention_bias,
        memory=None,
        attention_mask=None,
        training=None,
        use_causal_mask=False,
        use_auto_mask=True,
        return_attention_scores=False,
    ):
        query, key, value, relative_position_encoding = inputs

        query, key, value, memory = self._update_memory(query, key, value, memory)

        if use_auto_mask:
            attention_mask = self._compute_attention_mask(query, value, key=key, attention_mask=attention_mask, use_causal_mask=use_causal_mask)

        if not self._built_from_signature:
            self._build_from_signature(query, value, relative_position_encoding, key=key)

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

        pos_is_ragged = isinstance(relative_position_encoding, tf.RaggedTensor)
        if pos_is_ragged:
            relative_position_encoding = relative_position_encoding.to_tensor()

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S + M, N, H]
        key = self._key_dense(key)

        # `value` = [B, S + M, N, H]
        value = self._value_dense(value)

        # `position` = [B, R, N, H]
        position = self._encoding_dense(relative_position_encoding)

        attention_output, attention_scores = self._compute_attention(
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

        if query_is_ragged:
            attention_output = tf.RaggedTensor.from_tensor(attention_output, lengths=query_lengths)

        if return_attention_scores:
            return attention_output, attention_scores, memory
        return attention_output, memory

    def compute_output_shape(self, input_shape, content_attention_bias_shape, positional_attention_bias_shape, memory_shape=None):
        query_shape, key_shape, value_shape, _ = input_shape
        if key_shape is None:
            key_shape = value_shape

        query_shape = tf.TensorShape(query_shape)
        value_shape = tf.TensorShape(value_shape)
        key_shape = tf.TensorShape(key_shape)

        if query_shape[-1] != value_shape[-1]:
            raise ValueError(
                "The last dimension of `query_shape` and `value_shape` "
                f"must be equal, but are {query_shape[-1]}, {value_shape[-1]}. "
                "Received: query_shape={query_shape}, value_shape={value_shape}"
            )

        if value_shape[1:-1] != key_shape[1:-1]:
            raise ValueError(
                "All dimensions of `value` and `key`, except the last one, " f"must be equal. Received {value_shape} and " f"{key_shape}"
            )

        if self._output_shape:
            return query_shape[:-1].concatenate(self._output_shape)

        return query_shape, memory_shape
