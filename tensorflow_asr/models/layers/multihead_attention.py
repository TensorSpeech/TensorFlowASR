# pylint: disable=attribute-defined-outside-init
# Copyright 2022 Huy Le Nguyen (@nglehuy)
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

import importlib
import math

import tensorflow as tf
from keras.layers import EinsumDense
from keras.layers import MultiHeadAttention as KerasMultiHeadAttention

from tensorflow_asr.models.layers.memory import Memory
from tensorflow_asr.utils import shape_util
from tensorflow_asr.utils.env_util import KERAS_SRC

mha_module = importlib.import_module(f"{KERAS_SRC}.layers.attention.multi_head_attention")


def rel_left_shift(x, causal=False):
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
    b, n, t, r = shape_util.shape_list(x)

    # fmt: off
    if causal:
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]]) # [B, N, T, Th + T]
        x = tf.reshape(x, [b, n, -1])
        x = tf.pad(x, [[0, 0], [0, 0], [r - t, 0]])
        x = tf.reshape(x, [b, n, 1 + t, r])
        x = tf.slice(x, begin=[0, 0, 1, 0], size=[-1, -1, -1, -1]) # [B, N, T, Th + T]
    else:
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, 1]])  # [B, N, T, Th + 2*T] where R = Th + 2*T - 1, S = Th + T
        x = tf.reshape(x, [b, n, -1])  # [B, N, TTh + 2*TT]
        x = tf.pad(x, [[0, 0], [0, 0], [0, r - t]])  # [B, N, TTh + 2*TT + Th + 2*T - 1 - T] = [B, N, TTh + 2*TT + Th + T - 1]
        x = tf.reshape(x, [b, n, 1 + t, r])  # TTh + 2*TT + Th + T - 1 = TTh + 2*TT + Th + 2*T - T - 1 = Th(T + 1) + 2*T(T + 1) - (T + 1) = (T + 1)(Th + 2*T - 1) = (T + 1)R # pylint: disable=line-too-long
        x = tf.slice(x, begin=[0, 0, 0, (t - 1)], size=[-1, -1, t, -1]) # [B, N, T, Th + T]
    # fmt: on

    # x = tf.transpose(x, perm=[2, 3, 0, 1])  # BNTR -> TRBN
    # x_shape = tf.shape(x)

    # x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])  # shift on position time dimension R
    # x = tf.reshape(x, [x_shape[1] + 1, x_shape[0], x_shape[2], x_shape[3]])
    # x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    # x = tf.reshape(x, x_shape)
    # if mask_upper_triangle:
    #     x *= tf.reverse(tf.linalg.band_part(tf.ones((x_shape[0], x_shape[1]), x.dtype), 0, -1), [0, 1])[..., tf.newaxis, tf.newaxis]

    # x = tf.transpose(x, perm=[2, 3, 0, 1])  # TRBN -> BNTR
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
        memory_length=None,
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
        self._memory_length = memory_length
        self.stateful = self._memory_length is not None

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
        ) = mha_module._build_attention_equation(rank, attn_axes=self._attention_axes)
        norm_axes = tuple(range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
        self._softmax = tf.keras.layers.Softmax(axis=norm_axes, dtype=self.dtype)  # stable training
        self._dropout_layer = tf.keras.layers.Dropout(rate=self._dropout, dtype=self.dtype)

    def _masked_softmax(self, attention_scores, attention_mask=None):
        # Normalize the attention scores to probabilities.
        # `attention_scores` = [B, N, T, S]
        if attention_mask is not None:
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims,
            # key_attention_dims>)
            mask_expansion_axis = -len(self._attention_axes) * 2 - 1
            for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
                attention_mask = tf.expand_dims(attention_mask, axis=mask_expansion_axis)
        attention_scores = self._softmax(attention_scores, attention_mask)
        return tf.cast(attention_scores, self.dtype)

    def _build_from_signature(self, query, value, key=None):
        super()._build_from_signature(query, value, key)
        with tf.init_scope():  # pylint: disable=not-context-manager
            batch_size, _, dmodel = self._query_shape
            if self._memory_length is not None:
                self._memory = Memory(batch_size=batch_size, memory_length=self._memory_length, dmodel=dmodel, name="memory", dtype=self.dtype)
            else:
                self._memory = None

    def reset_caching(self):
        if self._memory is None:
            return None
        return self._memory.reset_caching()

    def _update_with_memory(self, query, key, value, caching=None):
        if self._memory is None:
            return query, key, value, caching

        key = self._memory.attach_memory(key, memories=caching)
        value = self._memory.attach_memory(value, memories=caching)

        caching = self._memory(query, memories=caching)  # update memory

        return query, key, value, caching

    def call(
        self,
        inputs,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
        use_auto_mask=True,
    ):
        query, key, value, caching, *_ = inputs

        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)

        query, key, value, caching = self._update_with_memory(query, key, value, caching=caching)

        if use_auto_mask:
            attention_mask = self._compute_attention_mask(query, value, key=key, attention_mask=attention_mask, use_causal_mask=use_causal_mask)

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

        if return_attention_scores:
            return attention_output, caching, attention_scores
        return attention_output, caching

    def compute_output_shape(self, input_shape):
        query_shape, _, _, caching_shape, *_ = input_shape
        return query_shape, caching_shape


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
        memory_length=None,
        kernel_initializer="variance_scaling",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_attention_bias=False,
        causal=False,
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
            memory_length,
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
        self._use_attention_bias = use_attention_bias
        self._causal = causal

    def _build_from_signature(self, query, value, relative_position_encoding, key=None):
        super()._build_from_signature(query=query, value=value, key=key)
        if hasattr(relative_position_encoding, "shape"):
            self._relative_position_encoding_shape = tf.TensorShape(relative_position_encoding.shape)
        else:
            self._relative_position_encoding_shape = tf.TensorShape(relative_position_encoding)
        with tf.init_scope():  # pylint: disable=not-context-manager
            einsum_equation, bias_axes, output_rank = mha_module._build_proj_equation(
                self._relative_position_encoding_shape.rank - 1, bound_dims=1, output_dims=2
            )
            self._encoding_dense = EinsumDense(
                einsum_equation,
                output_shape=mha_module._get_output_shape(output_rank - 1, [self._num_heads, self._key_dim]),
                bias_axes=bias_axes if self._use_bias else None,
                name="encoding",
                **self._get_common_kwargs_for_sublayer(),
            )
            if self._use_attention_bias:
                self.content_attention_bias = self.add_weight(
                    name="content_attention_bias",
                    shape=[self._num_heads, self._key_dim],
                    trainable=True,
                    initializer="zeros",
                    regularizer=self._bias_regularizer,
                    dtype=self.variable_dtype,
                )
                self.positional_attention_bias = self.add_weight(
                    name="positional_attention_bias",
                    shape=[self._num_heads, self._key_dim],
                    trainable=True,
                    initializer="zeros",
                    regularizer=self._bias_regularizer,
                    dtype=self.variable_dtype,
                )
            else:
                self.content_attention_bias, self.positional_attention_bias = None, None

    def _compute_attention(
        self,
        query,
        key,
        value,
        position,
        content_attention_bias=None,
        positional_attention_bias=None,
        attention_mask=None,
        training=None,
    ):
        cbias = self.content_attention_bias if content_attention_bias is None else content_attention_bias
        pbias = self.positional_attention_bias if positional_attention_bias is None else positional_attention_bias
        content_attention = tf.einsum(self._dot_product_equation, key, (query + tf.cast(cbias, query.dtype)))  # BSNH,BTNH->BNTS
        positional_attention = tf.einsum(self._dot_product_equation, position, (query + tf.cast(pbias, query.dtype)))  # BRNH,BTNH->BNTR
        positional_attention = rel_left_shift(positional_attention, causal=self._causal)  # BNTR -> BNTS
        positional_attention = tf.slice(
            positional_attention,
            begin=[0, 0, 0, tf.shape(positional_attention)[-1] - tf.shape(content_attention)[-1]],
            size=[-1, -1, -1, tf.shape(content_attention)[-1]],
        )

        attention_scores = content_attention + positional_attention
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self._key_dim)))

        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        attention_output = self._dropout_layer(attention_scores, training=training)

        attention_output = tf.einsum(self._combine_equation, attention_output, value)  # BNTS,BSNH->BTNH
        return attention_output, attention_scores

    def call(
        self,
        inputs,
        attention_mask=None,
        training=None,
        use_causal_mask=False,
        use_auto_mask=True,
        return_attention_scores=False,
    ):
        query, key, value, caching, relative_position_encoding, content_attention_bias, positional_attention_bias, *_ = inputs

        if not self._built_from_signature:
            self._build_from_signature(query, value, relative_position_encoding, key=key)

        query, key, value, caching = self._update_with_memory(query, key, value, caching=caching)

        if use_auto_mask:
            attention_mask = self._compute_attention_mask(query, value, key=key, attention_mask=attention_mask, use_causal_mask=use_causal_mask)

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

        if return_attention_scores:
            return attention_output, caching, attention_scores
        return attention_output, caching
