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

import collections

from keras.src.layers.attention import multi_head_attention as mha_module

from tensorflow_asr import keras, tf
from tensorflow_asr.models.layers.memory import Memory
from tensorflow_asr.utils import shape_util


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


@keras.utils.register_keras_serializable(package=__name__)
class MultiHeadAttention(keras.layers.MultiHeadAttention):
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
        seed=None,
        **kwargs,
    ):
        self._memory_length = memory_length
        self._memory = None
        if output_shape:
            if not isinstance(output_shape, collections.abc.Sized):
                output_shape = (output_shape,)
        super().__init__(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            dropout=dropout,
            use_bias=use_bias,
            output_shape=output_shape,
            attention_axes=attention_axes,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            seed=seed,
            **kwargs,
        )

    def build(self, input_shape):
        query_shape, key_shape, value_shape, *_ = input_shape
        if self._memory_length is not None:
            self._memory = Memory(
                batch_size=query_shape[0],
                memory_length=self._memory_length,
                dmodel=query_shape[-1],
                name="memory",
                dtype=self.dtype_policy,
            )
        return super().build(query_shape, value_shape, key_shape)

    def get_initial_state(self, batch_size: int):
        if self._memory is None:
            return None
        return {
            "key": self._memory.get_initial_state(batch_size),
            "value": self._memory.get_initial_state(batch_size),
        }

    def _with_memory(self, query, key, value, initial_state=None, training=False):
        if self._memory is None or initial_state is None:
            return query, key, value, initial_state

        new_key, new_key_memory = self._memory(key, memories=initial_state.get("key"), training=training)
        new_value, new_value_memory = self._memory(value, memories=initial_state.get("value"), training=training)

        new_states = {
            "key": new_key_memory,
            "value": new_value_memory,
        }

        return query, new_key, new_value, new_states

    def call(
        self,
        inputs,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        use_auto_mask=False,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
        initial_state=None,
        return_states=False,
        **kwargs,
    ):
        query, key, value, *_ = inputs

        if use_auto_mask:
            attention_mask = self._compute_attention_mask(
                query,
                value,
                query_mask=query_mask,
                value_mask=value_mask,
                key_mask=key_mask,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
            )

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S, N, H]
        key = self._key_dense(key)

        # `value` = [B, S, N, H]
        value = self._value_dense(value)

        if return_states:
            query, key, value, states = self._with_memory(query, key, value, initial_state, training)

        attention_output, attention_scores = self._compute_attention(query, key, value, attention_mask, training)
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            if return_states:
                return attention_output, states, attention_scores
            return attention_output, attention_scores

        if return_states:
            return attention_output, states
        return (attention_output,)

    def compute_output_shape(self, input_shape):
        query_shape, key_shape, value_shape, *_ = input_shape
        return super().compute_output_shape(query_shape, value_shape, key_shape)

    def compute_output_spec(
        self,
        inputs,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        use_auto_mask=False,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
        initial_state=None,
        return_states=False,
    ):
        query, value, key, *_ = inputs
        output_spec, *attention_score_spec = super().compute_output_spec(
            query, value, key, query_mask, value_mask, key_mask, attention_mask, return_attention_scores, training, use_causal_mask
        )
        if not return_states:
            return [output_spec] + attention_score_spec
        if self._memory_length is None:
            return [output_spec, None] + attention_score_spec
        states_shape = (query.shape[0], self._memory_length, query.shape[-1])
        states_spec = {
            "key": keras.KerasTensor(states_shape, dtype=self.compute_dtype),
            "value": keras.KerasTensor(states_shape, dtype=self.compute_dtype),
        }
        return [output_spec, states_spec] + attention_score_spec


@keras.utils.register_keras_serializable(package=__name__)
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
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        seed=None,
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
            seed,
            **kwargs,
        )
        self._use_attention_bias = use_attention_bias
        self._causal = causal

    def build(self, input_shape):
        *rest_input_shape, relpe_shape = input_shape
        relpe_rank = len(relpe_shape)
        einsum_equation, bias_axes, output_rank = mha_module._build_proj_equation(relpe_rank - 1, bound_dims=1, output_dims=2)
        self._relpe_dense = keras.layers.EinsumDense(
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
        return super().build(rest_input_shape)

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

        content_query = tf.multiply((query + tf.cast(cbias, query.dtype)), tf.cast(self._inverse_sqrt_key_dim, query.dtype))
        content_attention = tf.einsum(self._dot_product_equation, key, content_query)  # BSNH,BTNH->BNTS

        positional_query = tf.multiply((query + tf.cast(pbias, query.dtype)), tf.cast(self._inverse_sqrt_key_dim, query.dtype))
        positional_attention = tf.einsum(self._dot_product_equation, position, positional_query)  # BRNH,BTNH->BNTR
        positional_attention = rel_left_shift(positional_attention, causal=self._causal)  # BNTR -> BNTS
        positional_attention = tf.slice(
            positional_attention,
            begin=[0, 0, 0, tf.shape(positional_attention)[-1] - tf.shape(content_attention)[-1]],
            size=[-1, -1, -1, tf.shape(content_attention)[-1]],
        )

        attention_scores = content_attention + positional_attention

        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.dropout:
            final_attn_scores = self._dropout_layer(attention_scores, training=training)
        else:
            final_attn_scores = attention_scores

        # `context_layer` = [B, T, N, H]
        attention_output = tf.einsum(self._combine_equation, final_attn_scores, value)
        return attention_output, attention_scores

    def call(
        self,
        inputs,
        content_attention_bias=None,
        positional_attention_bias=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        use_auto_mask=False,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
        initial_state=None,
        return_states=False,
        **kwargs,
    ):
        query, key, value, relpe = inputs

        if use_auto_mask:
            attention_mask = self._compute_attention_mask(
                query,
                value,
                query_mask=query_mask,
                value_mask=value_mask,
                key_mask=key_mask,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
            )

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S, N, H]
        key = self._key_dense(key)

        # `value` = [B, S, N, H]
        value = self._value_dense(value)

        # `position` = [B, R, N, H]
        position = self._relpe_dense(relpe)

        if return_states:
            query, key, value, states = self._with_memory(query, key, value, initial_state, training)

        attention_output, attention_scores = self._compute_attention(
            query,
            key,
            value,
            position,
            content_attention_bias=content_attention_bias,
            positional_attention_bias=positional_attention_bias,
            attention_mask=attention_mask,
            training=training,
        )
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            if return_states:
                return attention_output, states, attention_scores
            return attention_output, attention_scores

        if return_states:
            return attention_output, states
        return (attention_output,)
