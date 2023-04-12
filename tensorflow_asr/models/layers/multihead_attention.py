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

from tensorflow_asr.models.base_layer import Layer


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


class MultiHeadAttention(Layer):
    def __init__(
        self,
        num_heads,
        head_size,
        output_size=None,
        dropout=0.0,
        use_projection_bias=True,
        return_attn_coef=False,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef

        self.dropout = tf.keras.layers.Dropout(dropout, name="dropout")
        self._droput_rate = dropout

    def build(self, input_shape):
        num_query_features = input_shape[0][-1]
        num_key_features = input_shape[1][-1]
        num_value_features = input_shape[2][-1] if len(input_shape) > 2 else num_key_features
        output_size = self.output_size if self.output_size is not None else num_value_features
        self.query_kernel = self.add_weight(
            name="query_kernel",
            shape=[self.num_heads, num_query_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.key_kernel = self.add_weight(
            name="key_kernel",
            shape=[self.num_heads, num_key_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.value_kernel = self.add_weight(
            name="value_kernel",
            shape=[self.num_heads, num_value_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.projection_kernel = self.add_weight(
            name="projection_kernel",
            shape=[self.num_heads, self.head_size, output_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_projection_bias:
            self.projection_bias = self.add_weight(
                name="projection_bias",
                shape=[output_size],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.projection_bias = None

    def call_qkv(self, query, key, value):
        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError("the number of elements in 'key' must be equal to " "the same as the number of elements in 'value'")
        # Linear transformations
        query = tf.einsum("BNI,HIO->BNHO", query, self.query_kernel)
        key = tf.einsum("BMI,HIO->BMHO", key, self.key_kernel)
        value = tf.einsum("BMI,HIO->BMHO", value, self.value_kernel)

        return query, key, value

    def call_attention(self, query, key, value, logits, training=False, attention_mask=None):
        # mask = attention mask with shape [B, Tquery, Tkey] with 1 is for positions we want to attend, 0 for masked
        if attention_mask is not None:
            if len(attention_mask.shape) < 2:
                raise ValueError("'mask' must have at least 2 dimensions")
            if query.shape[-3] != attention_mask.shape[-2]:
                raise ValueError("mask's second to last dimension must be equal to the number of elements in 'query'")
            if key.shape[-3] != attention_mask.shape[-1]:
                raise ValueError("mask's last dimension must be equal to the number of elements in 'key'")
        # apply mask
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, logits.dtype)

            # possibly expand on the head dimension so broadcasting works
            if len(attention_mask.shape) != len(logits.shape):
                attention_mask = tf.expand_dims(attention_mask, -3)

            logits += -10e9 * (1.0 - attention_mask)

        attn_coef = tf.nn.softmax(logits)

        # attention dropout
        attn_coef_dropout = self.dropout(attn_coef, training=training)

        # attention * value
        multihead_output = tf.einsum("BHNM,BMHI->BNHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = tf.einsum("BNHI,HIO->BNO", multihead_output, self.projection_kernel)

        if self.projection_bias is not None:
            output += self.projection_bias

        return output, attn_coef

    def call(
        self,
        inputs,
        training=False,
        attention_mask=None,
        use_causal_mask=False,
        use_auto_mask=True,
    ):
        query, key, value = inputs

        if use_auto_mask:
            attention_mask = compute_attention_mask(query=query, key=key, value=value, attention_mask=attention_mask, use_causal_mask=use_causal_mask)

        query, key, value = self.call_qkv(query, key, value)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        depth = tf.constant(self.head_size, dtype=query.dtype)
        query /= tf.sqrt(depth)

        # Calculate dot product attention
        logits = tf.einsum("BNHO,BMHO->BHNM", query, key)

        output, attn_coef = self.call_attention(query, key, value, logits, training=training, attention_mask=attention_mask)

        if self.return_attn_coef:
            return output, attn_coef
        return output

    def compute_output_shape(self, input_shape):
        num_value_features = input_shape[2][-1] if len(input_shape) > 2 else input_shape[1][-1]
        output_size = self.output_size if self.output_size is not None else num_value_features

        output_shape = input_shape[0][:-1] + (output_size,)

        if self.return_attn_coef:
            num_query_elements = input_shape[0][-2]
            num_key_elements = input_shape[1][-2]
            attn_coef_shape = input_shape[0][:-2] + (
                self.num_heads,
                num_query_elements,
                num_key_elements,
            )

            return output_shape, attn_coef_shape
        return output_shape

    def get_config(self):
        config = super().get_config()

        config.update(
            head_size=self.head_size,
            num_heads=self.num_heads,
            output_size=self.output_size,
            dropout=self._droput_rate,
            use_projection_bias=self.use_projection_bias,
            return_attn_coef=self.return_attn_coef,
            kernel_initializer=tf.keras.initializers.serialize(self.kernel_initializer),
            kernel_regularizer=tf.keras.regularizers.serialize(self.kernel_regularizer),
            kernel_constraint=tf.keras.constraints.serialize(self.kernel_constraint),
            bias_initializer=tf.keras.initializers.serialize(self.bias_initializer),
            bias_regularizer=tf.keras.regularizers.serialize(self.bias_regularizer),
            bias_constraint=tf.keras.constraints.serialize(self.bias_constraint),
        )

        return config


class MultiHeadRelativeAttention(MultiHeadAttention):
    def build(self, input_shape):
        num_pos_features = input_shape[-1][-1]
        self.pos_kernel = self.add_weight(
            name="pos_kernel",
            shape=[self.num_heads, num_pos_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        super().build(input_shape[:-1])

    @staticmethod
    def relative_shift(x):
        x_shape = tf.shape(x)
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2]])
        x = tf.reshape(x[:, :, 1:, :], x_shape)
        return x

    def call(
        self,
        inputs,
        content_attention_bias,
        positional_attention_bias,
        training=False,
        attention_mask=None,
        use_causal_mask=False,
        use_auto_mask=True,
    ):
        query, key, value, pos = inputs

        if use_auto_mask:
            attention_mask = compute_attention_mask(query=query, key=key, value=value, attention_mask=attention_mask, use_causal_mask=use_causal_mask)

        query, key, value = self.call_qkv(query, key, value)

        pos = tf.einsum("BMI,HIO->BMHO", pos, self.pos_kernel)

        query_with_u = query + tf.cast(content_attention_bias, query.dtype)
        query_with_v = query + tf.cast(positional_attention_bias, query.dtype)

        logits_with_u = tf.einsum("BNHO,BMHO->BHNM", query_with_u, key)
        logits_with_v = tf.einsum("BNHO,BMHO->BHNM", query_with_v, pos)
        logits_with_v = self.relative_shift(logits_with_v)

        logits = logits_with_u + logits_with_v[:, :, :, : tf.shape(logits_with_u)[3]]

        depth = tf.constant(self.head_size, dtype=logits.dtype)
        logits /= tf.sqrt(depth)

        output, attn_coef = self.call_attention(query, key, value, logits, training=training, attention_mask=attention_mask)

        if self.return_attn_coef:
            return output, attn_coef
        return output
