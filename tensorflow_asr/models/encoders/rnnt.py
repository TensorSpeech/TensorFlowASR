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
""" http://arxiv.org/abs/1811.06621 """

import tensorflow as tf
from packaging import version

from tensorflow_asr.models.base_layer import Layer, Reshape
from tensorflow_asr.models.layers.subsampling import TimeReduction
from tensorflow_asr.utils import layer_util, math_util


class RnnTransducerBlock(Layer):
    def __init__(
        self,
        reduction_factor: int = 0,
        dmodel: int = 640,
        rnn_type: str = "lstm",
        rnn_units: int = 2048,
        rnn_unroll: bool = False,
        layer_norm: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reduction = TimeReduction(reduction_factor, name="reduction") if reduction_factor > 0 else None
        self.rnn = layer_util.get_rnn(rnn_type)(
            units=rnn_units,
            return_sequences=True,
            name=rnn_type,
            unroll=rnn_unroll,
            return_state=True,
            zero_output_for_mask=True,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            # https://github.com/tensorflow/tensorflow/issues/61352#issuecomment-1647276639
            dtype=tf.float32 if self.dtype == tf.bfloat16 and version.parse(tf.version.VERSION) < version.parse("2.13.0") else None,
        )
        self.ln = (
            tf.keras.layers.LayerNormalization(name="ln", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer)
            if layer_norm
            else None
        )
        self.projection = tf.keras.layers.Dense(dmodel, name="projection", kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        if self.reduction is not None:
            outputs, outputs_length = self.reduction((outputs, outputs_length))
        if self.dtype != self.rnn.dtype:
            outputs = tf.cast(outputs, tf.float32)
        outputs, *_ = self.rnn(outputs, training=training)
        if self.dtype != self.rnn.dtype:
            outputs = tf.cast(outputs, self.dtype)
        if self.ln is not None:
            outputs = self.ln(outputs, training=training)
        outputs = self.projection(outputs, training=training)
        return outputs, outputs_length

    def compute_mask(self, inputs, mask=None):
        if self.reduction is not None:
            mask = self.reduction.compute_mask(inputs)
        return mask

    def call_next(self, inputs, inputs_length, previous_encoder_states):
        """
        Recognize function for encoder network from the previous encoder states

        Parameters
        ----------
        inputs : tf.Tensor, shape [B, T, E]
        previous_encoder_states : tf.Tensor, shape [nstates, B, rnn_units]

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor], shapes ([B, T, dmodel], [B], [nstates, B, rnn_units])
        """
        with tf.name_scope(f"{self.name}_call_next"):
            outputs, outputs_length = inputs, inputs_length
            outputs, *_states = self.rnn(
                outputs,
                training=False,
                initial_state=tf.unstack(previous_encoder_states, axis=0),
                mask=getattr(inputs, "_keras_mask", None),
            )
            new_states = tf.stack(_states, axis=0)
            if self.ln is not None:
                outputs = self.ln(outputs, training=False)
            if self.reduction is not None:
                outputs, outputs_length = self.reduction([outputs, outputs_length])
            outputs = self.projection(outputs, training=False)
            return outputs, outputs_length, new_states

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        if self.reduction is not None:
            output_shape, output_length_shape = self.reduction.compute_output_shape((output_shape, output_length_shape))
        output_shape = self.projection.compute_output_shape(output_shape)
        return output_shape, output_length_shape


class RnnTransducerEncoder(Layer):
    def __init__(
        self,
        reduction_factors: list = [6, 0, 0, 0, 0, 0, 0, 0],
        dmodel: int = 640,
        nlayers: int = 8,
        rnn_type: str = "lstm",
        rnn_units: int = 2048,
        rnn_unroll: bool = False,
        layer_norm: bool = True,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(reduction_factors) == nlayers, "reduction_factors length must be equal to nlayers"
        self.reshape = Reshape(name="reshape")

        self.time_reduction_factor = 1
        self.blocks = []
        for i in range(nlayers):
            block = RnnTransducerBlock(
                reduction_factor=reduction_factors[i],
                dmodel=dmodel,
                rnn_type=rnn_type,
                rnn_units=rnn_units,
                rnn_unroll=rnn_unroll,
                layer_norm=layer_norm,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"block_{i}",
            )
            self.blocks.append(block)
            self.time_reduction_factor *= getattr(block.reduction, "time_reduction_factor", 1)

    def get_initial_state(self, batch_size=1):
        """Get zeros states

        Returns:
            tf.Tensor: states having shape [num_rnns, 1 or 2, 1, P]
        """
        states = []
        for block in self.blocks:
            states.append(tf.stack(block.rnn.get_initial_state(tf.zeros([batch_size, 1, 1], dtype=self.dtype)), axis=0))
        return tf.stack(states, axis=0)

    def call(self, inputs, training=False):
        outputs, outputs_length = self.reshape(inputs)
        for block in self.blocks:
            outputs, outputs_length = block((outputs, outputs_length), training=training)
        return outputs, outputs_length

    def call_next(self, features, features_length, previous_encoder_states, *args, **kwargs):
        """
        Recognize function for encoder network from previous encoder states

        Parameters
        ----------
        features : tf.Tensor, shape [B, T, F, C]
        features_length : tf.Tensor, shape [B]
        previous_encoder_states : tf.Tensor, shape [B, nlayers, nstates, rnn_units] -> [nlayers, nstates, B, rnn_units]

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor], shape ([B, T, dmodel], [B], [nlayers, nstates, B, rnn_units] -> [B, nlayers, nstates, rnn_units])
        """
        with tf.name_scope(f"{self.name}_call_next"):
            previous_encoder_states = tf.transpose(previous_encoder_states, perm=[1, 2, 0, 3])
            outputs, outputs_length = self.reshape((features, features_length))
            new_states = []
            for i, block in enumerate(self.blocks):
                outputs, outputs_length, block_states = block.call_next(outputs, outputs_length, previous_encoder_states=previous_encoder_states[i])
                new_states.append(block_states)
            return outputs, outputs_length, tf.transpose(tf.stack(new_states, axis=0), perm=[2, 0, 1, 3])

    def compute_mask(self, inputs, mask=None):
        outputs, outputs_length = inputs
        maxlen = tf.shape(outputs)[1]
        maxlen, outputs_length = (math_util.get_reduced_length(length, self.time_reduction_factor) for length in (maxlen, outputs_length))
        mask = tf.sequence_mask(outputs_length, maxlen=maxlen, dtype=tf.bool)
        return mask, None

    def compute_output_shape(self, input_shape):
        output_shape = self.reshape.compute_output_shape(input_shape)
        for block in self.blocks:
            output_shape = block.compute_output_shape(output_shape)
        return output_shape
