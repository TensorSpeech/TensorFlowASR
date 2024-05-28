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

from tensorflow_asr import keras, tf
from tensorflow_asr.models.base_layer import Layer, Reshape
from tensorflow_asr.models.layers.subsampling import TimeReduction
from tensorflow_asr.utils import layer_util, math_util


@keras.utils.register_keras_serializable(package=__name__)
class RnnTransducerBlock(Layer):
    def __init__(
        self,
        reduction_position: str = "pre",
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
        assert reduction_position in ["post", "pre"], "reduction_position must be 'post' or 'pre'"
        self._reduction_position = reduction_position
        self.reduction = TimeReduction(reduction_factor, name="reduction", dtype=self.dtype) if reduction_factor > 0 else None
        self.rnn = layer_util.get_rnn(rnn_type)(
            units=rnn_units,
            return_sequences=True,
            name=rnn_type,
            unroll=rnn_unroll,
            return_state=True,
            zero_output_for_mask=True,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=self.dtype,
        )
        self.ln = (
            keras.layers.LayerNormalization(name="ln", gamma_regularizer=kernel_regularizer, beta_regularizer=bias_regularizer, dtype=self.dtype)
            if layer_norm
            else None
        )
        self.projection = keras.layers.Dense(
            dmodel,
            name="projection",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dtype=self.dtype,
        )

    def call(self, inputs, training=False):
        outputs, outputs_length = inputs
        if self._reduction_position == "pre":
            if self.reduction is not None:
                outputs, outputs_length = self.reduction((outputs, outputs_length))
        outputs, *_ = self.rnn(outputs, training=training)
        if self.ln is not None:
            outputs = self.ln(outputs, training=training)
        outputs = self.projection(outputs, training=training)
        if self._reduction_position == "post":
            if self.reduction is not None:
                outputs, outputs_length = self.reduction((outputs, outputs_length))
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
            if self._reduction_position == "pre":
                if self.reduction is not None:
                    outputs, outputs_length = self.reduction([outputs, outputs_length])
            outputs, *_states = self.rnn(
                outputs,
                training=False,
                initial_state=tf.unstack(previous_encoder_states, axis=0),
                mask=getattr(inputs, "_keras_mask", None),
            )
            new_states = tf.stack(_states, axis=0)
            if self.ln is not None:
                outputs = self.ln(outputs, training=False)
            outputs = self.projection(outputs, training=False)
            if self._reduction_position == "post":
                if self.reduction is not None:
                    outputs, outputs_length = self.reduction([outputs, outputs_length])
            return outputs, outputs_length, new_states

    def compute_output_shape(self, input_shape):
        output_shape, output_length_shape = input_shape
        if self.reduction is not None:
            output_shape, output_length_shape = self.reduction.compute_output_shape((output_shape, output_length_shape))
        output_shape = self.projection.compute_output_shape(output_shape)
        return output_shape, output_length_shape


@keras.utils.register_keras_serializable(package=__name__)
class RnnTransducerEncoder(Layer):
    def __init__(
        self,
        reduction_positions: list = ["pre", "pre", "pre", "pre", "pre", "pre", "pre", "pre"],
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
        assert len(reduction_positions) == nlayers, "reduction_positions length must be equal to nlayers"
        assert len(reduction_factors) == nlayers, "reduction_factors length must be equal to nlayers"
        self.reshape = Reshape(name="reshape", dtype=self.dtype)

        self.time_reduction_factor = 1
        self.blocks = []
        for i in range(nlayers):
            block = RnnTransducerBlock(
                reduction_position=reduction_positions[i],
                reduction_factor=reduction_factors[i],
                dmodel=dmodel,
                rnn_type=rnn_type,
                rnn_units=rnn_units,
                rnn_unroll=rnn_unroll,
                layer_norm=layer_norm,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"block_{i}",
                dtype=self.dtype,
            )
            self.blocks.append(block)
            self.time_reduction_factor *= getattr(block.reduction, "time_reduction_factor", 1)

    def get_initial_state(self, batch_size=1):
        """Get zeros states

        Returns:
        tf.Tensor, shape [B, num_rnns, nstates, state_size]
            Zero initialized states
        """
        states = []
        for block in self.blocks:
            states.append(tf.stack(block.rnn.get_initial_state(tf.zeros([batch_size, 1, 1], dtype=self.dtype)), axis=0))
        return tf.transpose(tf.stack(states, axis=0), perm=[2, 0, 1, 3])

    def call(self, inputs, training=False):
        outputs, outputs_length, caching = inputs
        outputs, outputs_length = self.reshape((outputs, outputs_length))
        for block in self.blocks:
            outputs, outputs_length = block((outputs, outputs_length), training=training)
        return outputs, outputs_length, caching

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
        outputs, outputs_length, caching = inputs
        maxlen = tf.shape(outputs)[1]
        maxlen, outputs_length = (math_util.get_reduced_length(length, self.time_reduction_factor) for length in (maxlen, outputs_length))
        mask = tf.sequence_mask(outputs_length, maxlen=maxlen, dtype=tf.bool)
        return mask, None, getattr(caching, "_keras_mask", None)

    def compute_output_shape(self, input_shape):
        *output_shape, caching_shape = input_shape
        output_shape = self.reshape.compute_output_shape(output_shape)
        for block in self.blocks:
            output_shape = block.compute_output_shape(output_shape)
        return *output_shape, caching_shape
