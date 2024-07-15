# Copyright 2023 Huy Le Nguyen (@nglehuy)
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

from tensorflow_asr import keras, tf
from tensorflow_asr.models.base_layer import Layer
from tensorflow_asr.utils import math_util


def _create_num_masked(tensor_mask):
    return tf.vectorized_map(lambda x: math_util.count(tf.cast(x, tf.int32), value=0), elems=tensor_mask, warn=False)


def _shift(tensor, shift):
    shifted_tensor, _ = tf.vectorized_map(lambda x: (tf.roll(x[0], shift=x[1], axis=0), x[1]), elems=(tensor, shift), warn=False)
    return shifted_tensor


@keras.utils.register_keras_serializable(package=__name__)
class Memory(Layer):
    """
    Memory Layer
    This layer `call` method will do 2 things:
        1. prepend memory hidden states to inputs -> new_inputs
        2. concatenating memory and inputs, then slice to memory length -> new_memory
    """

    def __init__(self, memory_length, dmodel, **kwargs):
        super().__init__(trainable=False, **kwargs)
        assert memory_length > 0, "memory_length must be integer"
        self.memory_length = memory_length
        self.dmodel = dmodel

    def _get_inputs(self, inputs, default_mask_value=1):
        inputs_mask = getattr(inputs, "_keras_mask", None)
        if inputs_mask is None:
            batch_size, max_length, *_ = tf.shape(inputs)
            inputs_mask = tf.cast(tf.ones((batch_size, max_length), dtype=tf.int32) * default_mask_value, dtype=tf.bool)
        return inputs, inputs_mask

    def get_initial_state(self, batch_size: int):
        memory = tf.zeros(shape=(batch_size, self.memory_length, self.dmodel), dtype=self.dtype)
        memory._keras_mask = tf.zeros(shape=(batch_size, self.memory_length), dtype=tf.bool)  # pylint: disable=protected-access
        return memory

    def call(self, inputs, memories=None, training=False):
        if memories is None:
            return None
        inputs, inputs_mask = self._get_inputs(inputs)
        memory, memory_mask = self._get_inputs(memories)
        # create new_inputs by prepending memory to inputs
        if training:
            memory = tf.stop_gradient(memory)
            memory_mask = tf.stop_gradient(memory_mask)
        new_inputs = tf.concat([memory, inputs], 1)  # prepend memory and inputs
        new_inputs_mask = tf.concat([memory_mask, inputs_mask], 1)
        new_inputs._keras_mask = new_inputs_mask  # pylint: disable=protected-access
        # create new_memory by slicing new_inputs to memory length
        new_memory = tf.slice(
            new_inputs,
            begin=[0, tf.shape(new_inputs)[1] - self.memory_length, 0],
            size=[-1, self.memory_length, -1],
        )
        new_memory_mask = tf.slice(
            new_inputs_mask,
            begin=[0, tf.shape(new_inputs_mask)[1] - self.memory_length],
            size=[-1, self.memory_length],
        )
        new_memory._keras_mask = new_memory_mask  # pylint: disable=protected-access
        return new_inputs, new_memory

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.memory_length, self.dmodel

    def compute_output_spec(self, *args, **kwargs):
        return super().compute_output_spec(*args, **kwargs)
