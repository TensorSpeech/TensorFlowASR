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

import tensorflow as tf

from tensorflow_asr.models.base_layer import Layer
from tensorflow_asr.utils import math_util


def _create_num_masked(tensor_mask):
    return tf.vectorized_map(lambda x: math_util.count(tf.cast(x, tf.int32), value=0), elems=tensor_mask, warn=False)


def _shift(tensor, shift):
    shifted_tensor, _ = tf.vectorized_map(lambda x: (tf.roll(x[0], shift=x[1], axis=0), x[1]), elems=(tensor, shift), warn=False)
    return shifted_tensor


class Memory(Layer):
    def __init__(self, batch_size, memory_length, dmodel, **kwargs):
        super().__init__(trainable=False, **kwargs)
        assert memory_length > 0, "memory_length must be integer"
        self.batch_size = batch_size
        self.memory_length = memory_length
        self.dmodel = dmodel
        # self.stateful = True
        # self.memory = tf.Variable(
        #     initial_value=tf.zeros(shape=(self.batch_size, self.memory_length, self.dmodel), dtype=self.dtype),
        #     trainable=False,
        #     name="memory",
        # )
        # self.memory_mask = tf.Variable(
        #     initial_value=tf.zeros(shape=(self.batch_size, self.memory_length), dtype=tf.bool),
        #     trainable=False,
        #     name="memory_mask",
        # )

    def _get_inputs(self, inputs, default_mask_value=1):
        inputs_mask = getattr(inputs, "_keras_mask", None)
        max_length = tf.shape(inputs)[1]
        if inputs_mask is None:
            inputs_mask = tf.cast(tf.ones([self.batch_size, max_length], dtype=tf.int32) * default_mask_value, dtype=tf.bool)
        return inputs, inputs_mask

    def attach_memory(self, inputs, memories=None):
        if memories is None:
            return inputs
        inputs, inputs_mask = self._get_inputs(inputs)
        memory, memory_mask = self._get_inputs(memories, default_mask_value=0)
        # shift memory and stop grad
        memory_shift = _create_num_masked(memory_mask)
        memory = _shift(memory, shift=memory_shift)
        memory = tf.stop_gradient(memory)
        memory_mask = _shift(memory_mask, shift=memory_shift)
        memory_mask = tf.stop_gradient(memory_mask)
        # prepend memory and inputs
        new_inputs = tf.concat([memory, inputs], 1)
        new_inputs._keras_mask = tf.concat([memory_mask, inputs_mask], 1)  # pylint: disable=protected-access
        return new_inputs

    def call(self, inputs, memories=None):
        if memories is None:
            return None
        inputs, inputs_mask = self._get_inputs(inputs)
        memory, memory_mask = self._get_inputs(memories, default_mask_value=0)
        # shift by memory mask
        shift = _create_num_masked(memory_mask)
        new_memory = _shift(memory, shift=shift)
        new_memory_mask = _shift(memory_mask, shift=shift)
        # prepend memory to inputs
        new_memory = tf.concat([new_memory, inputs], 1)
        new_memory_mask = tf.concat([new_memory_mask, inputs_mask], 1)
        # shift by inputs mask
        shift = _create_num_masked(inputs_mask)
        new_memory = _shift(new_memory, shift=shift)
        new_memory_mask = _shift(new_memory_mask, shift=shift)
        # slice combination of memory and inputs into memory_length
        new_memory = tf.slice(
            new_memory,
            begin=[0, tf.shape(new_memory)[1] - self.memory_length, 0],
            size=[-1, self.memory_length, -1],
        )
        new_memory_mask = tf.slice(
            new_memory_mask,
            begin=[0, tf.shape(new_memory_mask)[1] - self.memory_length],
            size=[-1, self.memory_length],
        )
        new_memory._keras_mask = new_memory_mask  # pylint: disable=protected-access
        return new_memory

    # def attach_memory(self, inputs):
    #     inputs, inputs_mask = self._get_inputs(inputs)
    #     # shift memory and stop grad
    #     memory_shift = _create_num_masked(self.memory_mask)
    #     memory = _shift(self.memory, shift=memory_shift)
    #     memory = tf.stop_gradient(memory)
    #     memory_mask = _shift(self.memory_mask, shift=memory_shift)
    #     memory_mask = tf.stop_gradient(memory_mask)
    #     # prepend memory and inputs
    #     new_inputs = tf.concat([memory, inputs], 1)
    #     new_inputs._keras_mask = tf.concat([memory_mask, inputs_mask], 1)  # pylint: disable=protected-access
    #     return new_inputs

    # def get_states(self):
    #     return (self.memory, self.memory_mask)

    # def reset_states(self, states=(None, None)):
    #     memory, memory_mask = states
    #     if memory is None:
    #         memory = tf.zeros(shape=(self.batch_size, self.memory_length, self.dmodel), dtype=self.dtype)
    #     if memory_mask is None:
    #         memory_mask = tf.zeros(shape=(self.batch_size, self.memory_length), dtype=tf.bool)
    #     self.add_update([tf.keras.backend.update(self.memory, memory), tf.keras.backend.update(self.memory_mask, memory_mask)])

    # def call(self, inputs):
    #     inputs, inputs_mask = self._get_inputs(inputs)
    #     # shift by memory mask
    #     shift = _create_num_masked(self.memory_mask)
    #     new_memory = _shift(self.memory, shift=shift)
    #     new_memory_mask = _shift(self.memory_mask, shift=shift)
    #     # prepend memory to inputs
    #     new_memory = tf.concat([new_memory, inputs], 1)
    #     new_memory_mask = tf.concat([new_memory_mask, inputs_mask], 1)
    #     # shift by inputs mask
    #     shift = _create_num_masked(inputs_mask)
    #     new_memory = _shift(new_memory, shift=shift)
    #     new_memory_mask = _shift(new_memory_mask, shift=shift)
    #     # slice combination of memory and inputs into memory_length
    #     new_memory = tf.slice(
    #         new_memory,
    #         begin=[0, tf.shape(new_memory)[1] - self.memory_length, 0],
    #         size=[-1, self.memory_length, -1],
    #     )
    #     new_memory_mask = tf.slice(
    #         new_memory_mask,
    #         begin=[0, tf.shape(new_memory_mask)[1] - self.memory_length],
    #         size=[-1, self.memory_length],
    #     )
    #     self.add_update([tf.keras.backend.update(self.memory, new_memory), tf.keras.backend.update(self.memory_mask, new_memory_mask)])
    #     new_memory._keras_mask = new_memory_mask  # pylint: disable=protected-access
    #     return new_memory

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.memory_length, self.dmodel
