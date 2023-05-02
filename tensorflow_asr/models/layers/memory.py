# Copyright 2023 Huy Le Nguyen (@usimarit)
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

from tensorflow_asr.models.base_layer import Layer


class Memory(Layer):
    def __init__(self, batch_size, memory_length, dmodel, **kwargs):
        super().__init__(**kwargs)
        assert memory_length > 0, "memory_length must be integer"
        self.batch_size = batch_size
        self.memory_length = memory_length
        self.dmodel = dmodel
        self.stateful = True
        self.memory = self.add_weight(
            shape=(self.batch_size, self.memory_length, self.dmodel),
            initializer="zeros",
            trainable=False,
            dtype=self.dtype,
            name="memory",
        )
        self.memory_mask = self.add_weight(
            shape=(self.batch_size, self.memory_length),
            initializer="ones",
            trainable=False,
            dtype=tf.bool,
            name="memory_mask",
        )

    def _attach_memory_item(self, item):
        (
            per_batch_memory,  # [M, D]
            per_batch_memory_mask,  # [M]
            per_batch_input,  # [T, D]
            per_batch_input_mask,  # [T]
        ) = item

        total_length = tf.cast(tf.shape(per_batch_input)[0] + self.memory_length, tf.int32)
        per_batch_real_memory = tf.boolean_mask(per_batch_memory, per_batch_memory_mask)
        per_batch_real_input = tf.boolean_mask(per_batch_input, per_batch_input_mask)
        per_batch_new_inputs = tf.concat([tf.stop_gradient(per_batch_real_memory), per_batch_real_input], 0)  # [m + t, D]
        real_length = tf.cast(tf.shape(per_batch_new_inputs)[0], tf.int32)
        per_batch_new_inputs = tf.pad(per_batch_new_inputs, paddings=[[0, tf.maximum(total_length - real_length, 0)], [0, 0]])
        per_batch_new_inputs_mask = tf.sequence_mask(real_length, total_length)
        return per_batch_new_inputs, per_batch_new_inputs_mask

    def attach_memory(self, inputs):
        inputs_mask = getattr(inputs, "_keras_mask", None)
        max_length = tf.shape(inputs)[1]
        if inputs_mask is None:
            inputs_mask = tf.ones([self.batch_size, max_length], dtype=tf.bool)
        memory = tf.stop_gradient(tf.cast(self.memory, inputs.dtype))
        memory_mask = tf.stop_gradient(self.memory_mask)
        new_inputs, new_inputs_mask = tf.map_fn(
            self._attach_memory_item,
            elems=[memory, memory_mask, inputs, inputs_mask],
            fn_output_signature=[
                tf.TensorSpec([None if inputs.shape[1] is None else inputs.shape[1] + self.memory_length, self.dmodel], dtype=inputs.dtype),
                tf.TensorSpec([None if inputs.shape[1] is None else inputs.shape[1] + self.memory_length], dtype=tf.bool),
            ],
        )
        new_inputs._keras_mask = new_inputs_mask  # pylint: disable=protected-access
        return new_inputs

    def _update_memory_item(self, item):
        (
            per_batch_memory,  # [M, D]
            per_batch_memory_mask,  # [M]
            per_batch_input,  # [T, D]
            per_batch_input_mask,  # [T]
        ) = item

        memory_length = tf.convert_to_tensor(self.memory_length, tf.int32)
        per_batch_real_memory = tf.boolean_mask(per_batch_memory, per_batch_memory_mask)
        per_batch_real_input = tf.boolean_mask(per_batch_input, per_batch_input_mask)
        per_batch_new_memory = tf.concat([per_batch_real_memory, per_batch_real_input], 0)  # [m + t, D]
        real_memory_length = tf.cast(tf.shape(per_batch_new_memory)[0], tf.int32)
        per_batch_new_memory = tf.slice(
            per_batch_new_memory,
            begin=[tf.maximum(real_memory_length - memory_length, 0), 0],
            size=[tf.minimum(memory_length, real_memory_length), self.dmodel],
        )  # [M, D] if M > m+t else [m+t, D]
        real_memory_length = tf.cast(tf.shape(per_batch_new_memory)[0], tf.int32)
        per_batch_new_memory = tf.pad(per_batch_new_memory, paddings=[[0, memory_length - real_memory_length], [0, 0]])
        per_batch_new_memory_mask = tf.sequence_mask(real_memory_length, memory_length)  # [M]
        return per_batch_new_memory, per_batch_new_memory_mask

    def call(self, inputs):
        inputs_mask = getattr(inputs, "_keras_mask", None)
        if inputs_mask is None:
            inputs_mask = tf.ones([self.batch_size, tf.shape(inputs)[1]], dtype=tf.bool)
        new_memory, new_memory_mask = tf.map_fn(
            self._update_memory_item,
            elems=[self.memory, self.memory_mask, inputs, inputs_mask],
            fn_output_signature=[
                tf.TensorSpec([self.memory_length, self.dmodel], dtype=inputs.dtype),
                tf.TensorSpec([self.memory_length], dtype=tf.bool),
            ],
        )
        self.add_update([tf.keras.backend.update(self.memory, new_memory), tf.keras.backend.update(self.memory_mask, new_memory_mask)])
        new_memory._keras_mask = new_memory_mask  # pylint: disable=protected-access
        return new_memory

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.memory_length, self.dmodel
