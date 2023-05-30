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


class LSTM(tf.keras.layers.LSTM):
    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0,
        recurrent_dropout=0,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        time_major=False,
        unroll=False,
        **kwargs
    ):
        super().__init__(
            units,
            activation,
            recurrent_activation,
            use_bias,
            kernel_initializer,
            recurrent_initializer,
            bias_initializer,
            unit_forget_bias,
            kernel_regularizer,
            recurrent_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            recurrent_constraint,
            bias_constraint,
            dropout,
            recurrent_dropout,
            return_sequences,
            return_state,
            go_backwards,
            stateful,
            time_major,
            unroll,
            **kwargs
        )
        self._could_use_gpu_kernel = self._could_use_gpu_kernel and tf.keras.mixed_precision.global_policy().name != "mixed_bfloat16"


class GRU(tf.keras.layers.GRU):
    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0,
        recurrent_dropout=0,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        time_major=False,
        reset_after=True,
        **kwargs
    ):
        super().__init__(
            units,
            activation,
            recurrent_activation,
            use_bias,
            kernel_initializer,
            recurrent_initializer,
            bias_initializer,
            kernel_regularizer,
            recurrent_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            recurrent_constraint,
            bias_constraint,
            dropout,
            recurrent_dropout,
            return_sequences,
            return_state,
            go_backwards,
            stateful,
            unroll,
            time_major,
            reset_after,
            **kwargs
        )
        self._could_use_gpu_kernel = self._could_use_gpu_kernel and tf.keras.mixed_precision.global_policy().name != "mixed_bfloat16"
