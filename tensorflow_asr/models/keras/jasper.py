# Copyright 2020 Huy Le Nguyen (@usimarit)
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

from .ctc import CtcModel
from ..jasper import Reshape, JasperBlock, JasperSubBlock


class Jasper(CtcModel):
    def __init__(self,
                 vocabulary_size: int,
                 dense: bool = False,
                 first_additional_block_channels: int = 256,
                 first_additional_block_kernels: int = 11,
                 first_additional_block_strides: int = 2,
                 first_additional_block_dilation: int = 1,
                 first_additional_block_dropout: int = 0.2,
                 nsubblocks: int = 5,
                 block_channels: list = [256, 384, 512, 640, 768],
                 block_kernels: list = [11, 13, 17, 21, 25],
                 block_dropout: list = [0.2, 0.2, 0.2, 0.3, 0.3],
                 second_additional_block_channels: int = 896,
                 second_additional_block_kernels: int = 1,
                 second_additional_block_strides: int = 1,
                 second_additional_block_dilation: int = 2,
                 second_additional_block_dropout: int = 0.4,
                 third_additional_block_channels: int = 1024,
                 third_additional_block_kernels: int = 1,
                 third_additional_block_strides: int = 1,
                 third_additional_block_dilation: int = 1,
                 third_additional_block_dropout: int = 0.4,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name: str = "jasper",
                 **kwargs):
        super(Jasper, self).__init__(name=name, **kwargs)

        assert len(block_channels) == len(block_kernels) == len(block_dropout)

        self.reshape = Reshape(name=f"{self.name}_reshape")

        self.first_additional_block = JasperSubBlock(
            channels=first_additional_block_channels,
            kernels=first_additional_block_kernels,
            strides=first_additional_block_strides,
            dropout=first_additional_block_dropout,
            dilation=first_additional_block_dilation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_first_block"
        )

        self.blocks = [
            JasperBlock(
                nsubblocks=nsubblocks,
                channels=block_channels[i],
                kernels=block_kernels[i],
                dropout=block_dropout[i],
                dense=dense,
                nresiduals=(i + 1) if dense else 1,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{self.name}_block_{i}"
            ) for i in range(len(block_channels))
        ]

        self.second_additional_block = JasperSubBlock(
            channels=second_additional_block_channels,
            kernels=second_additional_block_kernels,
            strides=second_additional_block_strides,
            dropout=second_additional_block_dropout,
            dilation=second_additional_block_dilation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_second_block"
        )

        self.third_additional_block = JasperSubBlock(
            channels=third_additional_block_channels,
            kernels=third_additional_block_kernels,
            strides=third_additional_block_strides,
            dropout=third_additional_block_dropout,
            dilation=third_additional_block_dilation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_third_block"
        )

        self.last_block = tf.keras.layers.Conv1D(
            filters=vocabulary_size, kernel_size=1,
            strides=1, padding="same",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_last_block"
        )

        self.time_reduction_factor = self.first_additional_block.reduction_factor
        self.time_reduction_factor *= self.second_additional_block.reduction_factor
        self.time_reduction_factor *= self.third_additional_block.reduction_factor

    def call(self, inputs, training=False, **kwargs):
        outputs = self.reshape(inputs)
        outputs = self.first_additional_block(outputs, training=training, **kwargs)

        residuals = []
        for block in self.blocks:
            outputs, residuals = block([outputs, residuals], training=training, **kwargs)

        outputs = self.second_additional_block(outputs, training=training, **kwargs)
        outputs = self.third_additional_block(outputs, training=training, **kwargs)
        outputs = self.last_block(outputs, training=training, **kwargs)
        return outputs

    def summary(self, line_length=100, **kwargs):
        super(Jasper, self).summary(line_length=line_length, **kwargs)

    def get_config(self):
        conf = self.reshape.get_config()
        conf.update(self.first_additional_block.get_config())
        for block in self.blocks:
            conf.update(block.get_config())
        conf.update(self.second_additional_block.get_config())
        conf.update(self.third_additional_block.get_config())
        conf.update(self.last_block.get_config())
        return conf
