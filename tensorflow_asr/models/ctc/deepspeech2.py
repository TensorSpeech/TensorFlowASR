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

from ...utils import layer_util, math_util
from ..layers.row_conv_1d import RowConv1D
from ..layers.sequence_wise_bn import SequenceBatchNorm
from .ctc import CtcModel


class Reshape(tf.keras.layers.Layer):
    def call(self, inputs): return math_util.merge_two_last_dims(inputs)


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 conv_type: str = "conv2d",
                 kernels: list = [11, 41],
                 strides: list = [2, 2],
                 filters: int = 32,
                 dropout: float = 0.1,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        CNN = layer_util.get_conv(conv_type)
        self.conv = CNN(filters=filters, kernel_size=kernels,
                        strides=strides, padding="same",
                        dtype=tf.float32, name=f"{self.name}_{conv_type}")
        self.bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn")
        self.relu = tf.keras.layers.ReLU(name=f"{self.name}_relu")
        self.do = tf.keras.layers.Dropout(dropout, name=f"{self.name}_dropout")

    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv(inputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.relu(outputs, training=training)
        outputs = self.do(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(ConvBlock, self).get_config()
        conf.update(self.conv.get_config())
        conf.update(self.bn.get_config())
        conf.update(self.relu.get_config())
        conf.update(self.do.get_config())
        return conf


class ConvModule(tf.keras.Model):
    def __init__(self,
                 conv_type: str = "conv2d",
                 kernels: list = [[11, 41], [11, 21], [11, 21]],
                 strides: list = [[2, 2], [1, 2], [1, 2]],
                 filters: list = [32, 32, 96],
                 dropout: float = 0.1,
                 **kwargs):
        super(ConvModule, self).__init__(**kwargs)

        assert len(kernels) == len(strides) == len(filters)
        assert dropout >= 0.0

        self.preprocess = None  # reshape from [B, T, F, C] to [B, T, F * C]
        if conv_type == "conv1d": self.preprocess = Reshape(name=f"{self.name}_preprocess")

        self.blocks = [
            ConvBlock(
                conv_type=conv_type,
                kernels=kernels[i],
                strides=strides[i],
                filters=filters[i],
                dropout=dropout,
                name=f"{self.name}_block_{i}"
            ) for i in range(len(filters))
        ]

        self.postprocess = None  # reshape from [B, T, F, C] to [B, T, F * C]
        if conv_type == "conv2d": self.postprocess = Reshape(name=f"{self.name}_postprocess")

        self.reduction_factor = 1
        for s in strides: self.reduction_factor *= s[0]

    def call(self, inputs, training=False, **kwargs):
        outputs = inputs
        if self.preprocess is not None: outputs = self.preprocess(outputs)
        for block in self.blocks:
            outputs = block(outputs, training=training, **kwargs)
        if self.postprocess is not None: outputs = self.postprocess(outputs)
        return outputs

    def get_config(self):
        conf = {}
        conf.update(self.preprocess.get_config())
        for block in self.blocks:
            conf.update(block.get_config())
        conf.update(self.postprocess.get_config())
        return conf


class RnnBlock(tf.keras.layers.Layer):
    def __init__(self,
                 rnn_type: str = "lstm",
                 units: int = 1024,
                 bidirectional: bool = True,
                 rowconv: int = 0,
                 dropout: float = 0.1,
                 **kwargs):
        super(RnnBlock, self).__init__(**kwargs)

        RNN = layer_util.get_rnn(rnn_type)
        self.rnn = RNN(units, dropout=dropout, return_sequences=True,
                       use_bias=True, name=f"{self.name}_{rnn_type}")
        if bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(self.rnn, name=f"{self.name}_b{rnn_type}")
        self.bn = SequenceBatchNorm(time_major=False, name=f"{self.name}_bn")
        self.rowconv = None
        if not bidirectional and rowconv > 0:
            self.rowconv = RowConv1D(filters=units, future_context=rowconv,
                                     name=f"{self.name}_rowconv")

    def call(self, inputs, training=False, **kwargs):
        outputs = self.rnn(inputs, training=training)
        outputs = self.bn(outputs, training=training)
        if self.rowconv is not None:
            outputs = self.rowconv(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(RnnBlock, self).get_config()
        conf.update(self.rnn.get_config())
        conf.update(self.bn.get_config())
        if self.rowconv is not None:
            conf.update(self.rowconv.get_config())
        return conf


class RnnModule(tf.keras.Model):
    def __init__(self,
                 nlayers: int = 5,
                 rnn_type: str = "lstm",
                 units: int = 1024,
                 bidirectional: bool = True,
                 rowconv: int = 0,
                 dropout: float = 0.1,
                 **kwargs):
        super(RnnModule, self).__init__(**kwargs)

        self.blocks = [
            RnnBlock(
                rnn_type=rnn_type,
                units=units,
                bidirectional=bidirectional,
                rowconv=rowconv,
                dropout=dropout,
                name=f"{self.name}_block_{i}"
            ) for i in range(nlayers)
        ]

    def call(self, inputs, training=False, **kwargs):
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs, training=training, **kwargs)
        return outputs

    def get_config(self):
        conf = {}
        for block in self.blocks:
            conf.update(block.get_config())
        return conf


class FcBlock(tf.keras.layers.Layer):
    def __init__(self,
                 units: int = 1024,
                 dropout: float = 0.1,
                 **kwargs):
        super(FcBlock, self).__init__(**kwargs)

        self.fc = tf.keras.layers.Dense(units, name=f"{self.name}_fc")
        self.bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn")
        self.relu = tf.keras.layers.ReLU(name=f"{self.name}_relu")
        self.do = tf.keras.layers.Dropout(dropout, name=f"{self.name}_dropout")

    def call(self, inputs, training=False, **kwargs):
        outputs = self.fc(inputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.relu(outputs, training=training)
        outputs = self.do(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(FcBlock, self).get_config()
        conf.update(self.fc.get_config())
        conf.update(self.bn.get_config())
        conf.update(self.relu.get_config())
        conf.update(self.do.get_config())
        return conf


class FcModule(tf.keras.Model):
    def __init__(self,
                 nlayers: int = 0,
                 units: int = 1024,
                 dropout: float = 0.1,
                 **kwargs):
        super(FcModule, self).__init__(**kwargs)

        self.blocks = [
            FcBlock(
                units=units,
                dropout=dropout,
                name=f"{self.name}_block_{i}"
            ) for i in range(nlayers)
        ]

    def call(self, inputs, training=False, **kwargs):
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs, training=training, **kwargs)
        return outputs

    def get_config(self):
        conf = {}
        for block in self.blocks:
            conf.update(block.get_config())
        return conf


class DeepSpeech2Encoder(tf.keras.Model):
    def __init__(self,
                 conv_type: str = "conv2d",
                 conv_kernels: list = [[11, 41], [11, 21], [11, 21]],
                 conv_strides: list = [[2, 2], [1, 2], [1, 2]],
                 conv_filters: list = [32, 32, 96],
                 conv_dropout: float = 0.1,
                 rnn_nlayers: int = 5,
                 rnn_type: str = "lstm",
                 rnn_units: int = 1024,
                 rnn_bidirectional: bool = True,
                 rnn_rowconv: int = 0,
                 rnn_dropout: float = 0.1,
                 fc_nlayers: int = 0,
                 fc_units: int = 1024,
                 fc_dropout: float = 0.1,
                 name="deepspeech2_encoder",
                 **kwargs):
        super().__init__(**kwargs)

        self.conv_module = ConvModule(
            conv_type=conv_type,
            kernels=conv_kernels,
            strides=conv_strides,
            filters=conv_filters,
            dropout=conv_dropout,
            name=f"{self.name}_conv_module"
        )

        self.rnn_module = RnnModule(
            nlayers=rnn_nlayers,
            rnn_type=rnn_type,
            units=rnn_units,
            bidirectional=rnn_bidirectional,
            rowconv=rnn_rowconv,
            dropout=rnn_dropout,
            name=f"{self.name}_rnn_module"
        )

        self.fc_module = FcModule(
            nlayers=fc_nlayers,
            units=fc_units,
            dropout=fc_dropout,
            name=f"{self.name}_fc_module"
        )

    def summary(self, line_length=100, **kwargs):
        self.conv_module.summary(line_length=line_length, **kwargs)
        self.rnn_module.summary(line_length=line_length, **kwargs)
        self.fc_module.summary(line_length=line_length, **kwargs)
        super().summary(line_length=line_length, **kwargs)

    def call(self, inputs, training, **kwargs):
        outputs = self.conv_module(inputs, training=training, **kwargs)
        outputs = self.rnn_module(outputs, training=training, **kwargs)
        outputs = self.fc_module(outputs, training=training, **kwargs)
        return outputs

    def get_config(self):
        conf = super().get_config()
        conf.update(self.conv_module.get_config())
        conf.update(self.rnn_module.get_config())
        conf.update(self.fc_module.get_config())
        return conf


class DeepSpeech2(CtcModel):
    def __init__(self,
                 vocabulary_size: int,
                 conv_type: str = "conv2d",
                 conv_kernels: list = [[11, 41], [11, 21], [11, 21]],
                 conv_strides: list = [[2, 2], [1, 2], [1, 2]],
                 conv_filters: list = [32, 32, 96],
                 conv_dropout: float = 0.1,
                 rnn_nlayers: int = 5,
                 rnn_type: str = "lstm",
                 rnn_units: int = 1024,
                 rnn_bidirectional: bool = True,
                 rnn_rowconv: int = 0,
                 rnn_dropout: float = 0.1,
                 fc_nlayers: int = 0,
                 fc_units: int = 1024,
                 fc_dropout: float = 0.1,
                 name: str = "deepspeech2",
                 **kwargs):
        super().__init__(
            encoder=DeepSpeech2Encoder(
                conv_type=conv_type,
                conv_kernels=conv_kernels,
                conv_strides=conv_strides,
                conv_filters=conv_filters,
                conv_dropout=conv_dropout,
                rnn_nlayers=rnn_nlayers,
                rnn_type=rnn_type,
                rnn_units=rnn_units,
                rnn_bidirectional=rnn_bidirectional,
                rnn_rowconv=rnn_rowconv,
                rnn_dropout=rnn_dropout,
                fc_nlayers=fc_nlayers,
                fc_units=fc_units,
                fc_dropout=fc_dropout,
                name=f"{name}_encoder"
            ),
            vocabulary_size=vocabulary_size,
            name=name,
            **kwargs
        )
        self.time_reduction_factor = self.encoder.conv_module.reduction_factor
