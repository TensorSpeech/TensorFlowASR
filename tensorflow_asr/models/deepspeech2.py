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

from ..utils.utils import get_rnn, get_conv, merge_two_last_dims
from .layers.row_conv_1d import RowConv1D
from .layers.sequence_wise_bn import SequenceBatchNorm
from .ctc import CtcModel


class Reshape(tf.keras.layers.Layer):
    def call(self, inputs): return merge_two_last_dims(inputs)


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
        if conv_type == "conv2d": self.preprocess = Reshape(name=f"{self.name}_preprocess")

        CNN = get_conv(conv_type)
        self.blocks = []
        for i, fil in enumerate(filters):
            conv = CNN(filters=fil, kernel_size=kernels[i],
                       strides=strides[i], padding="same",
                       dtype=tf.float32, name=f"{self.name}_cnn_{i}")
            bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn_{i}")
            relu = tf.keras.layers.ReLU(name=f"{self.name}_relu_{i}")
            do = tf.keras.layers.Dropout(dropout, name=f"{self.name}_dropout_{i}")
            self.blocks.append({"conv": conv, "bn": bn, "relu": relu, "do": do})

        self.postprocess = None  # reshape from [B, T, F, C] to [B, T, F * C]
        if conv_type == "conv1d": self.postprocess = Reshape(name=f"{self.name}_postprocess")

        self.reduction_factor = 1
        for s in strides: self.reduction_factor *= s[0]

    def call(self, inputs, training=False):
        outputs = inputs
        if self.preprocess is not None: outputs = self.preprocess(outputs)
        for block in self.blocks:
            outputs = block["conv"](outputs, training=training)
            outputs = block["bn"](outputs, training=training)
            outputs = block["relu"](outputs, training=training)
            outputs = block["do"](outputs, training=training)
        if self.postprocess is not None: outputs = self.postprocess(outputs)
        return outputs

    def get_config(self):
        conf = {}
        conf.update(self.preprocess.get_config())
        for block in self.blocks:
            for value in block.values():
                conf.update(value.get_config())
        conf.update(self.postprocess.get_config())
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

        RNN = get_rnn(rnn_type)
        self.blocks = []
        for i in range(nlayers):
            rnn = RNN(units, dropout=dropout, return_sequences=True,
                      use_bias=True, name=f"{self.name}_{rnn_type}_{i}")
            if bidirectional:
                rnn = tf.keras.layers.Bidirectional(rnn, name=f"{self.name}_b{rnn_type}_{i}")
            bn = SequenceBatchNorm(time_major=False, name=f"{self.name}_bn_{i}")
            rowconv = None
            if not bidirectional and rowconv > 0:
                rowconv = RowConv1D(filters=units, future_context=rowconv,
                                    name=f"{self.name}_rowconv_{i}")
            self.blocks.append({"rnn": rnn, "bn": bn, "rowconv": rowconv})

    def call(self, inputs, training=False):
        outputs = inputs
        for block in self.blocks:
            outputs = block["rnn"](outputs, training=training)
            outputs = block["bn"](outputs, training=training)
            if block["rowconv"] is not None:
                outputs = block["rowconv"](outputs, training=training)
        return outputs

    def get_config(self):
        conf = {}
        for block in self.blocks:
            for value in block.values():
                if value is not None: conf.update(value.get_config())
        return conf


class FCModule(tf.keras.Model):
    def __init__(self,
                 nlayers: int = 0,
                 units: int = 1024,
                 dropout: float = 0.1,
                 **kwargs):
        super(FCModule, self).__init__(**kwargs)

        self.blocks = []
        for i in range(nlayers):
            fc = tf.keras.layers.Dense(units, name=f"{self.name}_fc_{i}")
            bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn_{i}")
            relu = tf.keras.layers.ReLU(name=f"{self.name}_relu_{i}")
            do = tf.keras.layers.Dropout(dropout, name=f"{self.name}_dropout_{i}")
            self.blocks.append({"fc": fc, "bn": bn, "relu": relu, "do": do})

    def call(self, inputs, training=False):
        outputs = inputs
        for block in self.blocks:
            outputs = block["fc"](outputs, training=training)
            outputs = block["bn"](outputs, training=training)
            outputs = block["relu"](outputs, training=training)
            outputs = block["do"](outputs, training=training)
        return outputs


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
                 name: str = "deepspeech2"):
        super(DeepSpeech2, self).__init__(vocabulary_size=vocabulary_size, name=name)

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

        self.fc_module = FCModule(
            nlayers=fc_nlayers,
            units=fc_units,
            dropout=fc_dropout,
            name=f"{self.name}_fc_module"
        )

        self.time_reduction_factor = self.conv_module.reduction_factor

    def call(self, inputs, training=False):
        outputs = self.conv_module(inputs, training=training)
        outputs = self.rnn_module(outputs, training=training)
        outputs = self.fc_module(outputs, training=training)
        return super(DeepSpeech2, self).call(outputs, training=training)

    def summary(self, line_length=100, **kwargs):
        self.conv_module.summary(line_length=line_length, **kwargs)
        self.rnn_module.summary(line_length=line_length, **kwargs)
        self.fc_module.summary(line_length=line_length, **kwargs)
        super(DeepSpeech2, self).summary(line_length=line_length, **kwargs)

    def get_config(self):
        conf = super(DeepSpeech2, self).get_config()
        conf.update(self.conv_module.get_config())
        conf.update(self.rnn_module.get_config())
        conf.update(self.fc_module.get_config())
        return conf
