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

from .ctc import CtcModel
from ..deepspeech2 import ConvModule, RnnModule, FcModule


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
        super(DeepSpeech2, self).__init__(name=name, **kwargs)

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
            vocabulary_size=vocabulary_size,
            name=f"{self.name}_fc_module"
        )

        self.time_reduction_factor = self.conv_module.reduction_factor

    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv_module(inputs, training=training, **kwargs)
        outputs = self.rnn_module(outputs, training=training, **kwargs)
        outputs = self.fc_module(outputs, training=training, **kwargs)
        return outputs

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
