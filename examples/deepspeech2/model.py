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
"""
Read https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
to use cuDNN-LSTM
"""
import numpy as np
import tensorflow as tf

from tiramisu_asr.utils.utils import append_default_keys_dict, get_rnn
from tiramisu_asr.models.layers.row_conv_1d import RowConv1D
from tiramisu_asr.models.layers.sequence_wise_batch_norm import SequenceBatchNorm
from tiramisu_asr.models.layers.transpose_time_major import TransposeTimeMajor
from tiramisu_asr.models.layers.merge_two_last_dims import Merge2LastDims
from tiramisu_asr.models.ctc import CtcModel

DEFAULT_CONV = {
    "conv_type": 2,
    "conv_kernels": ((11, 41), (11, 21), (11, 21)),
    "conv_strides": ((2, 2), (1, 2), (1, 2)),
    "conv_filters": (32, 32, 96),
    "conv_dropout": 0.2
}

DEFAULT_RNN = {
    "rnn_layers": 3,
    "rnn_type": "gru",
    "rnn_units": 350,
    "rnn_activation": "tanh",
    "rnn_bidirectional": True,
    "rnn_rowconv": False,
    "rnn_rowconv_context": 2,
    "rnn_dropout": 0.2
}

DEFAULT_FC = {
    "fc_units": (1024,),
    "fc_dropout": 0.2
}


def create_ds2(input_shape: list, arch_config: dict, name: str = "deepspeech2"):
    conv_conf = append_default_keys_dict(DEFAULT_CONV, arch_config.get("conv_conf", {}))
    rnn_conf = append_default_keys_dict(DEFAULT_RNN, arch_config.get("rnn_conf", {}))
    fc_conf = append_default_keys_dict(DEFAULT_FC, arch_config.get("fc_conf", {}))
    assert len(conv_conf["conv_strides"]) == \
        len(conv_conf["conv_filters"]) == len(conv_conf["conv_kernels"])
    assert conv_conf["conv_type"] in [1, 2]
    assert rnn_conf["rnn_type"] in ["lstm", "gru", "rnn"]
    assert conv_conf["conv_dropout"] >= 0.0 and rnn_conf["rnn_dropout"] >= 0.0

    features = tf.keras.Input(shape=input_shape, name="features")
    layer = features

    if conv_conf["conv_type"] == 2:
        conv = tf.keras.layers.Conv2D
    else:
        layer = Merge2LastDims("conv1d_features")(layer)
        conv = tf.keras.layers.Conv1D
        ker_shape = np.shape(conv_conf["conv_kernels"])
        stride_shape = np.shape(conv_conf["conv_strides"])
        filter_shape = np.shape(conv_conf["conv_filters"])
        assert len(ker_shape) == 1 and len(stride_shape) == 1 and len(filter_shape) == 1

    # CONV Layers
    for i, fil in enumerate(conv_conf["conv_filters"]):
        layer = conv(filters=fil, kernel_size=conv_conf["conv_kernels"][i],
                     strides=conv_conf["conv_strides"][i], padding="same",
                     activation=None, dtype=tf.float32, name=f"cnn_{i}")(layer)
        layer = tf.keras.layers.BatchNormalization(name=f"cnn_bn_{i}")(layer)
        layer = tf.keras.layers.ReLU(name=f"cnn_relu_{i}")(layer)
        layer = tf.keras.layers.Dropout(conv_conf["conv_dropout"],
                                        name=f"cnn_dropout_{i}")(layer)

    if conv_conf["conv_type"] == 2:
        layer = Merge2LastDims("reshape_conv2d_to_rnn")(layer)

    rnn = get_rnn(rnn_conf["rnn_type"])

    # To time major
    if rnn_conf["rnn_bidirectional"]:
        layer = TransposeTimeMajor("transpose_to_time_major")(layer)

    # RNN layers
    for i in range(rnn_conf["rnn_layers"]):
        if rnn_conf["rnn_bidirectional"]:
            layer = tf.keras.layers.Bidirectional(
                rnn(rnn_conf["rnn_units"], activation=rnn_conf["rnn_activation"],
                    time_major=True, dropout=rnn_conf["rnn_dropout"],
                    return_sequences=True, use_bias=True),
                name=f"b{rnn_conf['rnn_type']}_{i}")(layer)
            layer = SequenceBatchNorm(time_major=True, name=f"sequence_wise_bn_{i}")(layer)
        else:
            layer = rnn(rnn_conf["rnn_units"], activation=rnn_conf["rnn_activation"],
                        dropout=rnn_conf["rnn_dropout"], return_sequences=True, use_bias=True,
                        name=f"{rnn_conf['rnn_type']}_{i}")(layer)
            layer = SequenceBatchNorm(time_major=False, name=f"sequence_wise_bn_{i}")(layer)
            if rnn_conf["rnn_rowconv"]:
                layer = RowConv1D(filters=rnn_conf["rnn_units"],
                                  future_context=rnn_conf["rnn_rowconv_context"],
                                  name=f"row_conv_{i}")(layer)

    # To batch major
    if rnn_conf["rnn_bidirectional"]:
        layer = TransposeTimeMajor("transpose_to_batch_major")(layer)

    # FC Layers
    if fc_conf["fc_units"]:
        assert fc_conf["fc_dropout"] >= 0.0

        for idx, units in enumerate(fc_conf["fc_units"]):
            layer = tf.keras.layers.Dense(units=units, activation=None,
                                          use_bias=True, name=f"hidden_fc_{idx}")(layer)
            layer = tf.keras.layers.BatchNormalization(name=f"hidden_fc_bn_{idx}")(layer)
            layer = tf.keras.layers.ReLU(name=f"hidden_fc_relu_{idx}")(layer)
            layer = tf.keras.layers.Dropout(fc_conf["fc_dropout"],
                                            name=f"hidden_fc_dropout_{idx}")(layer)

    return tf.keras.Model(inputs=features, outputs=layer, name=name)


class DeepSpeech2(CtcModel):
    def __init__(self,
                 input_shape: list,
                 arch_config: dict,
                 num_classes: int,
                 name: str = "deepspeech2"):
        super(DeepSpeech2, self).__init__(
            base_model=create_ds2(input_shape=input_shape,
                                  arch_config=arch_config,
                                  name=name),
            num_classes=num_classes,
            name=f"{name}_ctc"
        )
        self.time_reduction_factor = 1
        for s in arch_config["conv_conf"]["conv_strides"]:
            self.time_reduction_factor *= s[0]
