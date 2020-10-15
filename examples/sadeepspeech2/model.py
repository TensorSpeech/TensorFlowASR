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
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_asr.models.layers.positional_encoding import PositionalEncoding
from tensorflow_asr.models.layers.point_wise_ffn import PointWiseFFN
from tensorflow_asr.models.layers.sequence_wise_bn import SequenceBatchNorm
from tensorflow_asr.utils.utils import merge_two_last_dims
from tensorflow_asr.models.ctc import CtcModel

ARCH_CONFIG = {
    "subsampling": {
        "filters": 256,
        "kernel_size": 32,
        "strides": 2
    },
    "att": {
        "layers": 2,
        "head_size": 256,
        "num_heads": 4,
        "ffn_size": 1024,
        "dropout": 0.1
    },
    "rnn": {
        "layers": 2,
        "units": 512,
        "dropout": 0.0
    },
}


def create_sattds2(input_shape: list,
                   arch_config: dict,
                   name: str = "self_attention_ds2"):
    features = tf.keras.Input(shape=input_shape, name="features")
    layer = merge_two_last_dims(features)

    layer = tf.keras.layers.Conv1D(filters=arch_config["subsampling"]["filters"],
                                   kernel_size=arch_config["subsampling"]["kernel_size"],
                                   strides=arch_config["subsampling"]["strides"],
                                   padding="same")(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)

    for i in range(arch_config["att"]["layers"]):
        ffn = tf.keras.layers.LayerNormalization()(layer)

        ffn = PointWiseFFN(size=arch_config["att"]["ffn_size"],
                           output_size=layer.shape[-1],
                           dropout=arch_config["att"]["dropout"],
                           name=f"ffn1_{i}")(ffn)
        layer = tf.keras.layers.Add()([layer, 0.5 * ffn])
        layer = tf.keras.layers.LayerNormalization()(layer)
        pe = PositionalEncoding(name=f"pos_enc_{i}")(layer)
        att = tf.keras.layers.Add(name=f"pos_enc_add_{i}")([layer, pe])
        att = tfa.layers.MultiHeadAttention(head_size=arch_config["att"]["head_size"],
                                            num_heads=arch_config["att"]["num_heads"],
                                            name=f"mulhead_satt_{i}")([att, att, att])
        att = tf.keras.layers.Dropout(arch_config["att"]["dropout"],
                                      name=f"mhsa_dropout_{i}")(att)
        layer = tf.keras.layers.Add()([layer, att])
        ffn = tf.keras.layers.LayerNormalization()(layer)
        ffn = PointWiseFFN(size=arch_config["att"]["ffn_size"],
                           output_size=layer.shape[-1],
                           dropout=arch_config["att"]["dropout"],
                           name=f"ffn2_{i}")(ffn)
        layer = tf.keras.layers.Add()([layer, 0.5 * ffn])

    output = tf.keras.layers.LayerNormalization()(layer)

    for i in range(arch_config["rnn"]["layers"]):
        output = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=arch_config["rnn"]["units"],
                                 dropout=arch_config["rnn"]["dropout"],
                                 return_sequences=True))(output)
        output = SequenceBatchNorm(time_major=False, name=f"seq_bn_{i}")(output)

    return tf.keras.Model(inputs=features, outputs=output, name=name)


class SelfAttentionDS2(CtcModel):
    def __init__(self,
                 input_shape: list,
                 arch_config: dict,
                 num_classes: int,
                 name: str = "self_attention_ds2"):
        super(SelfAttentionDS2, self).__init__(
            base_model=create_sattds2(input_shape=input_shape,
                                      arch_config=arch_config,
                                      name=name),
            num_classes=num_classes,
            name=f"{name}_ctc"
        )
        self.time_reduction_factor = arch_config["subsampling"]["strides"]
