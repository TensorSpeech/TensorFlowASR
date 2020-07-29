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

from .activations import GLU
from .transducer import Transducer
from ..utils.utils import merge_two_last_dims
from .layers.positional_encoding import PositionalEncoding
from .layers.multihead_attention import MultiHeadAttention


class Conv2DSubsampling(tf.keras.layers.Layer):
    def __init__(self,
                 odim: int,
                 reduction_factor: int,
                 name="conv2dsubsampling",
                 **kwargs):
        super(Conv2DSubsampling, self).__init__(name=name, **kwargs)
        assert reduction_factor % 2 == 0, "reduction_factor must be divisible by 2"
        strides = (reduction_factor // 2, reduction_factor // 2)
        self.conv1 = tf.keras.layers.Conv2D(filters=odim, kernel_size=(3, 3),
                                            strides=strides, padding="same")
        self.relu1 = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=odim, kernel_size=(3, 3),
                                            strides=strides, padding="same")
        self.relu2 = tf.keras.layers.ReLU()
        self.linear = tf.keras.layers.Dense(odim)

    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv1(inputs, training=training)
        outputs = self.relu1(outputs)
        outputs = self.conv2(inputs, training=training)
        outputs = self.relu2(outputs)
        outputs = merge_two_last_dims(outputs)
        return self.linear(outputs, training=training)

    def get_config(self):
        conf = super(Conv2DSubsampling, self).get_config()
        conf.update(self.conv1.get_config())
        conf.update(self.relu1.get_config())
        conf.update(self.conv2.get_config())
        conf.update(self.relu2.get_config())
        conf.update(self.linear.get_config())
        return conf


class FFModule(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 dropout=0.0,
                 fc_factor=0.5,
                 name="ff_module",
                 **kwargs):
        super(FFModule, self).__init__(name=name, **kwargs)
        self.fc_factor = fc_factor
        self.ln = tf.keras.layers.LayerNormalization()
        self.ffn1 = tf.keras.layers.Dense(4 * input_dim)
        self.swish = tf.keras.layers.Activation(
            tf.keras.activations.swish, name="swish_activation")
        self.do1 = tf.keras.layers.Dropout(dropout)
        self.ffn2 = tf.keras.layers.Dense(input_dim)
        self.do2 = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()

    def call(self, inputs, training=False, **kwargs):
        outputs = self.ln(inputs, training=training)
        outputs = self.ffn1(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.res_add([inputs, self.fc_factor * outputs])
        return outputs

    def get_config(self):
        conf = super(FFModule, self).get_config()
        conf.update({"fc_factor": self.fc_factor})
        conf.update(self.ln.get_config())
        conf.update(self.ffn1.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.do1.get_config())
        conf.update(self.ffn2.get_config())
        conf.update(self.do2.get_config())
        conf.update(self.res_add.get_config())
        return conf


class MHSAModule(tf.keras.layers.Layer):
    def __init__(self,
                 head_size,
                 num_heads,
                 dropout=0.0,
                 name="mhsa_module",
                 **kwargs):
        super(MHSAModule, self).__init__(name=name, **kwargs)
        self.pc = PositionalEncoding()
        self.ln = tf.keras.layers.LayerNormalization()
        self.mha = MultiHeadAttention(head_size=head_size, num_heads=num_heads)
        self.do = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()

    def call(self, inputs, training=False, **kwargs):
        outputs = self.pc(inputs)
        outputs = self.ln(outputs, training=training)
        outputs = self.mha([outputs, outputs, outputs], training=training)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(MHSAModule, self).get_config()
        conf.update(self.pc.get_config())
        conf.update(self.ln.get_config())
        conf.update(self.mha.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf


class ConvModule(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 kernel_size=32,
                 dropout=0.0,
                 name="conv_module",
                 **kwargs):
        super(ConvModule, self).__init__(name=name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization()
        self.pw_conv_1 = tf.keras.layers.Conv1D(
            filters=2 * input_dim, kernel_size=1, strides=1,
            padding="same", name="pw_conv_1"
        )
        self.glu = GLU()
        self.dw_conv = tf.keras.layers.SeparableConv1D(
            filters=2 * input_dim, kernel_size=kernel_size, strides=1,
            padding="same", depth_multiplier=1, name="dw_conv"
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.swish = tf.keras.layers.Activation(
            tf.keras.activations.swish, name="swish_activation")
        self.pw_conv_2 = tf.keras.layers.Conv1D(filters=input_dim, kernel_size=1, strides=1,
                                                padding="same", name="pw_conv_2")
        self.do = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()

    def call(self, inputs, training=False, **kwargs):
        outputs = self.ln(inputs, training=training)
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.glu(outputs)
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(ConvModule, self).get_config()
        conf.update(self.ln.get_config())
        conf.update(self.pw_conv_1.get_config())
        conf.update(self.glu.get_config())
        conf.update(self.dw_conv.get_config())
        conf.update(self.bn.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.pw_conv_2.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf


class ConformerBlock(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 dropout=0.0,
                 fc_factor=0.5,
                 head_size=144,
                 num_heads=4,
                 kernel_size=32,
                 name="conformer_block",
                 **kwargs):
        super(ConformerBlock, self).__init__(name=name, **kwargs)
        self.ffm1 = FFModule(input_dim=input_dim,
                             dropout=dropout, fc_factor=fc_factor,
                             name="ff_module_1")
        self.mhsam = MHSAModule(head_size=head_size, num_heads=num_heads,
                                dropout=dropout)
        self.convm = ConvModule(input_dim=input_dim, kernel_size=kernel_size,
                                dropout=dropout)
        self.ffm2 = FFModule(input_dim=input_dim,
                             dropout=dropout, fc_factor=fc_factor,
                             name="ff_module_2")
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=False, **kwargs):
        outputs = self.ffm1(inputs, training=training)
        outputs = self.mhsam(outputs, training=training)
        outputs = self.convm(outputs, training=training)
        outputs = self.ffm2(outputs, training=training)
        outputs = self.ln(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(ConformerBlock, self).get_config()
        conf.update(self.ffm1.get_config())
        conf.update(self.mhsam.get_config())
        conf.update(self.convm.get_config())
        conf.update(self.ffm2.get_config())
        conf.update(self.ln.get_config())
        return conf


class ConformerEncoder(tf.keras.Model):
    def __init__(self,
                 dmodel=144,
                 reduction_factor=4,
                 num_blocks=16,
                 head_size=36,
                 num_heads=4,
                 kernel_size=32,
                 fc_factor=0.5,
                 dropout=0.0,
                 name="conformer_encoder",
                 **kwargs):
        super(ConformerEncoder, self).__init__(name=name, **kwargs)
        self.conv_subsampling = Conv2DSubsampling(
            odim=dmodel, reduction_factor=reduction_factor)
        self.do = tf.keras.layers.Dropout(dropout)
        self.conformer_blocks = []
        for i in range(num_blocks):
            conformer_block = ConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                name=f"conformer_block_{i}"
            )
            self.conformer_blocks.append(conformer_block)

    def call(self, inputs, training=False, **kwargs):
        # input with shape [B, T, V1, V2]
        outputs = self.conv_subsampling(inputs, training=training)
        outputs = self.do(outputs, training=training)
        for cblock in self.conformer_blocks:
            outputs = cblock(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(ConformerEncoder, self).get_config()
        conf.update(self.conv_subsampling.get_config())
        conf.update(self.do.get_config())
        for cblock in self.conformer_blocks:
            conf.update(cblock.get_config())
        return conf


class Conformer(Transducer):
    def __init__(self,
                 dmodel: int,
                 reduction_factor: int,
                 vocabulary_size: int,
                 num_blocks: int = 16,
                 head_size: int = 36,
                 num_heads: int = 4,
                 kernel_size: int = 32,
                 fc_factor: float = 0.5,
                 dropout: float = 0,
                 embed_dim: int = 512,
                 embed_dropout: int = 0,
                 num_lstms: int = 1,
                 lstm_units: int = 320,
                 joint_dim: int = 1024,
                 name: str = "conformer_transducer",
                 **kwargs):
        super(Conformer, self).__init__(
            encoder=ConformerEncoder(
                dmodel=dmodel,
                reduction_factor=reduction_factor,
                num_blocks=num_blocks,
                head_size=head_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                fc_factor=fc_factor,
                dropout=dropout,
            ),
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            num_lstms=num_lstms,
            lstm_units=lstm_units,
            joint_dim=joint_dim,
            name=name, **kwargs
        )
        self.time_reduction_factor = reduction_factor
