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
from .layers.multihead_attention import MultiHeadAttention, RelPositionMultiHeadAttention

L2 = tf.keras.regularizers.l2(1e-6)


class VGG2L(tf.keras.layers.Layer):
    def __init__(self,
                 odim: int,
                 reduction_factor: int = 4,
                 dropout: float = 0.0,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="vgg2l_subsampling",
                 **kwargs):
        super(VGG2L, self).__init__(name=name, **kwargs)
        assert reduction_factor % 2 == 0, "reduction_factor must be divisible by 2"
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=1,
            padding="same", name=f"{name}_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=1,
            padding="same", name=f"{name}_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.maxpool1 = tf.keras.layers.MaxPool2D(
            pool_size=(reduction_factor // 2, 2),
            padding="same", name=f"{name}_maxpool_1"
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, strides=1,
            padding="same", name=f"{name}_conv_3",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, strides=1,
            padding="same", name=f"{name}_conv_4",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.maxpool2 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            padding="same", name=f"{name}_maxpool_2"
        )
        self.linear = tf.keras.layers.Dense(
            odim, name=f"{name}_linear",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")

    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv1(inputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.conv2(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.maxpool1(outputs, training=training)

        outputs = self.conv3(inputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.conv4(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.maxpool2(outputs, training=training)

        outputs = merge_two_last_dims(outputs)
        outputs = self.linear(outputs, training=training)
        return self.do(outputs, training=training)

    def get_config(self):
        conf = super(VGG2L, self).get_config()
        conf.update(self.conv1.get_config())
        conf.update(self.conv2.get_config())
        conf.update(self.maxpool1.get_config())
        conf.update(self.conv3.get_config())
        conf.update(self.conv4.get_config())
        conf.update(self.maxpool2.get_config())
        conf.update(self.linear.get_config())
        conf.update(self.do.get_config())
        return conf


class VGG1L(tf.keras.layers.Layer):
    def __init__(self,
                 odim: int,
                 reduction_factor: int = 4,
                 dropout: float = 0.0,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="conv2d_subsampling",
                 **kwargs):
        super(VGG1L, self).__init__(name=name, **kwargs)
        assert reduction_factor % 2 == 0, "reduction_factor must be divisible by 2"
        self.conv1 = tf.keras.layers.Conv2D(
            filters=odim, kernel_size=3, strides=1,
            padding="same", name=f"{name}_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.maxpool1 = tf.keras.layers.MaxPool2D(
            pool_size=(reduction_factor // 2, 2),
            padding="same", name=f"{name}_maxpool_1"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=odim, kernel_size=3, strides=1,
            padding="same", name=f"{name}_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.linear = tf.keras.layers.Dense(
            odim, name=f"{name}_linear",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.maxpool2 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            padding="same", name=f"{name}_maxpool_2"
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")

    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv1(inputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.maxpool1(outputs, training=training)

        outputs = self.conv2(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.maxpool2(outputs, training=training)

        outputs = merge_two_last_dims(outputs)
        outputs = self.linear(outputs, training=training)
        return self.do(outputs, training=training)

    def get_config(self):
        conf = super(VGG1L, self).get_config()
        conf.update(self.conv1.get_config())
        conf.update(self.maxpool1.get_config())
        conf.update(self.conv2.get_config())
        conf.update(self.maxpool2.get_config())
        conf.update(self.linear.get_config())
        conf.update(self.do.get_config())
        return conf


class FFModule(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 dropout=0.0,
                 fc_factor=0.5,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="ff_module",
                 **kwargs):
        super(FFModule, self).__init__(name=name, **kwargs)
        self.fc_factor = fc_factor
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer
        )
        self.ffn1 = tf.keras.layers.Dense(
            4 * input_dim, name=f"{name}_dense_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.swish = tf.keras.layers.Activation(
            tf.keras.activations.swish, name=f"{name}_swish_activation")
        self.do1 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_1")
        self.ffn2 = tf.keras.layers.Dense(
            input_dim, name=f"{name}_dense_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.do2 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_2")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

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
                 mha_type="relmha",
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="mhsa_module",
                 **kwargs):
        super(MHSAModule, self).__init__(name=name, **kwargs)
        self.pe = PositionalEncoding(name=f"{name}_pe")
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer
        )
        if mha_type == "relmha":
            self.mha = RelPositionMultiHeadAttention(
                name=f"{name}_mhsa",
                head_size=head_size, num_heads=num_heads,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            )
        elif mha_type == "mha":
            self.mha = MultiHeadAttention(
                name=f"{name}_mhsa",
                head_size=head_size, num_heads=num_heads,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            )
        else:
            raise ValueError("mha_type must be either 'mha' or 'relmha'")
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")
        self.mha_type = mha_type

    def call(self, inputs, training=False, **kwargs):
        outputs = self.ln(inputs, training=training)
        pe = self.pe(outputs)
        if self.mha_type == "relmha":
            outputs = self.mha([outputs, outputs, outputs, pe], training=training)
        else:
            outputs = outputs + pe
            outputs = self.mha([outputs, outputs, outputs], training=training)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(MHSAModule, self).get_config()
        conf.update({"mha_type": self.mha_type})
        conf.update(self.pe.get_config())
        conf.update(self.ln.get_config())
        conf.update(self.rpmha.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf


class ConvModule(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 kernel_size=32,
                 dropout=0.0,
                 depth_multiplier=1,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="conv_module",
                 **kwargs):
        super(ConvModule, self).__init__(name=name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization()
        self.pw_conv_1 = tf.keras.layers.Conv1D(
            filters=2 * input_dim, kernel_size=1, strides=1,
            padding="same", name=f"{name}_pw_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.glu = GLU(name=f"{name}_glu")
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(1, kernel_size), strides=1,
            padding="same", name=f"{name}_dw_conv",
            depth_multiplier=depth_multiplier,
            depthwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name=f"{name}_bn",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer
        )
        self.swish = tf.keras.layers.Activation(
            tf.keras.activations.swish, name=f"{name}_swish_activation")
        self.pw_conv_2 = tf.keras.layers.Conv1D(
            filters=input_dim, kernel_size=1, strides=1,
            padding="same", name=f"{name}_pw_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(self, inputs, training=False, **kwargs):
        outputs = self.ln(inputs, training=training)
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.glu(outputs)
        outputs = tf.expand_dims(outputs, axis=1)
        outputs = self.dw_conv(outputs, training=training)
        outputs = tf.squeeze(outputs, axis=1)
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
                 head_size=36,
                 num_heads=4,
                 mha_type="relmha",
                 kernel_size=32,
                 depth_multiplier=1,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="conformer_block",
                 **kwargs):
        super(ConformerBlock, self).__init__(name=name, **kwargs)
        self.ffm1 = FFModule(
            input_dim=input_dim, dropout=dropout,
            fc_factor=fc_factor, name=f"{name}_ff_module_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.mhsam = MHSAModule(
            mha_type=mha_type,
            head_size=head_size, num_heads=num_heads,
            dropout=dropout, name=f"{name}_mhsa_module",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.convm = ConvModule(
            input_dim=input_dim, kernel_size=kernel_size,
            dropout=dropout, name=f"{name}_conv_module",
            depth_multiplier=depth_multiplier,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.ffm2 = FFModule(
            input_dim=input_dim, dropout=dropout,
            fc_factor=fc_factor, name=f"{name}_ff_module_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=kernel_regularizer
        )

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
                 subsampling="vgg2l",
                 dmodel=144,
                 reduction_factor=4,
                 num_blocks=16,
                 mha_type="relmha",
                 head_size=36,
                 num_heads=4,
                 kernel_size=32,
                 depth_multiplier=1,
                 fc_factor=0.5,
                 dropout=0.0,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="conformer_encoder",
                 **kwargs):
        super(ConformerEncoder, self).__init__(name=name, **kwargs)
        if subsampling == "vgg1l":
            self.conv_subsampling = VGG1L(
                odim=dmodel, reduction_factor=reduction_factor,
                dropout=dropout, name=f"{name}_subsampling",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            )
        elif subsampling == "vgg2l":
            self.conv_subsampling = VGG2L(
                odim=dmodel, reduction_factor=reduction_factor,
                dropout=dropout, name=f"{name}_subsampling",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            )
        else:
            raise ValueError("subsampling must be eight 'vgg1l' or 'vgg2l'")

        self.conformer_blocks = []
        for i in range(num_blocks):
            conformer_block = ConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                mha_type=mha_type,
                kernel_size=kernel_size,
                depth_multiplier=depth_multiplier,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{name}_block_{i}"
            )
            self.conformer_blocks.append(conformer_block)

    def call(self, inputs, training=False, **kwargs):
        # input with shape [B, T, V1, V2]
        outputs = self.conv_subsampling(inputs, training=training)
        for cblock in self.conformer_blocks:
            outputs = cblock(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(ConformerEncoder, self).get_config()
        conf.update(self.conv_subsampling.get_config())
        for cblock in self.conformer_blocks:
            conf.update(cblock.get_config())
        return conf


class Conformer(Transducer):
    def __init__(self,
                 subsampling: str = "vgg2l",
                 dmodel: int = 144,
                 reduction_factor: int = 4,
                 vocabulary_size: int = 29,
                 num_blocks: int = 16,
                 head_size: int = 36,
                 num_heads: int = 4,
                 mha_type: str = "relmha",
                 kernel_size: int = 32,
                 fc_factor: float = 0.5,
                 dropout: float = 0,
                 embed_dim: int = 512,
                 embed_dropout: int = 0,
                 num_lstms: int = 1,
                 lstm_units: int = 320,
                 joint_dim: int = 1024,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name: str = "conformer_transducer",
                 **kwargs):
        super(Conformer, self).__init__(
            encoder=ConformerEncoder(
                subsampling=subsampling,
                dmodel=dmodel,
                reduction_factor=reduction_factor,
                num_blocks=num_blocks,
                head_size=head_size,
                num_heads=num_heads,
                mha_type=mha_type,
                kernel_size=kernel_size,
                fc_factor=fc_factor,
                dropout=dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            ),
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            num_lstms=num_lstms,
            lstm_units=lstm_units,
            joint_dim=joint_dim,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=name, **kwargs
        )
        self.time_reduction_factor = reduction_factor
