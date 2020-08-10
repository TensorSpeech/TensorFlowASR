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

from . import Model
from ..utils.utils import shape_list


class DownConv(tf.keras.layers.Layer):
    def __init__(self, depth, kwidth=5, pool=2, name="downconv", **kwargs):
        super(DownConv, self).__init__(name=name, **kwargs)
        self.layer = tf.keras.layers.Conv2D(
            filters=depth,
            kernel_size=(kwidth, 1),
            strides=(pool, 1),
            padding="same",
            use_bias=True,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            bias_initializer=tf.keras.initializers.Zeros(),
            name=f"{name}_2d"
        )

    def call(self, inputs, training=False):
        return self.layer(inputs, training=training)

    def get_config(self):
        config = super(DownConv, self).get_config()
        config.update({"layer": self.layer})
        return config


class DeConv(tf.keras.layers.Layer):
    def __init__(self, depth, kwidth=5, dilation=2, name="deconv", **kwargs):
        super(DeConv, self).__init__(name=name, **kwargs)
        self.layer = tf.keras.layers.Conv2DTranspose(
            filters=depth,
            kernel_size=(kwidth, 1),
            strides=(dilation, 1),
            padding="same",
            use_bias=True,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            bias_initializer=tf.keras.initializers.Zeros(),
            name=f"{name}_2d"
        )

    def call(self, inputs, training=False):
        return self.layer(inputs, training=training)

    def get_config(self):
        config = super(DeConv, self).get_config()
        config.update({"layer": self.layer})
        return config


class VirtualBatchNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon: float = 1e-5, name="vbn", **kwargs):
        self.epsilon = epsilon
        super(VirtualBatchNorm, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.gamma = tf.Variable(
            initial_value=tf.random_normal_initializer(1., 0.02)(
                shape=[input_shape[-1]]),
            name="gamma", trainable=True
        )
        self.beta = tf.Variable(
            initial_value=tf.constant_initializer(0.)(
                shape=[input_shape[-1]]),
            name="beta", trainable=True
        )
        self.first = True
        self.mean = tf.Variable(
            initial_value=tf.zeros(shape=[input_shape[-1]]),
            name="mean", trainable=False
        )
        self.variance = tf.Variable(
            initial_value=tf.zeros(shape=[input_shape[-1]]),
            name="variance", trainable=False
        )

    def call(self, x, training=False, **kwargs):
        batch_size = shape_list(x)[0]
        new_coeff = 1. / (tf.cast(batch_size, x.dtype) + 1.)
        old_coeff = 1. - new_coeff
        new_mean, new_var = tf.nn.moments(x, axes=[0, 1, 2], keepdims=False)

        if self.first and training:
            self.first = False
            self.mean.assign(new_mean)
            self.variance.assign(new_var)

        new_mean = new_coeff * new_mean + old_coeff * self.mean
        new_var = new_coeff * new_var + old_coeff * self.variance
        normed = tf.nn.batch_normalization(x, mean=new_mean, variance=new_var,
                                           offset=self.beta, scale=self.gamma,
                                           variance_epsilon=self.epsilon)
        return tf.reshape(normed, shape_list(x))


class GaussianNoise(tf.keras.layers.Layer):
    def __init__(self, name="gaussian_noise", **kwargs):
        super(GaussianNoise, self).__init__(trainable=False, name=name, **kwargs)

    def call(self, inputs, noise_std=0.):
        noise = tf.keras.backend.random_normal(shape=tf.shape(inputs),
                                               mean=0.0, stddev=noise_std,
                                               dtype=tf.float32)
        return inputs + noise


class Reshape1to3(tf.keras.layers.Layer):
    def __init__(self, name="reshape_1_to_3", **kwargs):
        super(Reshape1to3, self).__init__(trainable=False, name=name, **kwargs)

    def call(self, inputs):
        width = inputs.get_shape().as_list()[1]
        return tf.reshape(inputs, [-1, width, 1, 1])

    def get_config(self):
        config = super(Reshape1to3, self).get_config()
        return config


class Reshape3to1(tf.keras.layers.Layer):
    def __init__(self, name="reshape_3_to_1", **kwargs):
        super(Reshape3to1, self).__init__(trainable=False, name=name, **kwargs)

    def call(self, inputs):
        width = inputs.get_shape().as_list()[1]
        return tf.reshape(inputs, [-1, width])

    def get_config(self):
        config = super(Reshape3to1, self).get_config()
        return config


class SeganPrelu(tf.keras.layers.Layer):
    def __init__(self, name="segan_prelu", **kwargs):
        super(SeganPrelu, self).__init__(trainable=True, name=name, **kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name="alpha",
                                     shape=input_shape[-1],
                                     initializer=tf.keras.initializers.zeros,
                                     dtype=tf.float32,
                                     trainable=True)

    def call(self, x):
        pos = tf.nn.relu(x)
        neg = self.alpha * (x - tf.abs(x)) * .5
        return pos + neg

    def get_config(self):
        config = super(SeganPrelu, self).get_config()
        return config


class GeneratorEncoder(tf.keras.Model):
    def __init__(self,
                 g_enc_depths,
                 kwidth,
                 pool,
                 name="segan_g_encoder",
                 **kwargs):
        super(GeneratorEncoder, self).__init__(name=name, **kwargs)
        self.blocks = []
        for i, depth in enumerate(g_enc_depths):
            dc = DownConv(depth=depth,
                          kwidth=kwidth,
                          pool=pool,
                          name=f"{name}_downconv_{i}")
            prelu = SeganPrelu(name=f"{name}_prelu_{i}")
            self.blocks.append({"downconv": dc, "prelu": prelu})

    def call(self, inputs, training=False):
        skips = []
        outputs = inputs
        for i, block in enumerate(self.blocks):
            outputs = block["downconv"](outputs, training=training)
            if i < len(self.blocks) - 1:
                skips.append(outputs)
            outputs = block["prelu"](outputs)
        return outputs, skips

    def get_config(self):
        conf = {}
        for block in self.blocks:
            conf.update(block["downconv"].get_config())
            conf.update(block["prelu"].get_config())
        return conf


class GeneratorDecoder(tf.keras.Model):
    def __init__(self,
                 g_dec_depths,
                 kwidth,
                 dilation,
                 name="segan_g_decoder",
                 **kwargs):
        super(GeneratorDecoder, self).__init__(name=name, **kwargs)
        self.blocks = []
        for i, depth in enumerate(g_dec_depths):
            dc = DeConv(depth=depth,
                        kwidth=kwidth,
                        dilation=dilation,
                        name=f"{name}_deconv_{i}")
            prelu = SeganPrelu(name=f"{name}_prelu_{i}")
            if i < len(g_dec_depths) - 1:
                skip = tf.keras.layers.Concatenate(axis=3, name=f"{name}_concat")
            else:
                skip = tf.keras.layers.Activation(tf.nn.tanh, name=f"{name}_tanh")
            self.blocks.append({"deconv": dc, "prelu": prelu, "skip": skip})

    def call(self, inputs, skips, training=False):
        outputs = inputs
        for i, block in enumerate(self.blocks):
            outputs = block["deconv"](outputs, training=training)
            outputs = block["prelu"](outputs)
            if i < len(self.blocks) - 1:
                _skip = skips[-(i + 1)]
                outputs = block["skip"]([outputs, _skip])
            else:
                outputs = block["skip"](outputs)
        return outputs

    def get_config(self):
        conf = {}
        for block in self.blocks:
            conf.update(block["deconv"].get_config())
            conf.update(block["prelu"].get_config())
        return conf


class Generator(Model):
    def __init__(self,
                 g_enc_depths,
                 window_size,
                 kwidth=31,
                 ratio=2,
                 name="segan_gen",
                 **kwargs):
        super(Generator, self).__init__(name=name, **kwargs)
        self.g_enc_depths = g_enc_depths
        g_dec_depths = g_enc_depths.copy()
        g_dec_depths.reverse()
        g_dec_depths = g_dec_depths[1:] + [1]
        self.window_size = window_size
        self.ratio = ratio
        self.inp = Reshape1to3("segan_g_reshape_input")
        self.encoder = GeneratorEncoder(g_enc_depths, kwidth=kwidth, pool=ratio,
                                        name=f"{name}_encoder")
        self.z = tf.keras.layers.Concatenate(axis=3, name=f"{name}_z")
        self.decoder = GeneratorDecoder(g_dec_depths, kwidth=kwidth, dilation=ratio,
                                        name=f"{name}_decoder")
        self.outp = Reshape3to1("segan_g_reshape_output")

    def _get_z_shape(self, batch_size):
        return [batch_size,
                self.window_size // (self.ratio ** len(self.g_enc_depths)),
                1,
                self.g_enc_depths[-1]]

    def get_z(self, batch_size, mean=0., stddev=1.):
        return tf.random.normal(self._get_z_shape(batch_size), mean=mean, stddev=stddev)

    def _build(self):
        input_shape = [self.window_size]
        z_shape = self._get_z_shape(None)
        noisy, z = tf.keras.Input(input_shape), tf.keras.Input(z_shape[1:])
        self([noisy, z], training=False)

    def summary(self, line_length=100):
        self.encoder.summary(line_length)
        self.decoder.summary(line_length)
        super(Generator, self).summary(line_length)

    def call(self, inputs, training=False, **kwargs):
        noisy, z = inputs
        outputs = self.inp(noisy)
        outputs, skips = self.encoder(outputs, training=training)
        outputs = self.z([z, outputs])
        outputs = self.decoder(outputs, skips, training=training)
        outputs = self.outp(outputs)
        return outputs

    def get_config(self):
        conf = self.noisy.get_config()
        conf.update(self.inp.get_config())
        conf.update(self.encoder.get_config())
        conf.update(self.z.get_config())
        conf.update(self.decoder.get_config())
        conf.update(self.outp.get_config())
        return conf


class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self,
                 depth,
                 kwidth,
                 pool,
                 name="segan_d_block",
                 leakyrelu=True,
                 **kwargs):
        super(DiscriminatorBlock, self).__init__(name=name, **kwargs)
        self.dc = DownConv(depth=depth, kwidth=kwidth, pool=pool, name=f"{name}_downconv")
        self.vbn = VirtualBatchNorm(name=f"{name}_vbn")
        if leakyrelu:
            self.relu = tf.keras.layers.LeakyReLU(alpha=0.3, name=f"{name}_leakyrelu")
        else:
            self.relu = tf.keras.layers.ReLU(name=f"{name}_relu")

    def call(self, inputs, training=False):
        outputs = self.dc(inputs, training=training)
        outputs = self.vbn(outputs, training=training)
        return self.relu(outputs, training=training)


class Discriminator(Model):
    def __init__(self,
                 d_num_fmaps,
                 window_size,
                 kwidth=31,
                 ratio=2,
                 leakyrelu=True,
                 name="segan_disc",
                 **kwargs):
        super(Discriminator, self).__init__(name=name, **kwargs)
        self.window_size = window_size
        self.clean_wav = Reshape1to3(f"{name}_reshape_1_to_3_clean")
        self.noisy_wav = Reshape1to3(f"{name}_reshape_1_to_3_noisy")
        self.concat = tf.keras.layers.Concatenate(name=f"{name}_concat", axis=3)
        self.gauss = GaussianNoise(name=f"{name}_gaussian_noise")
        self.blocks = []
        for block_idx, nfmaps in enumerate(d_num_fmaps):
            self.blocks.append(
                DiscriminatorBlock(depth=nfmaps, kwidth=kwidth, pool=ratio,
                                   leakyrelu=leakyrelu, name=f"{name}_block_{block_idx}"))
        self.conv = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(1, 1), strides=(1, 1), padding="same",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name=f"{name}_conv2d_1x1"
        )
        self.reshape = Reshape3to1(f"{name}_reshape_3_to_1")
        self.dense = tf.keras.layers.Dense(1, name=f"{name}_fully_connected")

    def _build(self):
        input_shape = [self.window_size]
        clean, noisy = tf.keras.Input(input_shape), tf.keras.Input(input_shape)
        self([clean, noisy], training=False)

    def call(self, inputs, training=False, noise_std=0., **kwargs):
        clean, noisy = inputs
        clean_out = self.clean_wav(clean)
        noisy_out = self.noisy_wav(noisy)
        outputs = self.concat([clean_out, noisy_out])
        outputs = self.gauss(outputs, noise_std=noise_std)
        for block in self.blocks:
            outputs = block(outputs, training=training)
        outputs = self.conv(outputs, training=training)
        outputs = self.reshape(outputs)
        return self.dense(outputs, training=training)
