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


class Generator(tf.keras.Model):
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
        self.g_dec_depths = g_dec_depths
        self.window_size = window_size
        self.ratio = ratio
        self.inp = Reshape1to3("segan_g_reshape_input")
        self.enc = []
        for layer_idx, layer_depth in enumerate(g_enc_depths):
            dc = DownConv(depth=layer_depth,
                          kwidth=kwidth,
                          pool=ratio,
                          name=f"segan_g_downconv_{layer_idx}")
            prelu = SeganPrelu(name=f"segan_g_downconv_prelu_{layer_idx}")
            self.enc.append({
                "downconv": dc,
                "prelu": prelu
            })
        self.z_concat = tf.keras.layers.Concatenate(axis=3)
        self.dec = []
        for layer_idx, layer_depth in enumerate(g_dec_depths):
            dc = DeConv(depth=layer_depth,
                        kwidth=kwidth,
                        dilation=ratio,
                        name=f"segan_g_deconv_{layer_idx}")
            prelu = SeganPrelu(name=f"segan_g_deconv_prelu_{layer_idx}")
            if layer_idx < len(g_dec_depths) - 1:
                concat = tf.keras.layers.Concatenate(
                    axis=3, name=f"concat_skip_{layer_idx}")
            else:
                concat = None
            self.dec.append({
                "deconv": dc,
                "prelu": prelu,
                "concat": concat
            })
        self.outp = Reshape3to1("segan_g_reshape_output")

    def get_z(self, batch_size, mean=0., stddev=1.):
        return tf.random.normal(
            [batch_size, self.window_size // (self.ratio ** len(self.enc)),
             1, self.g_enc_depths[-1]], mean=mean, stddev=stddev)

    def _build(self):
        audio = tf.random.normal([1, self.window_size], dtype=tf.float32)
        z = self.get_z(1)
        self([audio, z], training=False)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, **kwargs):
        noisy, z = inputs
        c = self.inp(noisy)
        skips = []
        for i, enc in enumerate(self.enc):
            c = enc["downconv"](c, training=training)
            if i < len(self.enc) - 1:
                skips.append(c)
            c = enc["prelu"](c)
        outputs = self.z_concat([z, c])
        for i, dec in enumerate(self.dec):
            outputs = dec["deconv"](outputs, training=training)
            outputs = dec["prelu"](outputs, training=training)
            if i < len(self.dec) - 1:
                _skip = skips[-(i + 1)]
                outputs = dec["concat"]([outputs, _skip])
            else:
                outputs = tf.nn.tanh(outputs)
        outputs = self.outp(outputs)
        return outputs

    def get_config(self):
        conf = self.noisy.get_config()
        for enc in self.enc:
            conf.update(enc[0].get_config())
            conf.update(enc[1].get_config())
        conf.update(self.z_concat.get_config())
        for dec in self.dec:
            conf.update(dec[0].get_config())
            conf.update(dec[1].get_config())
            if dec[2] is not None:
                conf.update(dec[2].get_config())
        conf.update(self.out.get_config())
        return conf


class Discriminator(tf.keras.Model):
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
        self.clean_wav = Reshape1to3("segan_d_reshape_1_to_3_clean")
        self.noisy_wav = Reshape1to3("segan_d_reshape_1_to_3_noisy")
        self.concat = tf.keras.layers.Concatenate(name="segan_d_concat_clean_noisy", axis=3)
        self.gauss = GaussianNoise(name="segan_d_gaussian_noise")
        self.blocks = []
        for block_idx, nfmaps in enumerate(d_num_fmaps):
            dc = DownConv(depth=nfmaps, kwidth=kwidth, pool=ratio,
                          name=f"segan_d_downconv_{block_idx}")
            vbn = VirtualBatchNorm(name=f"vbn_{block_idx}")
            if leakyrelu:
                relu = tf.keras.layers.LeakyReLU(
                    alpha=0.3, name=f"segan_d_leakyrelu_{block_idx}")
            else:
                relu = tf.keras.layers.ReLU(name=f"segan_d_relu_{block_idx}")
            self.blocks.append({
                "downconv": dc,
                "vbn": vbn,
                "relu": relu
            })
        self.conv = tf.keras.layers.Conv1D(
            filters=1, kernel_size=1, strides=1, padding="same",
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="segan_d_conv1d"
        )
        self.dense = tf.keras.layers.Dense(1, name="segan_d_fully_connected")

    def _build(self):
        clean = tf.random.normal([1, self.window_size], dtype=tf.float32)
        noisy = tf.random.normal([1, self.window_size], dtype=tf.float32)
        self([clean, noisy], training=False)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, noise_std=0., **kwargs):
        clean, noisy = inputs
        clean_out = self.clean_wav(clean)
        noisy_out = self.noisy_wav(noisy)
        outputs = self.concat([clean_out, noisy_out])
        outputs = self.gauss(outputs, noise_std=noise_std)
        for i in range(len(self.blocks)):
            outputs = self.blocks[i]["downconv"](outputs, training=training)
            outputs = self.blocks[i]["vbn"](outputs, training=training)
            outputs = self.blocks[i]["relu"](outputs, training=training)
        outputs = tf.squeeze(outputs, axis=2)
        outputs = self.conv(outputs, training=training)
        outputs = tf.squeeze(outputs, axis=-1)
        return self.dense(outputs, training=training)
