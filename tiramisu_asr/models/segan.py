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
            bias_initializer=tf.keras.initializers.Zeros()
        )

    def call(self, inputs, training=False):
        return self.layer(inputs, training=training)

    def get_config(self):
        config = super(DownConv, self).get_config()
        config.update({"layer": self.layer})
        return config

    def from_config(self, config):
        return self(**config)


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
            bias_initializer=tf.keras.initializers.Zeros()
        )

    def call(self, inputs, training=False):
        return self.layer(inputs, training=training)

    def get_config(self):
        config = super(DeConv, self).get_config()
        config.update({"layer": self.layer})
        return config

    def from_config(self, config):
        return self(**config)


class VirtualBatchNorm:
    def __init__(self, x, name, epsilon=1e-5):
        assert isinstance(epsilon, float)
        self.epsilon = epsilon
        self.name = name
        self.batch_size = tf.cast(tf.shape(x)[0], tf.float32)
        self.gamma = tf.Variable(
            initial_value=tf.random_normal_initializer(1., 0.02)(
                shape=[x.get_shape().as_list()[-1]]),
            name="gamma", trainable=True
        )
        self.beta = tf.Variable(
            initial_value=tf.constant_initializer(0.)(
                shape=[x.get_shape().as_list()[-1]]),
            name="beta", trainable=True
        )
        mean, var = tf.nn.moments(x, axes=[0, 1, 2], keepdims=False)
        self.mean = mean
        self.variance = var

    def __call__(self, x):
        new_coeff = 1. / (self.batch_size + 1.)
        old_coeff = 1. - new_coeff
        new_mean, new_var = tf.nn.moments(x, axes=[0, 1, 2], keepdims=False)
        new_mean = new_coeff * new_mean + old_coeff * self.mean
        new_var = new_coeff * new_var + old_coeff * self.variance
        return tf.nn.batch_normalization(x, mean=new_mean, variance=new_var,
                                         offset=self.beta, scale=self.gamma,
                                         variance_epsilon=self.epsilon)


class GaussianNoise(tf.keras.layers.Layer):
    def __init__(self, name, noise_std, **kwargs):
        super(GaussianNoise, self).__init__(trainable=False, name=name, **kwargs)
        self.noise_std = noise_std

    def call(self, inputs, training=False):
        noise = tf.keras.backend.random_normal(shape=tf.shape(inputs),
                                               mean=0.0, stddev=self.noise_std,
                                               dtype=tf.float32)
        return inputs + noise


class Reshape1to3(tf.keras.layers.Layer):
    def __init__(self, name="reshape_1_to_3", **kwargs):
        super(Reshape1to3, self).__init__(trainable=False, name=name, **kwargs)

    def call(self, inputs, training=False):
        width = inputs.get_shape().as_list()[1]
        return tf.reshape(inputs, [-1, width, 1, 1])

    def get_config(self):
        config = super(Reshape1to3, self).get_config()
        return config

    def from_config(self, config):
        return self(**config)


class Reshape3to1(tf.keras.layers.Layer):
    def __init__(self, name="reshape_3_to_1", **kwargs):
        super(Reshape3to1, self).__init__(trainable=False, name=name, **kwargs)

    def call(self, inputs, training=False):
        width = inputs.get_shape().as_list()[1]
        return tf.reshape(inputs, [-1, width])

    def get_config(self):
        config = super(Reshape3to1, self).get_config()
        return config

    def from_config(self, config):
        return self(**config)


class SeganPrelu(tf.keras.layers.Layer):
    def __init__(self, name="segan_prelu", **kwargs):
        super(SeganPrelu, self).__init__(trainable=True, name=name, **kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name="alpha",
                                     shape=input_shape[-1],
                                     initializer=tf.keras.initializers.zeros,
                                     dtype=tf.float32,
                                     trainable=True)

    def call(self, x, training=False):
        pos = tf.nn.relu(x)
        neg = self.alpha * (x - tf.abs(x)) * .5
        return pos + neg

    def get_config(self):
        config = super(SeganPrelu, self).get_config()
        return config

    def from_config(self, config):
        return self(**config)


class Z(tf.keras.layers.Layer):
    def __init__(self, mean=0., stddev=1., name="segan_z", **kwargs):
        self.mean = mean,
        self.stddev = stddev
        super(Z, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=False):
        z = tf.random.normal(shape=tf.shape(inputs),
                             name="z", mean=self.mean, stddev=self.stddev)
        return tf.keras.layers.Concatenate(axis=3)([z, inputs])

    def get_config(self):
        config = super(Z, self).get_config()
        config.update({
            "mean":   self.mean,
            "stddev": self.stddev
        })
        return config

    def from_config(self, config):
        return self(**config)


def create_generator(g_enc_depths, window_size, kwidth=31, ratio=2):
    g_dec_depths = g_enc_depths.copy()
    g_dec_depths.reverse()
    g_dec_depths = g_dec_depths[1:] + [1]
    skips = []

    # input_shape = [batch_size, 16384]
    signal = tf.keras.Input(shape=(window_size,),
                            name="noisy_input", dtype=tf.float32)
    c = Reshape1to3("segan_g_reshape_input")(signal)
    # Encoder
    for layer_idx, layer_depth in enumerate(g_enc_depths):
        c = DownConv(depth=layer_depth,
                     kwidth=kwidth,
                     pool=ratio,
                     name=f"segan_g_downconv_{layer_idx}")(c)
        if layer_idx < len(g_enc_depths) - 1:
            skips.append(c)
        c = SeganPrelu(name=f"segan_g_downconv_prelu_{layer_idx}")(c)
    # Z
    output = Z()(c)
    # Decoder
    for layer_idx, layer_depth in enumerate(g_dec_depths):
        output = DeConv(depth=layer_depth,
                        kwidth=kwidth,
                        dilation=ratio,
                        name=f"segan_g_deconv_{layer_idx}")(output)
        output = SeganPrelu(name=f"segan_g_deconv_prelu_{layer_idx}")(output)
        if layer_idx < len(g_dec_depths) - 1:
            _skip = skips[-(layer_idx + 1)]
            output = tf.keras.layers.Concatenate(axis=3, name=f"concat_skip_{layer_idx}")([output, _skip])

    reshape_output = Reshape3to1("segan_g_reshape_output")(output)
    # output_shape = [batch_size, 16384]

    return tf.keras.Model(inputs=signal, outputs=reshape_output, name="segan_gen")


def create_generator_v2(g_enc_depths, window_size, kwidth=31, ratio=2):
    g_dec_depths = g_enc_depths.copy()
    g_dec_depths.reverse()
    g_dec_depths = g_dec_depths[1:] + [1]
    skips = []

    # input_shape = [batch_size, 16384]
    signal = tf.keras.Input(shape=(window_size,),
                            name="noisy_input", dtype=tf.float32)
    c = Reshape1to3("segan_g_reshape_input")(signal)
    # Encoder
    for layer_idx, layer_depth in enumerate(g_enc_depths):
        c = DownConv(depth=layer_depth,
                     kwidth=kwidth,
                     pool=ratio,
                     name=f"segan_g_downconv_{layer_idx}")(c)
        if layer_idx < len(g_enc_depths) - 1:
            skips.append(c)
        c = SeganPrelu(name=f"segan_g_downconv_prelu_{layer_idx}")(c)
    # Z
    z = tf.keras.Input(shape=c.get_shape().as_list()[1:], dtype=tf.float32, name="z_input")
    output = tf.keras.layers.Concatenate(axis=3)([z, c])
    # Decoder
    for layer_idx, layer_depth in enumerate(g_dec_depths):
        output = DeConv(depth=layer_depth,
                        kwidth=kwidth,
                        dilation=ratio,
                        name=f"segan_g_deconv_{layer_idx}")(output)
        output = SeganPrelu(name=f"segan_g_deconv_prelu_{layer_idx}")(output)
        if layer_idx < len(g_dec_depths) - 1:
            _skip = skips[-(layer_idx + 1)]
            output = tf.keras.layers.Concatenate(axis=3, name=f"concat_skip_{layer_idx}")([output, _skip])

    reshape_output = Reshape3to1("segan_g_reshape_output")(output)
    # output_shape = [batch_size, 16384]

    return tf.keras.Model(inputs=[signal, z], outputs=reshape_output, name="segan_gen")


def make_z_as_input(generator: tf.keras.Model, model_config: dict, speech_config: dict):
    generator_v2 = create_generator_v2(g_enc_depths=model_config["g_enc_depths"],
                                       window_size=speech_config["window_size"],
                                       kwidth=model_config["kwidth"], ratio=model_config["ratio"])
    generator_v2.set_weights(generator.get_weights())
    return generator_v2


def create_discriminator(d_num_fmaps, window_size, kwidth=31, ratio=2, noise_std=0.):
    clean_signal = tf.keras.Input(shape=(window_size,),
                                  name="disc_clean_input", dtype=tf.float32)
    noisy_signal = tf.keras.Input(shape=(window_size,),
                                  name="disc_noisy_input", dtype=tf.float32)

    clean_wav = Reshape1to3("segan_d_reshape_1_to_3_clean")(clean_signal)
    noisy_wav = Reshape1to3("segan_d_reshape_1_to_3_noisy")(noisy_signal)
    hi = tf.keras.layers.Concatenate(name="segan_d_concat_clean_noisy",
                                     axis=3)([clean_wav, noisy_wav])
    # after concatenation shape = [batch_size, 16384, 1, 2]

    hi = GaussianNoise(noise_std=noise_std, name="segan_d_gaussian_noise")(hi)

    for block_idx, nfmaps in enumerate(d_num_fmaps):
        hi = DownConv(depth=nfmaps, kwidth=kwidth, pool=ratio,
                      name=f"segan_d_downconv_{block_idx}")(hi)
        hi = VirtualBatchNorm(hi, name=f"segan_d_vbn_{block_idx}")(hi)
        hi = tf.keras.layers.LeakyReLU(alpha=0.3, name=f"segan_d_leakyrelu_{block_idx}")(hi)

    hi = tf.squeeze(hi, axis=2)
    hi = tf.keras.layers.Conv1D(filters=1, kernel_size=1,
                                strides=1, padding="same",
                                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                                name="segan_d_conv1d")(hi)
    hi = tf.squeeze(hi, axis=-1)
    hi = tf.keras.layers.Dense(1, name="segan_d_fully_connected")(hi)
    # output_shape = [1]
    return tf.keras.Model(inputs={
        "clean": clean_signal,
        "noisy": noisy_signal,
    }, outputs=hi, name="segan_disc")
