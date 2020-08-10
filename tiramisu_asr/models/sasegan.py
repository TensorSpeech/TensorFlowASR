# Copyright 2020 Huy Le Nguyen (@usimarit) and Huy Phan (@pquochuy)
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

from .segan import VirtualBatchNorm, GaussianNoise, Reshape1to3, Reshape3to1, SeganPrelu
from ..utils.utils import shape_list


def l2_normalize(v, eps=1e-12):
    """l2 normize the input vector."""
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


class SpectralNorm(tf.keras.constraints.Constraint):
    """Performs Spectral Normalization on a weight tensor.
    Specifically it divides the weight tensor by its largest singular value. This
    is intended to stabilize GAN training, by making the discriminator satisfy a
    local 1-Lipschitz constraint.
    Based on [Spectral Normalization for Generative Adversarial Networks][sn-gan]
    [sn-gan] https://openreview.net/pdf?id=B1QRgziT-
    Args:
      num_iters: Number of SN iterations.
    Returns:
      w_bar: The normalized weight tensor
    """

    def __init__(self, num_iters=1):
        self.num_iters = num_iters

    def __call__(self, weights):
        w_shape = shape_list(weights)
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
        u_ = tf.random.truncated_normal([1, w_shape[-1]])
        for _ in range(self.num_iters):
            v_ = l2_normalize(tf.matmul(u_, w_mat, transpose_b=True))
            u_ = l2_normalize(tf.matmul(v_, w_mat))
        sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
        w_mat /= sigma
        w_bar = tf.reshape(w_mat, w_shape)
        return w_bar

    def get_config(self):
        return {'num_iters': self.num_iters}


class SnPointWiseConv(tf.keras.layers.Layer):
    def __init__(self,
                 depth,
                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
                 name="sn_pointwise_conv",
                 **kwargs):
        super(SnPointWiseConv, self).__init__(name=name, **kwargs)
        self.layer = tf.keras.layers.Conv2D(
            filters=depth,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer=kernel_initializer,
            use_bias=False,
            kernel_constraint=SpectralNorm()
        )

    def call(self, inputs, training=False):
        return self.layer(inputs, training=training)

    def get_config(self):
        config = super(SnPointWiseConv, self).get_config()
        config.update(self.layer.get_config())
        return config


class SnNonLocalBlockSim(tf.keras.layers.Layer):
    def __init__(self,
                 num_channels,
                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
                 name="sn_non_local_block_sim",
                 **kwargs):
        super(SnNonLocalBlockSim, self).__init__(name=name, **kwargs)
        self.sn_pointwise_conv_theta = SnPointWiseConv(
            (num_channels // 8), kernel_initializer=kernel_initializer,
            name=f"{name}_pw_conv_theta"
        )
        self.sn_pointwise_conv_phi = SnPointWiseConv(
            (num_channels // 8), kernel_initializer=kernel_initializer,
            name=f"{name}_pw_conv_phi"
        )
        self.sn_pointwise_conv_g = SnPointWiseConv(
            (num_channels // 2), kernel_initializer=kernel_initializer,
            name=f"{name}_pw_conv_g"
        )
        self.sn_pointwise_conv_attn = SnPointWiseConv(
            num_channels, kernel_initializer=kernel_initializer,
            name=f"{name}_pw_conv_attn"
        )
        self.max_pool_phi = tf.keras.layers.MaxPool2D(
            pool_size=(4, 1), strides=[4, 1],
            name=f"{name}_max_pool_phi"
        )
        self.max_pool_g = tf.keras.layers.MaxPool2D(
            pool_size=(4, 1), strides=[4, 1],
            name=f"{name}_max_pool_g"
        )
        self.sigma = self.add_weight(name="sigma_ratio", shape=[],
                                     initializer=tf.constant_initializer(0.0))

    def call(self, inputs, training=False, **kwargs):
        batch_size, h, w, num_channels = shape_list(inputs)
        location_num = h * w
        downsampled_num = location_num // 4

        # theta path
        theta = self.sn_pointwise_conv_theta(inputs, training=training)
        theta = tf.reshape(theta, [batch_size, location_num, num_channels // 8])

        # phi path
        phi = self.sn_pointwise_conv_phi(inputs, training=training)
        phi = self.max_pool_phi(phi, training=training)
        phi = tf.reshape(phi, [batch_size, downsampled_num, num_channels // 8])

        # attn
        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)

        # g path
        g = self.sn_pointwise_conv_g(inputs, training=training)
        g = self.max_pool_g(g, training=training)
        g = tf.reshape(g, [batch_size, downsampled_num, num_channels // 2])

        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels // 2])
        attn_g = self.sn_pointwise_conv_attn(attn_g, training=training)

        return inputs + self.sigma * attn_g

    def get_config(self):
        conf = super(SnNonLocalBlockSim, self).get_config()
        conf.update(self.sn_pointwise_conv_theta.get_config())
        conf.update(self.sn_pointwise_conv_phi.get_config())
        conf.update(self.sn_pointwise_conv_g.get_config())
        conf.update(self.sn_pointwise_conv_attn.get_config())
        conf.update(self.max_pool_phi.get_config())
        conf.update(self.max_pool_g.get_config())
        return conf


class SnDownConv(tf.keras.layers.Layer):
    def __init__(self,
                 depth,
                 kwidth=5,
                 pool=2,
                 name="sn_downconv",
                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                 bias_initializer=tf.keras.initializers.Zeros(),
                 use_bias=True,
                 **kwargs):
        super(SnDownConv, self).__init__(name=name, **kwargs)
        self.layer = tf.keras.layers.Conv2D(
            filters=depth,
            kernel_size=(kwidth, 1),
            strides=(pool, 1),
            padding="same",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_constraint=SpectralNorm(),
            name=f"{name}_2d"
        )

    def call(self, inputs, training=False):
        return self.layer(inputs, training=training)

    def get_config(self):
        config = super(SnDownConv, self).get_config()
        config.update(self.layer.get_config())
        return config


class SnDeConv(tf.keras.layers.Layer):
    def __init__(self,
                 depth,
                 kwidth=5,
                 dilation=2,
                 name="sn_deconv",
                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                 bias_initializer=tf.keras.initializers.Zeros(),
                 use_bias=True,
                 **kwargs):
        super(SnDeConv, self).__init__(name=name, **kwargs)
        self.layer = tf.keras.layers.Conv2DTranspose(
            filters=depth,
            kernel_size=(kwidth, 1),
            strides=(dilation, 1),
            padding="same",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_constraint=SpectralNorm(),
            name=f"{name}_2d"
        )

    def call(self, inputs, training=False):
        return self.layer(inputs, training=training)

    def get_config(self):
        config = super(SnDeConv, self).get_config()
        config.update(self.layer.get_config())
        return config


class GeneratorEncoder(tf.keras.Model):
    def __init__(self,
                 g_enc_depths,
                 kwidth,
                 pool,
                 att_layer_idx,
                 name="sasegan_g_encoder",
                 **kwargs):
        super(GeneratorEncoder, self).__init__(name=name, **kwargs)
        self.blocks = []
        self.att_layer_idx = att_layer_idx
        self.attn = SnNonLocalBlockSim(
            g_enc_depths[att_layer_idx],
            name=f"{self.name}_sn_non_local_block_sim_enc"
        )
        for i, depth in enumerate(g_enc_depths):
            dc = SnDownConv(depth=depth,
                            kwidth=kwidth,
                            pool=pool,
                            name=f"{name}_sn_downconv_{i}")
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
            if i == self.att_layer_idx:
                outputs = self.attn(outputs, training=training)
        return outputs, skips

    def get_config(self):
        conf = self.attn.get_config()
        for block in self.blocks:
            conf.update(block["downconv"].get_config())
            conf.update(block["prelu"].get_config())
        return conf


class GeneratorDecoder(tf.keras.Model):
    def __init__(self,
                 g_dec_depths,
                 kwidth,
                 dilation,
                 att_layer_idx,
                 name="sasegan_g_decoder",
                 **kwargs):
        super(GeneratorDecoder, self).__init__(name=name, **kwargs)
        self.blocks = []
        self.att_layer_idx = att_layer_idx
        self.attn = SnNonLocalBlockSim(
            g_dec_depths[att_layer_idx] * 2,  # the att placed after concat
            name=f"{self.name}_sn_non_local_block_sim_dec"
        )
        for i, depth in enumerate(g_dec_depths):
            dc = SnDeConv(depth=depth,
                          kwidth=kwidth,
                          dilation=dilation,
                          name=f"{name}_sn_deconv_{i}")
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
            if i == self.att_layer_idx:
                outputs = self.attn(outputs, training=training)
        return outputs

    def get_config(self):
        conf = self.attn.get_config()
        for block in self.blocks:
            conf.update(block["deconv"].get_config())
            conf.update(block["prelu"].get_config())
        return conf


class Generator(tf.keras.Model):
    def __init__(self,
                 g_enc_depths,
                 window_size,
                 att_layer_idx,
                 kwidth=31,
                 ratio=2,
                 name="sasegan_generator",
                 **kwargs):
        super(Generator, self).__init__(name=name, **kwargs)
        self.g_enc_depths = g_enc_depths
        g_dec_depths = g_enc_depths.copy()
        g_dec_depths.reverse()
        g_dec_depths = g_dec_depths[1:] + [1]
        self.window_size = window_size
        self.ratio = ratio
        self.inp = Reshape1to3(f"{self.name}_reshape_input")
        self.encoder = GeneratorEncoder(g_enc_depths, kwidth=kwidth, pool=ratio,
                                        att_layer_idx=att_layer_idx, name=f"{name}_encoder")
        self.z = tf.keras.layers.Concatenate(axis=3, name=f"{self.name}_z")
        self.decoder = GeneratorDecoder(g_dec_depths, kwidth=kwidth, dilation=ratio,
                                        att_layer_idx=(len(g_enc_depths) - att_layer_idx - 1),
                                        name=f"{name}_decoder")
        self.outp = Reshape3to1(f"{self.name}_reshape_output")

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
        conf = self.inp.get_config()
        conf.update(self.encoder.get_config())
        conf.update(self.z.get_config())
        conf.update(self.decoder.get_config())
        conf.update(self.out.get_config())
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
        self.dc = SnDownConv(depth=depth, kwidth=kwidth, pool=pool, name=f"{name}_sndownconv")
        self.vbn = VirtualBatchNorm(name=f"{name}_vbn")
        if leakyrelu:
            self.relu = tf.keras.layers.LeakyReLU(alpha=0.3, name=f"{name}_leakyrelu")
        else:
            self.relu = tf.keras.layers.ReLU(name=f"{name}_relu")

    def call(self, inputs, training=False):
        outputs = self.dc(inputs, training=training)
        outputs = self.vbn(outputs, training=training)
        return self.relu(outputs, training=training)


class Discriminator(tf.keras.Model):
    def __init__(self,
                 d_num_fmaps,
                 window_size,
                 att_layer_idx,
                 kwidth=31,
                 ratio=2,
                 leakyrelu=True,
                 name="sasegan_disc",
                 **kwargs):
        super(Discriminator, self).__init__(name=name, **kwargs)
        self.window_size = window_size
        self.att_layer_idx = att_layer_idx
        self.clean_wav = Reshape1to3(f"{name}_reshape_1_to_3_clean")
        self.noisy_wav = Reshape1to3(f"{name}_reshape_1_to_3_noisy")
        self.concat = tf.keras.layers.Concatenate(name=f"{name}_concat", axis=3)
        self.gauss = GaussianNoise(name=f"{name}_gaussian_noise")
        self.blocks = []
        for block_idx, nfmaps in enumerate(d_num_fmaps):
            self.blocks.append(
                DiscriminatorBlock(depth=nfmaps, kwidth=kwidth, pool=ratio,
                                   leakyrelu=leakyrelu, name=f"{name}_block_{block_idx}"))
        self.attn = SnNonLocalBlockSim(d_num_fmaps[self.att_layer_idx],
                                       name=f"{name}_sn_non_local_block_sim")
        self.conv = tf.keras.layers.Conv2D(
            filters=1, kernel_size=1, strides=1, padding="same",
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
        for i, block in enumerate(self.blocks):
            outputs = block(outputs, training=training)
            if i == self.att_layer_idx:
                outputs = self.attn(outputs, training=training)
        outputs = self.conv(outputs, training=training)
        outputs = self.reshape(outputs)
        return self.dense(outputs, training=training)
