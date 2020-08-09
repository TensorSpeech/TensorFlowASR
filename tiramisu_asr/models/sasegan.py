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
        super(SnNonLocalBlockSim, self).__init__(name=name, **kwargs)

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
        self.g_dec_depths = g_dec_depths
        self.window_size = window_size
        self.att_layer_idx = att_layer_idx
        self.inp = Reshape1to3(f"{self.name}_reshape_input")
        self._create_encoder(kwidth, ratio)
        self.z_concat = tf.keras.layers.Concatenate(axis=3, name=f"{self.name}_z")
        self._create_decoder(kwidth, ratio)
        self.outp = Reshape3to1(f"{self.name}_reshape_output")

    def _create_encoder(self, kwidth, ratio):
        self.enc = []
        for layer_idx, layer_depth in enumerate(self.g_enc_depths):
            dc = SnDownConv(depth=layer_depth,
                            kwidth=kwidth,
                            pool=ratio,
                            name=f"segan_g_downconv_{layer_idx}")
            prelu = SeganPrelu(name=f"segan_g_downconv_prelu_{layer_idx}")
            self.enc.append({
                "downconv": dc,
                "prelu": prelu
            })
        self.enc_attn = SnNonLocalBlockSim(
            self.g_enc_depths[self.att_layer_idx],
            name=f"{self.name}_sn_non_local_block_sim_enc"
        )

    def _create_decoder(self, kwidth, ratio):
        self.dec = []
        for layer_idx, layer_depth in enumerate(self.g_dec_depths):
            dc = SnDeConv(depth=layer_depth,
                          kwidth=kwidth,
                          dilation=ratio,
                          name=f"segan_g_deconv_{layer_idx}")
            prelu = SeganPrelu(name=f"segan_g_deconv_prelu_{layer_idx}")
            if layer_idx < len(self.g_dec_depths) - 1:
                concat = tf.keras.layers.Concatenate(
                    axis=3, name=f"concat_skip_{layer_idx}")
            else:
                concat = None
            self.dec.append({
                "deconv": dc,
                "prelu": prelu,
                "concat": concat
            })
        self.dec_attn = SnNonLocalBlockSim(
            self.g_dec_depths[len(self.g_enc_depths) - self.att_layer_idx - 1],
            name=f"{self.name}_sn_non_local_block_sim_dec"
        )

    def get_z(self, batch_size, mean=0., stddev=1.):
        return tf.random.normal(
            [batch_size, self.window_size // (self.ratio ** len(self.enc)),
             1, self.g_enc_depths[-1]], mean=mean, stddev=stddev)

    def _build(self):
        audio = tf.random.normal([1, self.window_size], dtype=tf.float32)
        z = self.get_z(1)
        self([audio, z], training=False)

    def call(self, inputs, training=False, **kwargs):
        noisy, z = inputs
        c = self.inp(noisy)
        skips = []
        for i, enc in enumerate(self.enc):
            c = enc["downconv"](c, training=training)
            if i < len(self.enc) - 1:
                skips.append(c)
            if i == self.att_layer_idx:
                c = self.enc_attn(c, training=training)
            c = enc["prelu"](c)
        outputs = self.z_concat([z, c])
        for i, dec in enumerate(self.dec):
            outputs = dec["deconv"](outputs, training=training)
            outputs = dec["prelu"](outputs, training=training)
            if i < len(self.dec) - 1:
                _skip = skips[-(i + 1)]
                outputs = dec["concat"]([outputs, _skip])
            elif i == len(self.g_enc_depths) - self.att_layer_idx - 1:
                outputs = self.dec_attn(outputs, training=training)
            else:
                outputs = tf.nn.tanh(outputs)
        outputs = self.outp(outputs)
        return outputs

    def get_config(self):
        conf = self.noisy.get_config()
        for enc in self.enc:
            conf.update(enc[0].get_config())
            conf.update(enc[1].get_config())
        conf.update(self.enc_attn.get_config())
        conf.update(self.z_concat.get_config())
        for dec in self.dec:
            conf.update(dec[0].get_config())
            conf.update(dec[1].get_config())
            if dec[2] is not None:
                conf.update(dec[2].get_config())
        conf.update(self.dec_attn.get_config())
        conf.update(self.out.get_config())
        return conf


class Discriminator(tf.keras.Model):
    def __init__(self,
                 d_num_fmaps,
                 window_size,
                 att_layer_idx,
                 kwidth=31,
                 ratio=2,
                 leakyrelu=True,
                 name="segan_disc",
                 **kwargs):
        super(Discriminator, self).__init__(name=name, **kwargs)
        self.window_size = window_size
        self.att_layer_idx = att_layer_idx
        self.clean_wav = Reshape1to3("segan_d_reshape_1_to_3_clean")
        self.noisy_wav = Reshape1to3("segan_d_reshape_1_to_3_noisy")
        self.concat = tf.keras.layers.Concatenate(name="segan_d_concat_clean_noisy", axis=3)
        self.gauss = GaussianNoise(name="segan_d_gaussian_noise")
        self.blocks = []
        for block_idx, nfmaps in enumerate(d_num_fmaps):
            dc = SnDownConv(depth=nfmaps, kwidth=kwidth, pool=ratio,
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
        self.attn = SnNonLocalBlockSim(d_num_fmaps[self.att_layer_idx],
                                       name=f"{name}_sn_non_local_block_sim")
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
            if i == self.att_layer_idx:
                outputs = self.attn(outputs, training=training)
        outputs = tf.squeeze(outputs, axis=2)
        outputs = self.conv(outputs, training=training)
        outputs = tf.squeeze(outputs, axis=-1)
        return self.dense(outputs, training=training)
