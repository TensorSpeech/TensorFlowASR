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
from __future__ import absolute_import

import tensorflow as tf
from models.segan.Ops import DownConv, DeConv, \
    Reshape1to3, Reshape3to1, SeganPrelu


class Z(tf.keras.layers.Layer):
    def __init__(self, mean=0., stddev=1., name="segan_z", **kwargs):
        self.mean = mean,
        self.stddev = stddev
        super(Z, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=False):
        z = tf.random.normal(shape=tf.shape(inputs),
                             name="z", mean=self.mean, stddev=self.stddev)
        return tf.keras.layers.Concatenate(axis=3)([z, inputs])


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


@tf.function
def generator_loss(y_true, y_pred, l1_lambda, d_fake_logit):
    l1_loss = l1_lambda * tf.reduce_mean(tf.abs(tf.subtract(y_pred, y_true)))
    g_adv_loss = tf.reduce_mean(tf.math.squared_difference(d_fake_logit, 1.))
    return l1_loss, g_adv_loss
