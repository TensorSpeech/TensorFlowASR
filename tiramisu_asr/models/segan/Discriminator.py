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

from .Ops import DownConv, VirtualBatchNorm, GaussianNoise, Reshape1to3


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
