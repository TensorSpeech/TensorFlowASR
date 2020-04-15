from __future__ import absolute_import

import tensorflow as tf
from models.segan.Ops import DownConv, VirtualBatchNorm, \
  GaussianNoise, Reshape1to3


class DiscBlock(tf.keras.layers.Layer):
  def __init__(self, kwidth, nfmaps, pooling=2, name="disc_block"):
    self.name = name
    self.kwidth = kwidth
    self.nfmaps = nfmaps
    self.pooling = pooling
    super(DiscBlock, self).__init__(name=name)

  def __call__(self, inputs):
    hi = DownConv(depth=self.nfmaps,
                  kwidth=self.kwidth,
                  pool=self.pooling,
                  name=f"{self.name}_downconv")(inputs)
    hi = VirtualBatchNorm(hi, name=f"{self.name}_vbn")(hi)
    hi = tf.keras.layers.LeakyReLU(
      alpha=0.3, name=f"{self.name}_leakyrelu")(hi)
    return hi


class Discriminator(tf.keras.Model):
  def __init__(self, d_num_fmaps, noise_std, kwidth=31, pooling=2):
    super(Discriminator, self).__init__()
    self.d_num_fmaps = d_num_fmaps
    self.kwidth = kwidth
    self.pooling = pooling
    self.noise_std = noise_std

  def __call__(self, clean_wav, noisy_wav):
    # clean_wav_shape = [batch_size, 16384]
    # noisy_wav_shape = [batch_size, 16384]
    clean_wav = Reshape1to3("segan_d_reshape_1_to_3_clean")(clean_wav)
    noisy_wav = Reshape1to3("segan_d_reshape_1_to_3_noisy")(noisy_wav)
    hi = tf.keras.layers.Concatenate(name="segan_d_concat_clean_noisy",
                                     axis=2)([clean_wav, noisy_wav])
    # after concatenation shape = [batch_size, 16384, 1, 2]

    hi = GaussianNoise(std=self.noise_std,
                       name="segan_d_gaussian_noise")(hi)

    for block_idx, nfmaps in enumerate(self.d_num_fmaps):
      hi = DiscBlock(kwidth=self.kwidth,
                     nfmaps=nfmaps,
                     pooling=self.pooling,
                     name=f"segan_d_{block_idx}")(hi)
    hi = tf.keras.layers.Flatten(name="segan_d_flatten")(hi)
    hi = tf.expand_dims(hi, -1, name="segan_d_expand_dims")
    hi = tf.keras.layers.Conv1D(filters=1, kernel_size=1,
                                strides=1, padding="same",
                                name="segan_d_conv1d")(hi)
    hi = tf.squeeze(hi, name="segan_d_squeeze")
    hi = tf.keras.layers.Dense(1, name="segan_d_fully_connected")(hi)
    # output_shape = [1]
    return hi
