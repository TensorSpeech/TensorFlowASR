from __future__ import absolute_import

import tensorflow as tf
from models.segan.Ops import DownConv, VirtualBatchNorm, \
  GaussianNoise, Reshape1to3, PreEmph


class DiscBlock(tf.keras.layers.Layer):
  def __init__(self, kwidth, nfmaps, pooling=2, name="disc_block", **kwargs):
    self.kwidth = kwidth
    self.nfmaps = nfmaps
    self.pooling = pooling
    super(DiscBlock, self).__init__(name=name, **kwargs)

  def call(self, inputs, training=False):
    hi = DownConv(depth=self.nfmaps,
                  kwidth=self.kwidth,
                  pool=self.pooling,
                  name=f"downconv")(inputs, training=training)
    hi = VirtualBatchNorm(hi, name=f"vbn")(hi)
    hi = tf.keras.layers.LeakyReLU(alpha=0.3, name=f"leakyrelu")(hi)
    return hi


def create_discriminator(d_num_fmaps, window_size, noise_std=0.,
                         kwidth=31, ratio=2, coeff=0.95):
  clean_signal = tf.keras.Input(shape=(window_size,),
                                name="disc_clean_input", dtype=tf.float32)
  noisy_signal = tf.keras.Input(shape=(window_size,),
                                name="disc_noisy_input", dtype=tf.float32)

  clean_wav = PreEmph(coeff=coeff, name="disc_clean_preemph")(clean_signal)
  noisy_wav = PreEmph(coeff=coeff, name="disc_noisy_preemph")(noisy_signal)
  clean_wav = Reshape1to3("segan_d_reshape_1_to_3_clean")(clean_wav)
  noisy_wav = Reshape1to3("segan_d_reshape_1_to_3_noisy")(noisy_wav)
  hi = tf.keras.layers.Concatenate(name="segan_d_concat_clean_noisy",
                                   axis=3)([clean_wav, noisy_wav])
  # after concatenation shape = [batch_size, 16384, 1, 2]

  hi = GaussianNoise(std=noise_std,
                     name="segan_d_gaussian_noise")(hi)

  for block_idx, nfmaps in enumerate(d_num_fmaps):
    hi = DiscBlock(kwidth=kwidth,
                   nfmaps=nfmaps,
                   pooling=ratio,
                   name=f"segan_d_{block_idx}")(hi)
  hi = tf.keras.layers.Flatten(name="segan_d_flatten")(hi)
  hi = tf.expand_dims(hi, -1, name="segan_d_expand_dims")
  hi = tf.keras.layers.Conv1D(filters=1, kernel_size=1,
                              strides=1, padding="same",
                              name="segan_d_conv1d")(hi)
  hi = tf.squeeze(hi, name="segan_d_squeeze")
  hi = tf.keras.layers.Dense(1, name="segan_d_fully_connected")(hi)
  # output_shape = [1]
  return tf.keras.Model(inputs=[clean_signal, noisy_signal], outputs=hi)


@tf.function
def discriminator_loss(d_real_logit, d_fake_logit):
  real_loss = tf.reduce_mean(tf.math.squared_difference(d_real_logit, 1.))
  fake_loss = tf.reduce_mean(tf.math.squared_difference(d_fake_logit, 0.))
  return real_loss + fake_loss
