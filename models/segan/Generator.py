from __future__ import absolute_import

import tensorflow as tf
from models.segan.Ops import DownConv, DeConv


class Z(tf.keras.layers.Layer):
  def __init__(self, mean=0., stddev=1., name="segan_z"):
    self.mean = mean,
    self.stddev = stddev
    self.name = name
    super(Z, self).__init__(name=name)

  def __call__(self, inputs):
    batch = tf.shape(inputs)[0]
    dim = tf.shape(inputs)[1]
    depth = tf.shape(inputs)[-1]
    z = tf.keras.backend.random_normal(shape=[batch, dim, depth],
                                       mean=self.mean, stddev=self.stddev)
    return tf.keras.layers.Concatenate(axis=2, name=f"{self.name}_concat")([z, inputs])


class GEncoder(tf.keras.layers.Layer):
  def __init__(self, g_enc_depths, kwidth=5, ratio=2,
               name="segan_g_encoder"):
    self.g_enc_depths = g_enc_depths
    self.kwidth = kwidth
    self.ratio = ratio
    self.skips = []
    self.name = name
    super(GEncoder, self).__init__(name=name)

  def __call__(self, inputs):
    c = inputs
    for layer_idx, layer_depth in enumerate(self.g_enc_depths):
      c = DownConv(depth=layer_depth,
                   kwidth=self.kwidth,
                   pool=self.ratio,
                   name=f"{self.name}_conv_{layer_idx}")(c)
      if layer_idx < len(self.g_enc_depths) - 1:
        self.skips.append(c)
      c = tf.keras.layers.PReLU(name=f"{self.name}_prelu_{layer_idx}")(c)
    return c, self.skips


class GDecoder(tf.keras.layers.Layer):
  def __init__(self, g_dec_depths, kwidth=5,
               ratio=2, name="segan_g_decoder"):
    self.g_dec_depths = g_dec_depths
    self.kwidth = kwidth
    self.ratio = ratio
    self.name = name
    super(GDecoder, self).__init__(name=name)

  def __call__(self, inputs, skips):
    assert skips and len(skips) > 0
    output = inputs
    for layer_idx, layer_depth in enumerate(self.g_enc_depths):
      output = DeConv(depth=layer_depth,
                      kwidth=self.kwidth,
                      dilation=self.ratio,
                      name=f"{self.name}_deconv_{layer_idx}")(output)
      output = tf.keras.layers.PReLU(name=f"{self.name}_prelu_{layer_idx}")(output)
      _skip = skips[-(layer_idx + 1)]
      output = tf.keras.layers.Concatenate(axis=2)([output, _skip])
    return output


class Generator(tf.keras.Model):
  def __init__(self, g_enc_depths, g_dec_depths, kwidth=31, ratio=2):
    super(Generator, self).__init__()
    self.kwidth = kwidth
    self.ratio = ratio
    self.encoder = GEncoder(g_enc_depths=g_enc_depths,
                            kwidth=self.kwidth,
                            ratio=self.ratio)
    self.z = Z()
    self.decoder = GDecoder(g_dec_depths=g_dec_depths,
                            kwidth=self.kwidth,
                            ratio=self.ratio)

  def __call__(self, inputs):
    output, skips = self.encoder(inputs)
    output = self.z(output)
    output = self.decoder(inputs=output, skips=skips)
    return output
