from __future__ import absolute_import

import tensorflow as tf
from models.segan.Ops import DownConv, DeConv, \
  Reshape1to3, Reshape3to1, PreEmph, DeEmph


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

  def __call__(self, inputs, training=False):
    # input_shape = [batch_size, 16384, 1, 1]
    c = inputs
    for layer_idx, layer_depth in enumerate(self.g_enc_depths):
      c = DownConv(depth=layer_depth,
                   kwidth=self.kwidth,
                   pool=self.ratio,
                   name=f"{self.name}_conv_{layer_idx}")(c, training)
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

  def __call__(self, inputs, skips, training=False):
    assert skips and len(skips) > 0
    output = inputs
    for layer_idx, layer_depth in enumerate(self.g_enc_depths):
      output = DeConv(depth=layer_depth,
                      kwidth=self.kwidth,
                      dilation=self.ratio,
                      name=f"{self.name}_deconv_{layer_idx}")(output, training)
      output = tf.keras.layers.PReLU(name=f"{self.name}_prelu_{layer_idx}")(output)
      _skip = skips[-(layer_idx + 1)]
      output = tf.keras.layers.Concatenate(axis=2)([output, _skip])
    return output
    # output_shape = [batch_size, 16384, 1, 1]


class Generator(tf.keras.Model):
  def __init__(self, g_enc_depths, kwidth=31, ratio=2, coeff=0.95):
    super(Generator, self).__init__()
    self.kwidth = kwidth
    self.ratio = ratio
    self.g_dec_depths = g_enc_depths.copy()
    self.g_dec_depths.reverse()
    self.pre_emph = PreEmph(coeff=coeff, name="segan_g_preemph")
    self.reshape_input = Reshape1to3("segan_g_reshape_input")
    self.encoder = GEncoder(g_enc_depths=g_enc_depths,
                            kwidth=self.kwidth,
                            ratio=self.ratio)
    self.z = Z()
    self.decoder = GDecoder(g_dec_depths=self.g_dec_depths,
                            kwidth=self.kwidth,
                            ratio=self.ratio)
    self.reshape_output = Reshape3to1("segan_g_reshape_output")
    self.de_emph = DeEmph(coeff=coeff, name="segan_g_deemph")

  def __call__(self, inputs, training=False):
    # input_shape = [batch_size, 16384]
    inputs = self.pre_emph(inputs)
    inputs = self.reshape_input(inputs)
    output, skips = self.encoder(inputs, training)
    output = self.z(output)
    output = self.decoder(inputs=output,
                          skips=skips,
                          training=training)
    output = self.reshape_output(output)
    # output_shape = [batch_size, 16384]
    return self.de_emph(output)

  @tf.function
  def loss(self, y_true, y_pred, l1_lambda, d_fake_logit):
    l1_loss = l1_lambda * tf.reduce_mean(tf.abs(tf.math.subtract(y_pred, y_true)))
    g_adv_loss = tf.reduce_mean(tf.math.squared_difference(d_fake_logit, 1.))
    return l1_loss + g_adv_loss
