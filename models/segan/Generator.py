from __future__ import absolute_import

import tensorflow as tf
from models.segan.Ops import DownConv, DeConv, \
  Reshape1to3, Reshape3to1, PreEmph, DeEmph, SeganPrelu


class Z(tf.keras.layers.Layer):
  def __init__(self, mean=0., stddev=1., name="segan_z", **kwargs):
    self.mean = mean,
    self.stddev = stddev
    super(Z, self).__init__(name=name, **kwargs)

  def call(self, inputs, training=False):
    z = tf.keras.backend.random_normal(shape=tf.shape(inputs),
                                       mean=self.mean, stddev=self.stddev)
    return tf.keras.layers.Concatenate(axis=3)([z, inputs])


class GEncoder(tf.keras.layers.Layer):
  def __init__(self, g_enc_depths, kwidth=5, ratio=2,
               name="segan_g_encoder", **kwargs):
    self.g_enc_depths = g_enc_depths
    self.kwidth = kwidth
    self.ratio = ratio
    self.skips = []
    super(GEncoder, self).__init__(name=name, **kwargs)

  def call(self, inputs, training=False):
    # input_shape = [batch_size, 16384, 1, 1]
    c = inputs
    for layer_idx, layer_depth in enumerate(self.g_enc_depths):
      c = DownConv(depth=layer_depth,
                   kwidth=self.kwidth,
                   pool=self.ratio,
                   name=f"downconv_{layer_idx}")(c, training=training)
      if layer_idx < len(self.g_enc_depths) - 1:
        self.skips.append(c)
      c = SeganPrelu(name=f"downconv_prelu_{layer_idx}")(c)
    return c, self.skips


class GDecoder(tf.keras.layers.Layer):
  def __init__(self, g_dec_depths, skips, kwidth=5,
               ratio=2, name="segan_g_decoder", **kwargs):
    self.g_dec_depths = g_dec_depths
    self.kwidth = kwidth
    self.ratio = ratio
    self.skips = skips
    super(GDecoder, self).__init__(name=name, **kwargs)

  def call(self, inputs, training=False):
    assert self.skips and len(self.skips) > 0
    output = inputs
    for layer_idx, layer_depth in enumerate(self.g_dec_depths):
      output = DeConv(depth=layer_depth,
                      kwidth=self.kwidth,
                      dilation=self.ratio,
                      name=f"deconv_{layer_idx}")(output, training=training)
      output = SeganPrelu(name=f"deconv_prelu_{layer_idx}")(output)
      _skip = self.skips[-(layer_idx + 1)]
      output = tf.keras.layers.Concatenate(axis=3, name=f"concat_skip_{layer_idx}")([output, _skip])
    return DeConv(depth=1, kwidth=self.kwidth, dilation=self.ratio,
                  name=f"deconv_last")(output, training=training)
    # output_shape = [batch_size, 16384, 1, 1]


def create_generator(g_enc_depths, window_size, kwidth=31, ratio=2, coeff=0.95):
  g_dec_depths = g_enc_depths.copy()
  g_dec_depths.reverse()
  g_dec_depths = g_dec_depths[1:]

  # input_shape = [batch_size, 16384]
  signal = tf.keras.Input(shape=(window_size,),
                          name="noisy_input", dtype=tf.float32)
  pre_emph = PreEmph(coeff=coeff, name="segan_g_preemph")(signal)
  reshape_input = Reshape1to3("segan_g_reshape_input")(pre_emph)
  encoder, skips = GEncoder(g_enc_depths=g_enc_depths,
                            kwidth=kwidth,
                            ratio=ratio)(reshape_input)
  z = Z()(encoder)
  decoder = GDecoder(g_dec_depths=g_dec_depths, skips=skips,
                     kwidth=kwidth, ratio=ratio)(z)
  reshape_output = Reshape3to1("segan_g_reshape_output")(decoder)
  de_emph = DeEmph(coeff=coeff, name="segan_g_deemph")(reshape_output)
  # output_shape = [batch_size, 16384]

  return tf.keras.Model(inputs=signal, outputs=de_emph)


@tf.function
def generator_loss(y_true, y_pred, l1_lambda, d_fake_logit):
  l1_loss = l1_lambda * tf.reduce_mean(tf.abs(tf.math.subtract(y_pred, y_true)))
  g_adv_loss = tf.reduce_mean(tf.math.squared_difference(d_fake_logit, 1.))
  return l1_loss + g_adv_loss
