from __future__ import absolute_import

import tensorflow as tf
from models.segan.Ops import DownConv, DeConv, \
  Reshape1to3, Reshape3to1, PreEmph, SeganPrelu


class Z(tf.keras.layers.Layer):
  def __init__(self, batch_size, mean=0., stddev=1., name="segan_z", **kwargs):
    self.mean = mean,
    self.stddev = stddev
    self.batch_size = batch_size
    super(Z, self).__init__(name=name, **kwargs)

  def build(self, input_shape):
    self.z = self.add_weight(shape=[self.batch_size] + input_shape[1:], name="z", trainable=False,
                             initializer=tf.random_normal_initializer(mean=self.mean, stddev=self.stddev))

  def call(self, inputs, training=False):
    return tf.keras.layers.Concatenate(axis=3)([self.z, inputs])


def create_generator(batch_size, g_enc_depths, window_size, kwidth=31, ratio=2, coeff=0.95):
  g_dec_depths = g_enc_depths.copy()
  g_dec_depths.reverse()
  g_dec_depths = g_dec_depths[1:]
  skips = []

  # input_shape = [batch_size, 16384]
  signal = tf.keras.Input(shape=(window_size,), batch_size=batch_size,
                          name="noisy_input", dtype=tf.float32)
  pre_emph = PreEmph(coeff=coeff, name="segan_g_preemph")(signal)
  c = Reshape1to3("segan_g_reshape_input")(pre_emph)
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
  output = Z(batch_size=batch_size)(c)
  # Decoder
  for layer_idx, layer_depth in enumerate(g_dec_depths):
    output = DeConv(depth=layer_depth,
                    kwidth=kwidth,
                    dilation=ratio,
                    name=f"segan_g_deconv_{layer_idx}")(output)
    output = SeganPrelu(name=f"segan_g_deconv_prelu_{layer_idx}")(output)
    _skip = skips[-(layer_idx + 1)]
    output = tf.keras.layers.Concatenate(axis=3, name=f"concat_skip_{layer_idx}")([output, _skip])
  output = DeConv(depth=1, kwidth=kwidth, dilation=ratio, name=f"segan_g_deconv_last")(output)

  reshape_output = Reshape3to1("segan_g_reshape_output")(output)
  # output_shape = [batch_size, 16384]

  return tf.keras.Model(inputs=signal, outputs=reshape_output, name="segan_gen")


@tf.function
def generator_loss(y_true, y_pred, l1_lambda, d_fake_logit):
  l1_loss = l1_lambda * tf.reduce_mean(tf.abs(tf.subtract(y_pred, y_true)))
  g_adv_loss = tf.reduce_mean(tf.math.squared_difference(d_fake_logit, 1.))
  return l1_loss + g_adv_loss
