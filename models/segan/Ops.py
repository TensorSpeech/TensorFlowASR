from __future__ import absolute_import

import tensorflow as tf


class DownConv(tf.keras.layers.Layer):
  def __init__(self, depth, kwidth=5, pool=2, name="downconv"):
    super(DownConv, self).__init__(name=name)
    self.layer = tf.keras.layers.Conv2D(
      filters=depth,
      kernel_size=(kwidth, 1),
      strides=(pool, 1),
      padding="same",
      use_bias=True,
      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
      bias_initializer=tf.keras.initializers.zeros
    )

  def __call__(self, inputs):
    return self.layer(inputs)


class DeConv(tf.keras.layers.Layer):
  def __init__(self, depth, kwidth=5, dilation=2, name="deconv"):
    super(DeConv, self).__init__(name=name)
    self.layer = tf.keras.layers.Conv2DTranspose(
      filters=depth,
      kernel_size=(kwidth, 1),
      strides=(dilation, 1),
      padding="same",
      use_bias=True,
      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
      bias_initializer=tf.keras.initializers.zeros
    )
