from __future__ import absolute_import

import tensorflow as tf
import numpy as np


class PreEmph(tf.keras.layers.Layer):
  def __init__(self, coeff=0.95, name="pre_emph"):
    super(PreEmph, self).__init__(name=name, trainable=False)
    self.coeff = coeff

  def __call__(self, inputs):
    # input_shape = [batch_size, 16384]
    def map_fn(elem):
      x0 = tf.reshape(elem[0], [1, ])
      diff = elem[1:] - self.coeff * elem[:-1]
      return tf.concat([x0, diff], axis=0)
    return tf.map_fn(map_fn, inputs)


class DeEmph(tf.keras.layers.Layer):
  def __init__(self, coeff=0.95, name="de_emph"):
    super(DeEmph, self).__init__(name=name, trainable=False)
    self.coeff = coeff

  def __call__(self, inputs):
    # input_shape = [batch_size, 16384]
    def map_fn(elem):
      elem = elem.numpy()
      if self.coeff <= 0:
        return elem
      x = np.zeros(elem.shape[0], dtype=np.float32)
      x[0] = elem[0]
      for n in range(1, elem.shape[0], 1):
        x[n] = self.coeff * x[n - 1] + elem[n]
      return tf.convert_to_tensor(x)
    return tf.map_fn(map_fn, inputs)


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

  def __call__(self, inputs, training=False):
    return self.layer(inputs, training)


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

  def __call__(self, inputs, training=False):
    return self.layer(inputs, training)


class VirtualBatchNorm(tf.keras.layers.Layer):
  def __init__(self, x, name, epsilon=1e-5):
    super(VirtualBatchNorm, self).__init__()
    assert isinstance(epsilon, float)
    shape = x.get_shape().as_list()
    assert len(shape) == 3, shape
    self.epsilon = epsilon
    self.name = name
    self.mean = tf.reduce_mean(x, [0, 1], keep_dims=True)
    self.mean_sq = tf.reduce_mean(tf.square(x), [0, 1], keep_dims=True)
    self.batch_size = int(x.get_shape()[0])
    assert x is not None
    assert self.mean is not None
    assert self.mean_sq is not None
    out = self.__normalize(x, self.mean, self.mean_sq)
    self.reference_out = out

  def build(self, input_shape):
    self.gamma = self.add_weight(
      shape=[input_shape[-1]], name=f"{self.name}_gamma",
      initializer=tf.random_normal_initializer(1., 0.02)
    )
    self.beta = self.add_weight(
      shape=[input_shape[-1]], name=f"{self.name}_beta",
      initializer=tf.keras.initializers.constant(0.)
    )

  def __call__(self, x):
    new_coeff = 1 / (self.batch_size + 1.)
    old_coeff = 1. - new_coeff
    new_mean = tf.reduce_mean(x, [0, 1], keep_dims=True)
    new_mean_sq = tf.reduce_mean(tf.square(x), [0, 1], keep_dims=True)
    mean = new_coeff * new_mean + old_coeff * self.mean
    mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
    out = self.__normalize(x, mean, mean_sq)
    return out

  def __normalize(self, x, mean, mean_sq):
    gamma = tf.reshape(self.gamma, [1, 1, -1])
    beta = tf.reshape(self.beta, [1, 1, -1])
    std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
    out = x - mean
    out = out / std
    out = out * gamma
    out = out + beta
    return out


class GaussianNoise(tf.keras.layers.Layer):
  def __init__(self, name, std):
    super(GaussianNoise, self).__init__(name=name, trainable=False)
    self.std = std

  def __call__(self, inputs):
    noise = tf.keras.backend.random_normal(
      shape=inputs.get_shape().as_list(),
      mean=0.0, stddev=self.std,
      dtype=tf.float32)
    return inputs + noise


class Reshape1to3(tf.keras.layers.Layer):
  def __init__(self, name="reshape_1_to_3"):
    super(Reshape1to3, self).__init__(name=name, trainable=False)

  def __call__(self, inputs):
    batch_size = tf.shape(inputs)[0]
    return tf.reshape(inputs, [batch_size, -1, 1, 1])


class Reshape3to1(tf.keras.layers.Layer):
  def __init__(self, name="reshape_3_to_1"):
    super(Reshape3to1, self).__init__(name=name, trainable=False)

  def __call__(self, inputs):
    batch_size = tf.shape(inputs)[0]
    return tf.reshape(inputs, [batch_size, -1])
