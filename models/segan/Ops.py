from __future__ import absolute_import

import tensorflow as tf


class PreEmph(tf.keras.layers.Layer):
  def __init__(self, coeff=0.95, name="pre_emph", **kwargs):
    super(PreEmph, self).__init__(name=name, trainable=False, **kwargs)
    self.coeff = coeff
    self.cname = name

  def call(self, inputs, trainig=False):
    # input_shape = [batch_size, 16384]
    def map_fn(elem):
      x0 = tf.reshape(elem[0], [1, ])
      diff = elem[1:] - self.coeff * elem[:-1]
      return tf.concat([x0, diff], axis=0)

    return tf.map_fn(map_fn, inputs, name=self.cname)


class DownConv(tf.keras.layers.Layer):
  def __init__(self, depth, kwidth=5, pool=2, name="downconv", **kwargs):
    super(DownConv, self).__init__(name=name, **kwargs)
    self.layer = tf.keras.layers.Conv2D(
      filters=depth,
      kernel_size=(kwidth, 1),
      strides=(pool, 1),
      padding="same",
      use_bias=True,
      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
      bias_initializer=tf.keras.initializers.zeros
    )

  def call(self, inputs, training=False):
    return self.layer(inputs, training=training)


class DeConv(tf.keras.layers.Layer):
  def __init__(self, depth, kwidth=5, dilation=2, name="deconv", **kwargs):
    super(DeConv, self).__init__(name=name, **kwargs)
    self.layer = tf.keras.layers.Conv2DTranspose(
      filters=depth,
      kernel_size=(kwidth, 1),
      strides=(dilation, 1),
      padding="same",
      use_bias=True,
      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
      bias_initializer=tf.keras.initializers.zeros
    )

  def call(self, inputs, training=False):
    return self.layer(inputs, training=training)


class VirtualBatchNorm(tf.keras.layers.Layer):
  def __init__(self, batch_size, name, epsilon=1e-5, **kwargs):
    super(VirtualBatchNorm, self).__init__(name=name, **kwargs)
    assert isinstance(epsilon, float)
    self.epsilon = epsilon
    self.batch_size = batch_size
    self.first = False

  def build(self, input_shape):
    init = tf.random_normal_initializer()
    self.gamma = self.add_weight(
      shape=[input_shape[-1]], name="gamma",
      initializer=tf.random_normal_initializer(1., 0.02),
      trainable=True
    )
    self.beta = self.add_weight(
      shape=[input_shape[-1]], name="beta",
      initializer=tf.keras.initializers.constant(0.),
      trainable=True
    )
    self.mean = tf.Variable(initial_value=init(shape=[1, 1, 1] + [input_shape[-1]]),
                            name="mean", dtype=tf.float32, trainable=False)
    self.mean_sq = tf.Variable(initial_value=init(shape=[1, 1, 1] + [input_shape[-1]]),
                               name="mean_sq", dtype=tf.float32, trainable=False)
    self.built = True

  def call(self, x, **kwargs):
    if not self.first:
      self.mean.assign(tf.reduce_mean(x, [0, 1, 2], keepdims=True))
      self.mean_sq.assign(tf.reduce_mean(tf.square(x), [0, 1, 2], keepdims=True))
      self.first = True

    new_coeff = 1. / (self.batch_size + 1.)
    old_coeff = 1. - new_coeff
    new_mean = tf.reduce_mean(x, [1, 2], keepdims=True)
    new_mean_sq = tf.reduce_mean(tf.square(x), [1, 2], keepdims=True)
    mean = new_coeff * new_mean + old_coeff * self.mean
    mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
    out = self.normalize(x, mean, mean_sq)
    return out

  def normalize(self, x, mean, mean_sq):
    gamma = tf.reshape(self.gamma, [1, 1, 1, -1])
    beta = tf.reshape(self.beta, [1, 1, 1, -1])
    std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
    out = x - mean
    out = out / std
    out = out * gamma
    out = out + beta
    return out


class GaussianNoise(tf.keras.layers.Layer):
  def __init__(self, name, std, **kwargs):
    super(GaussianNoise, self).__init__(trainable=False, name=name, **kwargs)
    self.std = std
    self.cname = name

  def call(self, inputs, training=False):
    noise = tf.keras.backend.random_normal(
      shape=tf.shape(inputs),
      mean=0.0, stddev=self.std,
      dtype=tf.float32)
    return inputs + noise


class Reshape1to3(tf.keras.layers.Layer):
  def __init__(self, name="reshape_1_to_3", **kwargs):
    super(Reshape1to3, self).__init__(trainable=False, name=name, **kwargs)

  def call(self, inputs, training=False):
    batch_size = tf.shape(inputs)[0]
    width = inputs.get_shape().as_list()[1]
    return tf.reshape(inputs, [batch_size, width, 1, 1])


class Reshape3to1(tf.keras.layers.Layer):
  def __init__(self, name="reshape_3_to_1", **kwargs):
    super(Reshape3to1, self).__init__(trainable=False, name=name, **kwargs)

  def call(self, inputs, training=False):
    batch_size = tf.shape(inputs)[0]
    width = inputs.get_shape().as_list()[1]
    return tf.reshape(inputs, [batch_size, width])


class SeganPrelu(tf.keras.layers.Layer):
  def __init__(self, name="segan_prelu", **kwargs):
    super(SeganPrelu, self).__init__(trainable=True, name=name, **kwargs)

  def build(self, input_shape):
    self.alpha = self.add_weight(name="alpha",
                                 shape=input_shape[-1],
                                 initializer=tf.keras.initializers.zeros,
                                 dtype=tf.float32,
                                 trainable=True)

  def call(self, x, training=False):
    pos = tf.nn.relu(x)
    neg = self.alpha * (x - tf.abs(x)) * .5
    return pos + neg
