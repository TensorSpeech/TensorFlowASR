from __future__ import absolute_import
import tensorflow as tf


class VBN(object):
  """
  Virtual Batch Normalization
  (modified from https://github.com/openai/improved-gan/ definition)
  """

  def __init__(self, x, name, epsilon=1e-5):
    """
    x is the reference batch
    """
    assert isinstance(epsilon, float)

    shape = x.get_shape().as_list()
    assert len(shape) == 3, shape
    with tf.variable_scope(name) as scope:
      assert name.startswith("d_") or name.startswith("g_")
      self.epsilon = epsilon
      self.name = name
      self.mean = tf.reduce_mean(x, [0, 1], keep_dims=True)
      self.mean_sq = tf.reduce_mean(tf.square(x), [0, 1], keep_dims=True)
      self.batch_size = int(x.get_shape()[0])
      assert x is not None
      assert self.mean is not None
      assert self.mean_sq is not None
      out = self._normalize(x, self.mean, self.mean_sq, "reference")
      self.reference_output = out

  def __call__(self, x):

    shape = x.get_shape().as_list()
    with tf.variable_scope(self.name) as scope:
      new_coeff = 1. / (self.batch_size + 1.)
      old_coeff = 1. - new_coeff
      new_mean = tf.reduce_mean(x, [0, 1], keep_dims=True)
      new_mean_sq = tf.reduce_mean(tf.square(x), [0, 1], keep_dims=True)
      mean = new_coeff * new_mean + old_coeff * self.mean
      mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
      out = self._normalize(x, mean, mean_sq, "live")
      return out

  def _normalize(self, x, mean, mean_sq, message):
    # make sure this is called with a variable scope
    shape = x.get_shape().as_list()
    assert len(shape) == 3
    self.gamma = tf.get_variable("gamma", [shape[-1]],
                                 initializer=tf.random_normal_initializer(1., 0.02))
    gamma = tf.reshape(self.gamma, [1, 1, -1])
    self.beta = tf.get_variable("beta", [shape[-1]],
                                initializer=tf.constant_initializer(0.))
    beta = tf.reshape(self.beta, [1, 1, -1])
    assert self.epsilon is not None
    assert mean_sq is not None
    assert mean is not None
    std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
    out = x - mean
    out = out / std
    out = out * gamma
    out = out + beta
    return out
