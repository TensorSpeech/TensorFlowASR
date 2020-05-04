from __future__ import absolute_import

import tensorflow as tf
from utils.Utils import mask_nan, get_length
from utils.Schedules import BoundExponentialDecay


class GetLength(tf.keras.layers.Layer):
  def __init__(self, name, **kwargs):
    super(GetLength, self).__init__(name=name, trainable=False, **kwargs)

  def call(self, inputs, **kwargs):
    return get_length(inputs)

  def get_config(self):
    config = super(GetLength, self).get_config()
    return config

  def from_config(self, config):
    return self(**config)


def create_ctc_model(base_model, num_classes, speech_conf,
                     learning_rate, min_lr=0.0, streaming_size=None):
  if streaming_size:
    # Fixed input shape is required for live streaming audio
    if speech_conf["is_delta"]:
      input_shape = (1, streaming_size, speech_conf["num_feature_bins"] * 3, 1)
    else:
      input_shape = (1, streaming_size, speech_conf["num_feature_bins"], 1)
    features = tf.keras.layers.Input(batch_shape=input_shape,
                                     dtype=tf.float32, name="features")
    # features = self.speech_featurizer(signal)
    outputs = base_model(features=features, streaming=True)
  else:
    if speech_conf["is_delta"]:
      input_shape = (None, speech_conf["num_feature_bins"] * 3, 1)
    else:
      input_shape = (None, speech_conf["num_feature_bins"], 1)
    features = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32, name="features")
    # features = self.speech_featurizer(signal)
    outputs = base_model(features=features, streaming=False)

  batch_size = tf.shape(outputs)[0]
  n_hidden = outputs.get_shape().as_list()[-1]
  # reshape from [B, T, A] --> [B*T, A].
  # Output shape: [n_steps * batch_size, n_hidden]
  outputs = tf.reshape(outputs, [-1, n_hidden])

  # Fully connected layer
  outputs = tf.keras.layers.Dense(units=num_classes,
                                  activation='softmax',
                                  name="fully_connected",
                                  use_bias=True)(outputs)

  outputs = tf.reshape(outputs,
                       [batch_size, -1, num_classes],
                       name="logits")

  model = tf.keras.Model(inputs=features, outputs=outputs)
  optimizer = base_model.optimizer(learning_rate=learning_rate, momentum=0.99, nesterov=True)
  return model, optimizer


# @tf.function
# def loss(self, y_true, y_pred, input_length, label_length):
#   loss = tf.keras.backend.ctc_batch_cost(
#     y_pred=y_pred,
#     input_length=tf.expand_dims(input_length, -1),
#     y_true=tf.cast(y_true, tf.int32),
#     label_length=tf.expand_dims(label_length, -1))
#   return tf.reduce_mean(mask_nan(loss))

@tf.function
def ctc_loss(y_true, y_pred, input_length, label_length, num_classes):
  return tf.reduce_mean(tf.nn.ctc_loss(
    labels=tf.cast(y_true, tf.int32),
    logit_length=input_length,
    logits=y_pred,
    label_length=label_length,
    logits_time_major=False,
    blank_index=num_classes - 1))


@tf.function
def ctc_loss_keras(y_true, y_pred):
  label_length = get_length(y_true)
  input_length = get_length(y_pred)
  return tf.keras.backend.ctc_batch_cost(
    y_pred=y_pred,
    input_length=tf.expand_dims(input_length, -1),
    y_true=tf.cast(y_true, tf.int32),
    label_length=tf.expand_dims(label_length, -1))
