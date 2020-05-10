from __future__ import absolute_import

import tensorflow as tf
from utils.Utils import get_length


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
                     last_activation='linear', streaming_size=None, name="ctc_model"):
  if streaming_size:
    # Fixed input shape is required for live streaming audio
    if speech_conf["is_delta"]:
      input_shape = (1, streaming_size, speech_conf["num_feature_bins"] * 3)
    else:
      input_shape = (1, streaming_size, speech_conf["num_feature_bins"])
    features = tf.keras.layers.Input(batch_shape=input_shape,
                                     dtype=tf.float32, name="features")
    # features = self.speech_featurizer(signal)
    outputs = base_model(features=features, streaming=True)
  else:
    if speech_conf["is_delta"]:
      input_shape = (None, speech_conf["num_feature_bins"] * 3)
    else:
      input_shape = (None, speech_conf["num_feature_bins"])
    features = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32, name="features")
    # features = self.speech_featurizer(signal)
    outputs = base_model(features=features, streaming=False)

  # Fully connected layer
  outputs = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(units=num_classes, activation=last_activation,
                          use_bias=True), name="fully_connected")(outputs)

  model = tf.keras.Model(inputs=features, outputs=outputs, name=name)
  return model, base_model.optimizer


def create_ctc_train_model(ctc_model, last_activation, num_classes, name="ctc_train_model"):
  input_length = tf.keras.Input(shape=(), dtype=tf.int32, name="input_length")
  label_length = tf.keras.Input(shape=(), dtype=tf.int32, name="label_length")
  label = tf.keras.Input(shape=(None,), dtype=tf.int32, name="label")

  if last_activation != "softmax":
    ctc_loss = tf.keras.layers.Lambda(
      ctc_loss_keras_2, arguments={"num_classes": num_classes},
      output_shape=(1,), name="ctc_loss")([ctc_model.outputs[0], input_length, label, label_length])
  else:
    ctc_loss = tf.keras.layers.Lambda(
      ctc_loss_keras, output_shape=(1,),
      name="ctc_loss")([ctc_model.outputs[0], input_length, label, label_length])

  return tf.keras.Model(
    inputs=(ctc_model.inputs, input_length, label, label_length),
    outputs=ctc_loss, name=name)


@tf.function
def ctc_loss_1(y_true, y_pred, input_length, label_length, num_classes):
  return tf.reduce_mean(tf.keras.backend.ctc_batch_cost(
    y_pred=y_pred,
    input_length=tf.expand_dims(input_length, -1),
    y_true=tf.cast(y_true, tf.int32),
    label_length=tf.expand_dims(label_length, -1)
  ))


@tf.function
def ctc_loss(y_true, y_pred, input_length, label_length, num_classes):
  return tf.reduce_mean(tf.nn.ctc_loss(
    labels=tf.cast(y_true, tf.int32),
    logit_length=input_length,
    logits=y_pred,
    label_length=label_length,
    logits_time_major=False,
    blank_index=num_classes - 1
  ))


@tf.function
def ctc_loss_keras(layer):
  y_pred, input_length, y_true, label_length = layer
  return tf.keras.backend.ctc_batch_cost(
    y_pred=y_pred,
    input_length=tf.expand_dims(input_length, -1),
    y_true=y_true,
    label_length=tf.expand_dims(label_length, -1))


@tf.function
def ctc_loss_keras_2(layer, **kwargs):
  num_classes = kwargs["num_classes"]
  y_pred, input_length, y_true, label_length = layer
  return tf.reduce_mean(tf.nn.ctc_loss(
    labels=y_true,
    logit_length=input_length,
    logits=y_pred,
    label_length=label_length,
    logits_time_major=False,
    blank_index=num_classes - 1))
