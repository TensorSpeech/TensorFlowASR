from __future__ import absolute_import

import tensorflow as tf
from utils.Utils import get_length, mask_nan
from utils.Schedules import BoundExponentialDecay


class CTCModel:
  def __init__(self, num_classes, num_feature_bins, base_model,
               learning_rate, min_lr=0.0, streaming_size=None):
    self.optimizer = base_model.optimizer(
      learning_rate=BoundExponentialDecay(
          min_lr=min_lr,
          initial_learning_rate=learning_rate,
          decay_steps=5000,
          decay_rate=0.9,
          staircase=True)
    )
    self.num_classes = num_classes
    self.num_feature_bins = num_feature_bins
    self.streaming_size = streaming_size
    self.model = self.create(base_model)

  def create(self, base_model):
    if self.streaming_size:
      # Fixed input shape is required for live streaming audio
      features = tf.keras.layers.Input(
        batch_shape=(1, self.streaming_size, self.num_feature_bins, 1),
        dtype=tf.float32,
        name="features")
      outputs = base_model(features=features, streaming=True)
    else:
      features = tf.keras.layers.Input(
        shape=(None, self.num_feature_bins, 1),
        dtype=tf.float32,
        name="features")
      outputs = base_model(features=features, streaming=False)

    batch_size = tf.shape(outputs)[0]
    n_hidden = outputs.get_shape().as_list()[-1]
    # reshape from [B, T, A] --> [B*T, A].
    # Output shape: [n_steps * batch_size, n_hidden]
    outputs = tf.reshape(outputs, [-1, n_hidden])

    # Fully connected layer
    outputs = tf.keras.layers.Dense(units=self.num_classes,
                                    activation='softmax',
                                    name="fully_connected",
                                    use_bias=True)(outputs)

    outputs = tf.reshape(outputs,
                         [batch_size, -1, self.num_classes],
                         name="logits")

    model = tf.keras.Model(inputs=features, outputs=outputs)
    return model

  @tf.function
  def loss(self, y_true, y_pred):
    label_length = get_length(y_true)
    input_length = get_length(y_pred)
    loss = tf.keras.backend.ctc_batch_cost(
      y_pred=y_pred,
      input_length=input_length,
      y_true=tf.cast(tf.squeeze(y_true, -1), tf.int32),
      label_length=label_length)
    return mask_nan(loss)

  # @tf.function
  # def loss(self, y_true, y_pred):
  #   label_length = get_length(y_true)
  #   input_length = get_length(y_pred)
  #   loss = tf.nn.ctc_loss(
  #     labels=tf.cast(tf.squeeze(y_true, -1), tf.int32),
  #     logit_length=input_length,
  #     logits=y_pred,
  #     label_length=label_length,
  #     logits_time_major=False,
  #     blank_index=self.num_classes - 1)
  #   return mask_nan(loss)

  def predict(self, *args, **kwargs):
    return self.model.predict(*args, **kwargs)

  def compile(self, *args, **kwargs):
    return self.model.compile(*args, **kwargs)

  def fit(self, *args, **kwargs):
    return self.model.fit(*args, **kwargs)

  def load_model(self, model_file):
    self.model = tf.keras.models.load_model(model_file)

  def load_weights(self, model_file):
    self.model.load_weights(model_file)

  def summary(self, *args, **kwargs):
    return self.model.summary(*args, **kwargs)

  def to_json(self):
    return self.model.to_json()
