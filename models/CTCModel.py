from __future__ import absolute_import

import tensorflow as tf
from utils.Utils import mask_nan, get_length
from utils.Schedules import BoundExponentialDecay
from featurizers.SpeechFeaturizer import SpeechFeaturizer


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


class CTCModel:
  def __init__(self, base_model, num_classes, sample_rate, frame_ms, stride_ms,
               num_feature_bins, learning_rate, min_lr=0.0, streaming_size=None):
    self.optimizer = base_model.optimizer(learning_rate=learning_rate)
    self.num_classes = num_classes
    self.streaming_size = streaming_size
    self.speech_featurizer = SpeechFeaturizer(sample_rate=sample_rate, frame_ms=frame_ms,
                                              stride_ms=stride_ms, num_feature_bins=num_feature_bins,
                                              feature_type="mfcc", name="speech_featurizer")
    self.model = self.create(base_model)

  def __call__(self, *args, **kwargs):
    return self.model(*args, **kwargs)

  def create(self, base_model):
    if self.streaming_size:
      # Fixed input shape is required for live streaming audio
      signal = tf.keras.layers.Input(batch_shape=(1, self.streaming_size),
                                     dtype=tf.float32, name="features")
      features = self.speech_featurizer(signal)
      outputs = base_model(features=features, streaming=True)
    else:
      signal = tf.keras.layers.Input(shape=(None,), dtype=tf.float32, name="features")
      features = self.speech_featurizer(signal)
      outputs = base_model(features=features, streaming=False)

    input_length = GetLength(name="input_length")(features)

    batch_size = tf.shape(outputs)[0]
    n_hidden = outputs.get_shape().as_list()[-1]
    # reshape from [B, T, A] --> [B*T, A].
    # Output shape: [n_steps * batch_size, n_hidden]
    outputs = tf.reshape(outputs, [-1, n_hidden])

    # Fully connected layer
    outputs = tf.keras.layers.Dense(units=self.num_classes,
                                    activation='softmax',
                                    name="fully_connected_softmax",
                                    use_bias=True)(outputs)

    outputs = tf.reshape(outputs,
                         [batch_size, -1, self.num_classes],
                         name="logits")

    model = tf.keras.Model(inputs=signal, outputs=[outputs, input_length])
    return model

  @tf.function
  def loss(self, y_true, y_pred, input_length, label_length):
    loss = tf.keras.backend.ctc_batch_cost(
      y_pred=y_pred,
      input_length=tf.expand_dims(input_length, -1),
      y_true=tf.cast(y_true, tf.int32),
      label_length=tf.expand_dims(label_length, -1))
    return tf.reduce_mean(mask_nan(loss))

  # @tf.function
  # def loss(self, y_true, y_pred, input_length, label_length):
  #   loss = tf.nn.ctc_loss(
  #     labels=tf.cast(y_true, tf.int32),
  #     logit_length=input_length,
  #     logits=y_pred,
  #     label_length=label_length,
  #     logits_time_major=False,
  #     blank_index=self.num_classes - 1)
  #   return tf.reduce_mean(mask_nan(loss))

  def predict(self, *args, **kwargs):
    return self.model.predict(*args, **kwargs)

  def compile(self, *args, **kwargs):
    return self.model.compile(*args, **kwargs)

  def fit(self, *args, **kwargs):
    return self.model.fit(*args, **kwargs)

  def load_model(self, model_file):
    self.model = tf.saved_model.load(model_file)

  def load_weights(self, model_file):
    self.model.load_weights(model_file)

  def summary(self, *args, **kwargs):
    return self.model.summary(*args, **kwargs)

  def to_json(self):
    return self.model.to_json()

  def save(self, model_file):
    return self.model.save(model_file)
