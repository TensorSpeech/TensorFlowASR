from __future__ import absolute_import

import tensorflow as tf
from utils.Utils import get_length
from featurizers.SpeechFeaturizer import compute_feature_dim
try:
  from warprnnt_tensorflow import rnnt_loss
except Exception as e:
  raise ImportError(f"{e}: Install warprnnt_tensorflow from https://github.com/HawkAaron/warp-transducer/tree/master/tensorflow_binding")

def create_transducer_model(base_model, num_classes, speech_conf,
                            last_activation='linear', streaming_size=None, name="transducer_model"):
  feature_dim, channel_dim = compute_feature_dim(speech_conf)
  if streaming_size:
    # Fixed input shape is required for live streaming audio
    x = tf.keras.layers.Input(batch_shape=(1, streaming_size, feature_dim, channel_dim),
                              dtype=tf.float32, name="features")
    y = tf.keras.layers.Input(batch_shape=(1, None), dtype=tf.int32, name="predicted")
    # features = self.speech_featurizer(signal)
    outputs = base_model(x=x, y=y, streaming=True)
  else:
    features = tf.keras.layers.Input(shape=(None, feature_dim, channel_dim),
                                     dtype=tf.float32, name="features")
    y = tf.keras.layers.Input(batch_shape=(None,), dtype=tf.int32, name="predicted")
    # features = self.speech_featurizer(signal)
    outputs = base_model(x=x, y=y, streaming=False)

  # Fully connected layer
  outputs = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(units=num_classes, activation=last_activation,
                          use_bias=True), name="fully_connected")(outputs)

  model = tf.keras.Model(inputs=features, outputs=outputs, name=name)
  return model, base_model.optimizer

@tf.function
def transducer_loss(y_true, y_pred, input_length, label_length, last_activation="linear"):
  y_true = tf.cast(y_true, dtype=tf.int32)
  if last_activation != "softmax":
    y_pred = tf.nn.log_softmax(y_pred)
  return rnnt_loss(y_pred, y_true, input_length, label_length - 1)
