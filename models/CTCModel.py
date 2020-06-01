from __future__ import absolute_import

import tensorflow as tf
from featurizers.SpeechFeaturizer import compute_feature_dim


def create_ctc_model(base_model, num_classes, speech_conf, streaming_size=None, name="ctc_model"):
    feature_dim, channel_dim = compute_feature_dim(speech_conf)
    if streaming_size:
        # Fixed input shape is required for live streaming audio
        features = tf.keras.Input(batch_shape=(1, streaming_size, feature_dim, channel_dim),
                                  dtype=tf.float32, name="features")
        # features = self.speech_featurizer(signal)
        outputs = base_model(features=features, streaming=True)
    else:
        features = tf.keras.Input(shape=(None, feature_dim, channel_dim),
                                  dtype=tf.float32, name="features")
        # features = self.speech_featurizer(signal)
        outputs = base_model(features=features, streaming=False)

    # Fully connected layer
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(units=num_classes, activation="linear", dtype=tf.float32,
                              use_bias=True), name="fully_connected")(outputs)

    model = tf.keras.Model(inputs=features, outputs=outputs, name=name)
    return model, base_model.optimizer


def create_ctc_train_model(ctc_model, num_classes, name="ctc_train_model"):
    input_length = tf.keras.Input(shape=(), dtype=tf.int32, name="input_length")
    label_length = tf.keras.Input(shape=(), dtype=tf.int32, name="label_length")
    label = tf.keras.Input(shape=(None,), dtype=tf.int32, name="label")

    ctc_loss = tf.keras.layers.Lambda(
        ctc_loss_keras, output_shape=(1,), arguments={"num_classes": num_classes},
        name="ctc_loss")([ctc_model.outputs[0], input_length, label, label_length])

    return tf.keras.Model(inputs=(ctc_model.inputs, input_length, label, label_length),
                          outputs=ctc_loss, name=name)


@tf.function
def ctc_loss(y_true, y_pred, input_length, label_length, num_classes):
    loss = tf.nn.ctc_loss(
        labels=tf.cast(y_true, tf.int32),
        logit_length=input_length,
        logits=y_pred,
        label_length=label_length,
        logits_time_major=False,
        blank_index=num_classes - 1
    )
    return tf.reduce_mean(loss)


@tf.function
def ctc_loss_keras(layer, **kwargs):
    num_classes = kwargs["num_classes"]
    y_pred, input_length, y_true, label_length = layer
    loss = tf.nn.ctc_loss(
        labels=y_true,
        logit_length=input_length,
        logits=y_pred,
        label_length=label_length,
        logits_time_major=False,
        blank_index=num_classes - 1
    )
    return tf.reduce_mean(loss)
