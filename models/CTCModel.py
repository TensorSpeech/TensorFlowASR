from __future__ import absolute_import

import tensorflow as tf
from utils.Utils import wer, cer, mask_nan
from utils.Schedules import BoundExponentialDecay


def ctc_lambda_func(args):
  y_pred, input_length, labels, label_length = args
  y_pred = tf.math.log(
    tf.transpose(y_pred, perm=[1, 0, 2]) +
    tf.keras.backend.epsilon())
  loss = tf.nn.ctc_loss(
    labels=tf.keras.backend.ctc_label_dense_to_sparse(
      tf.cast(labels, tf.int32),
      tf.cast(label_length, tf.int32)),
    logits=tf.cast(y_pred, tf.float32),
    label_length=None,
    logit_length=tf.cast(input_length, tf.int32),
    blank_index=0)
  loss = mask_nan(loss)
  return tf.cast(loss, tf.float64)


def decode_lambda_func(args, **arguments):
  y_pred, input_length = args
  decoder = arguments["decoder"]
  return decoder.decode(probs=y_pred,
                        input_length=tf.squeeze(input_length))


def test_lambda_func(args, **arguments):
  y_pred, input_length, labels = args
  decoder = arguments["decoder"]
  predictions = decoder.decode(probs=y_pred,
                               input_length=tf.squeeze(input_length))
  string_labels = decoder.convert_to_string(labels)
  predictions = tf.expand_dims(predictions, 1)
  string_labels = tf.expand_dims(string_labels, 1)
  outputs = tf.concat([predictions, string_labels], axis=-1)

  def cal_each_er(elem):
    pred = elem[0].numpy().decode("utf-8")
    target = elem[1].numpy().decode("utf-8")
    print(pred)
    cal_wer = wer(decode=pred, target=target)
    cal_cer = cer(decode=pred, target=target)
    return tf.convert_to_tensor([cal_wer, cal_cer])

  return tf.map_fn(cal_each_er, outputs, dtype=tf.float64)


def create_ctc_model(num_classes, num_feature_bins,
                     learning_rate, base_model, decoder,
                     mode="train", min_lr=0.0):
  if mode == "infer_streaming":
    bsize = 1
  else:
    bsize = None
  # Convolution layers
  features = tf.keras.layers.Input(shape=(None, num_feature_bins, 1),
                                   batch_size=bsize,
                                   dtype=tf.float64,
                                   name="features")
  input_length = tf.keras.layers.Input(shape=(),
                                       dtype=tf.int64,
                                       batch_size=bsize,
                                       name="input_length")
  labels = tf.keras.layers.Input(shape=(None,),
                                 dtype=tf.int64,
                                 batch_size=bsize,
                                 name="labels")
  label_length = tf.keras.layers.Input(shape=(),
                                       dtype=tf.int64,
                                       batch_size=bsize,
                                       name="label_length")

  if mode == 'infer_streaming':
    outputs = base_model(features=features, streaming=True)
  else:
    outputs = base_model(features=features, streaming=False)

  # Fully connected layer
  outputs = tf.keras.layers.Dense(
    units=num_classes,
    activation=tf.keras.activations.softmax,
    use_bias=True)(outputs)

  if mode == "train":
    # Lambda layer for computing loss function
    loss_out = tf.keras.layers.Lambda(
      ctc_lambda_func,
      output_shape=(),
      name="ctc_loss")([outputs, input_length,
                        labels, label_length])

    train_model = tf.keras.Model(inputs={
      "features"    : features,
      "input_length": input_length,
      "labels"      : labels,
      "label_length": label_length
    }, outputs=loss_out)

    # y_true is None because of dummy label and loss is calculated
    # in the layer lambda
    train_model.compile(
      optimizer=base_model.optimizer(
        learning_rate=BoundExponentialDecay(
          min_lr=min_lr,
          initial_learning_rate=learning_rate,
          decay_steps=5000,
          decay_rate=0.9,
          staircase=True)),
      loss={"ctc_loss": lambda y_true, y_pred: tf.reduce_mean(y_pred)}
    )
    return train_model
  if mode in ["infer", "infer_single", "infer_streaming"]:
    # Lambda layer for decoding to text
    decode_out = tf.keras.layers.Lambda(
      decode_lambda_func,
      output_shape=(None,),
      name="ctc_decoder",
      arguments={"decoder": decoder},
      dynamic=True)([outputs, input_length])

    infer_model = tf.keras.Model(inputs={
      "features"    : features,
      "input_length": input_length
    }, outputs=decode_out)

    infer_model.compile(
      optimizer=base_model.optimizer(lr=learning_rate),
      loss={"ctc_decoder": lambda y_true, y_pred: y_pred}
    )
    return infer_model
  if mode == "test":
    # Lambda layer for analysis
    test_out = tf.keras.layers.Lambda(
      test_lambda_func,
      output_shape=(None,),
      name="ctc_test",
      arguments={"decoder": decoder},
      dynamic=True)([outputs, input_length, labels])

    test_model = tf.keras.Model(inputs={
      "features"    : features,
      "input_length": input_length,
      "labels"      : labels
    }, outputs=test_out)

    test_model.compile(
      optimizer=base_model.optimizer(lr=learning_rate),
      loss={"ctc_test": lambda y_true, y_pred: y_pred}
    )
    return test_model
  raise ValueError("mode must be either 'train', 'infer' or 'test'")
