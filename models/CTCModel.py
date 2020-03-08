from __future__ import absolute_import

import tensorflow as tf
from utils.Utils import wer, cer, mask_nan
from utils.Schedules import BoundExponentialDecay


def ctc_lambda_func(args):
  y_pred, input_length, labels, label_length = args
  label_length = tf.expand_dims(label_length, 1)
  input_length = tf.expand_dims(input_length, 1)
  loss = tf.keras.backend.ctc_batch_cost(
    y_pred=y_pred,
    input_length=input_length,
    y_true=labels,
    label_length=label_length)
  return mask_nan(loss)


def decode_lambda_func(args, **arguments):
  y_pred, input_length = args
  decoder = arguments["decoder"]
  return decoder.decode(probs=y_pred,
                        input_length=tf.squeeze(input_length,
                                                axis=-1))


def test_lambda_func(args, **arguments):
  y_pred, input_length, labels = args
  decoder = arguments["decoder"]
  predictions = decoder.decode(probs=y_pred,
                               input_length=tf.squeeze(input_length,
                                                       axis=-1))
  string_labels = decoder.convert_to_string(labels)
  predictions = tf.expand_dims(predictions, 1)
  string_labels = tf.expand_dims(string_labels, 1)
  outputs = tf.concat([predictions, string_labels], axis=-1)

  def cal_each_er(elem):
    pred = elem[0].numpy().decode("utf-8")
    target = elem[1].numpy().decode("utf-8")
    print(pred)
    print(target)
    cal_wer = wer(decode=pred, target=target)
    cal_cer = cer(decode=pred, target=target)
    return tf.convert_to_tensor([cal_wer, cal_cer])

  return tf.map_fn(cal_each_er, outputs, dtype=tf.int32)


def create_ctc_model(num_classes, num_feature_bins,
                     learning_rate, base_model, decoder,
                     mode="train", min_lr=0.0):
  bsize = 1 if mode == "infer_streaming" else None
  # Convolution layers
  features = tf.keras.layers.Input(shape=(None, num_feature_bins, 1),
                                   batch_size=bsize,
                                   dtype=tf.float32,
                                   name="features")
  input_length = tf.keras.layers.Input(shape=(),
                                       dtype=tf.int32,
                                       batch_size=bsize,
                                       name="input_length")

  if mode == 'infer_streaming':
    outputs = base_model(features=features, streaming=True)
  else:
    outputs = base_model(features=features, streaming=False)

  batch_size = tf.shape(outputs)[0]
  n_hidden = outputs.get_shape().as_list()[-1]
  # reshape from [B, T, A] --> [B*T, A].
  # Output shape: [n_steps * batch_size, n_hidden]
  outputs = tf.reshape(outputs, [-1, n_hidden])

  # Fully connected layer
  outputs = tf.keras.layers.Dense(
    units=num_classes,
    activation='softmax',
    name="fully_connected",
    use_bias=True)(outputs)

  outputs = tf.reshape(outputs,
                       [batch_size, -1, num_classes],
                       name="logits")

  if mode == "train":
    labels = tf.keras.layers.Input(shape=(None,),
                                   dtype=tf.int32,
                                   batch_size=bsize,
                                   name="labels")
    label_length = tf.keras.layers.Input(shape=(),
                                         dtype=tf.int32,
                                         batch_size=bsize,
                                         name="label_length")
    # Lambda layer for computing loss function
    loss_out = tf.keras.layers.Lambda(
      ctc_lambda_func,
      output_shape=(),
      name="ctc_loss")([outputs, input_length,
                        labels, label_length])

    train_model = tf.keras.Model(inputs={
      "features": features,
      "input_length": input_length,
      "labels": labels,
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
      loss={"ctc_loss": lambda y_true, y_pred: y_pred}
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
      "features": features,
      "input_length": input_length
    }, outputs=decode_out)

    return infer_model
  if mode == "test":
    labels = tf.keras.layers.Input(shape=(None,),
                                   dtype=tf.int32,
                                   batch_size=bsize,
                                   name="labels")
    # Lambda layer for analysis
    test_out = tf.keras.layers.Lambda(
      test_lambda_func,
      output_shape=(None,),
      name="ctc_test",
      arguments={"decoder": decoder},
      dynamic=True)([outputs, input_length, labels])

    test_model = tf.keras.Model(inputs={
      "features": features,
      "input_length": input_length,
      "labels": labels
    }, outputs=test_out)

    return test_model
  raise ValueError("mode must be either 'train', 'infer' or 'test'")
