from __future__ import absolute_import

import tensorflow as tf


def ctc_lambda_func(args):
    y_pred, input_length, labels, label_length = args
    return tf.keras.backend.ctc_batch_cost(y_true=labels, y_pred=y_pred,
                                           input_length=input_length, label_length=label_length)


def decode_lambda_func(args, **arguments):
    y_pred, input_length = args
    decoder = arguments["decoder"]
    return decoder.decode(probs=y_pred, input_length=tf.squeeze(input_length))


def test_lambda_func(args, **arguments):
    y_pred, input_length, labels = args
    decoder = arguments["decoder"]
    predictions = decoder.decode(probs=y_pred, input_length=tf.squeeze(input_length))
    return predictions, labels


class CTCModel:
    def __init__(self, num_classes, num_feature_bins, learning_rate, base_model, decoder):
        self.num_classes = num_classes
        self.num_feature_bins = num_feature_bins
        self.learning_rate = learning_rate
        self.base_model = base_model
        self.train_model, self.test_model, self.infer_model = self.__create(decoder)

    def __create(self, decoder):
        # Convolution layers
        features = tf.keras.layers.Input(shape=(None, self.num_feature_bins, 1), name="features")
        input_length = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="input_length")
        labels = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="labels")
        label_length = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="label_length")

        outputs = self.base_model(features=features)

        # Fully connected layer
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Dense(units=self.num_classes,
                                        activation=tf.keras.activations.softmax, use_bias=True)(outputs)

        # Lambda layer for computing loss function
        loss_out = tf.keras.layers.Lambda(ctc_lambda_func, output_shape=(1,), name="ctc_loss")(
            [outputs, input_length, labels, label_length])

        # Lambda layer for decoding to text
        decode_out = tf.keras.layers.Lambda(decode_lambda_func, output_shape=(None,), name="ctc_decoder",
                                            arguments={"decoder": decoder}, dynamic=True)([outputs, input_length])

        # Lambda layer for analysis
        test_out = tf.keras.layers.Lambda(test_lambda_func, output_shape=(None, None), name="ctc_test",
                                          arguments={"decoder": decoder}, dynamic=True)([outputs, input_length, labels])

        train_model = tf.keras.Model(inputs={
            "features": features,
            "input_length": input_length,
            "labels": labels,
            "label_length": label_length
        }, outputs=loss_out)

        # y_true is None because of dummy label and loss is calculated in the layer lambda
        train_model.compile(optimizer=self.base_model.optimizer(lr=self.learning_rate),
                            loss={"ctc_loss": lambda y_true, y_pred: y_pred})

        infer_model = tf.keras.Model(inputs={
            "features": features,
            "input_length": input_length
        }, outputs=decode_out)

        infer_model.compile(optimizer=self.base_model.optimizer(lr=self.learning_rate),
                            loss={"ctc_decoder": lambda y_true, y_pred: y_pred})

        test_model = tf.keras.Model(inputs={
            "features": features,
            "input_length": input_length,
            "labels": labels
        }, outputs=test_out)

        test_model.compile(optimizer=self.base_model.optimizer(lr=self.learning_rate),
                           loss={"ctc_test": lambda y_true, y_pred: y_pred})

        return train_model, test_model, infer_model
