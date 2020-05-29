from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from models.aaconvds2.AAConv2D import AAConv2D
from models.deepspeech2.SequenceBatchNorm import SequenceBatchNorm

DEFAULT_CONV = {
    "kernels": ((11, 41), (11, 21), (11, 21)),
    "fout": (32, 32, 96),
    "Nh": 8,
    "k": 2,  # dk = fout * k
    "v": 0.25,  # dv = fout * v,
    "relative": False,
    "max_pool": (2, 2)
}

DEFAULT_RNN = {
    "rnn_layers": 3,
    "rnn_type": "gru",
    "rnn_units": 350,
    "rnn_activation": "tanh",
    "rnn_bidirectional": True,
    "rnn_rowconv": False,
    "rnn_rowconv_context": 2,
    "rnn_dropout": 0.2
}

DEFAULT_FC = {
    "fc_units": (1024,),
    "fc_dropout": 0.2
}


def append_default_keys_dict(default_dict, dest_dict):
    for key in default_dict.keys():
        if key not in dest_dict.keys():
            dest_dict[key] = default_dict[key]
    return dest_dict


class AAConvDeepSpeech2:
    def __init__(self, conv_conf=DEFAULT_CONV, rnn_conf=DEFAULT_RNN, fc_conf=DEFAULT_FC,
                 optimizer=tf.keras.optimizers.SGD(lr=0.0002, momentum=0.99, nesterov=True)):
        self.optimizer = optimizer
        self.conv_conf = append_default_keys_dict(DEFAULT_CONV, conv_conf)
        self.rnn_conf = append_default_keys_dict(DEFAULT_RNN, rnn_conf)
        self.fc_conf = append_default_keys_dict(DEFAULT_FC, fc_conf)
        assert len(conv_conf["fout"]) == len(conv_conf["kernels"])
        assert rnn_conf["rnn_type"] in ["lstm", "gru", "rnn"]
        assert rnn_conf["rnn_dropout"] >= 0.0

    @staticmethod
    def merge_filter_to_channel(x):
        f, c = x.get_shape().as_list()[2:]
        return tf.keras.layers.Reshape([-1, f * c])(x)

    def __call__(self, features, streaming=False):
        # CONV Layers
        layer = AAConv2D(fout=self.conv_conf["fout"][0], k=self.conv_conf["kernels"][0],
                         dk=self.conv_conf["fout"][0] * self.conv_conf["k"], dv=self.conv_conf["fout"][0] * self.conv_conf["v"],
                         Nh=self.conv_conf["Nh"], name="aaconv2d_0", relative=self.conv_conf["relative"])(features)
        layer = tf.keras.layers.BatchNormalization(name="aaconv2d_bn_0")(layer)
        layer = tf.keras.layers.ReLU(name="aaconv2d_relu_0")(layer)
        if self.conv_conf["max_pool"]:
            layer = tf.keras.layers.MaxPooling2D(pool_size=self.conv_conf["max_pool"], strides=(2, 2), padding="same")(layer)

        for i, fil in enumerate(self.conv_conf["fout"][1:], start=1):
            layer = AAConv2D(fout=fil, k=self.conv_conf["kernels"][i],
                             dk=fil * self.conv_conf["k"], dv=fil * self.conv_conf["v"],
                             Nh=self.conv_conf["Nh"], name="aaconv2d_0", relative=self.conv_conf["relative"])(features)
            layer = tf.keras.layers.BatchNormalization(name=f"aaconv2d_bn_{i}")(layer)
            layer = tf.keras.layers.ReLU(name=f"aaconv2d_relu_{i}")(layer)

        layer = self.merge_filter_to_channel(layer)

        if self.rnn_conf["rnn_type"] == "rnn":
            rnn = tf.keras.layers.SimpleRNN
        elif self.rnn_conf["rnn_type"] == "lstm":
            rnn = tf.keras.layers.LSTM
        else:
            rnn = tf.keras.layers.GRU

        # To time major
        if self.rnn_conf["rnn_bidirectional"]:
            layer = tf.transpose(layer, perm=[1, 0, 2])

        # RNN layers
        for i in range(self.rnn_conf["rnn_layers"]):
            if self.rnn_conf["rnn_bidirectional"]:
                layer = tf.keras.layers.Bidirectional(
                    rnn(self.rnn_conf["rnn_units"], activation=self.rnn_conf["rnn_activation"], time_major=True,
                        dropout=self.rnn_conf["rnn_dropout"], return_sequences=True, use_bias=True),
                    name=f"b{self.rnn_conf['rnn_type']}_{i}")(layer)
                layer = SequenceBatchNorm(time_major=True, name=f"sequence_wise_bn_{i}")(layer)
            else:
                layer = rnn(self.rnn_conf["rnn_units"], activation=self.rnn_conf["rnn_activation"],
                            dropout=self.rnn_conf["rnn_dropout"], return_sequences=True, use_bias=True,
                            name=f"{self.rnn_conf['rnn_type']}_{i}")(layer)
                layer = SequenceBatchNorm(time_major=False, name=f"sequence_wise_bn_{i}")(layer)

        # To batch major
        if self.rnn_conf["rnn_bidirectional"]:
            layer = tf.transpose(layer, perm=[1, 0, 2])

        # FC Layers
        if self.fc_conf["fc_units"]:
            assert self.fc_conf["fc_dropout"] >= 0.0

            for idx, units in enumerate(self.fc_conf["fc_units"]):
                layer = tf.keras.layers.Dense(units=units, activation=None,
                                              use_bias=True, name=f"hidden_fc_{idx}")(layer)
                layer = tf.keras.layers.BatchNormalization(name=f"hidden_fc_bn_{idx}")(layer)
                layer = tf.keras.layers.ReLU(name=f"hidden_fc_relu_{idx}")(layer)
                layer = tf.keras.layers.Dropout(self.fc_conf["fc_dropout"], name=f"hidden_fc_dropout_{idx}")(layer)

        return layer
