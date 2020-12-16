from typing import List
import tensorflow as tf
from .transducer import Transducer
from ..utils.utils import merge_two_last_dims

L2 = tf.keras.regularizers.l2(1e-6)


def get_activation(activation: str = "silu"):
    activation = activation.lower()
    if activation in ["silu", "swish"]: return tf.nn.silu
    elif activation == "relu": return tf.nn.relu
    elif activation == "linear": return tf.keras.activations.linear
    else: raise ValueError("activation must be either 'silu', 'swish', 'relu' or 'linear'")


class Reshape(tf.keras.layers.Layer):
    def call(self, inputs): return merge_two_last_dims(inputs)


class ResConvModule(tf.keras.layers.Layer):
    def __init__(self,
                 filters: int = 256,
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 **kwargs):
        super(ResConvModule, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(
            filters=filters, kernel_size=1, strides=1, padding="same",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, name=f"{self.name}_conv"
        )
        self.bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn")

    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv(inputs, training=training)
        outputs = self.bn(outputs, training=training)
        return outputs


class ConvModule(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size: int = 3,
                 strides: int = 1,
                 filters: int = 256,
                 activation: str = "silu",
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 **kwargs):
        super(ConvModule, self).__init__(**kwargs)
        self.conv = tf.keras.layers.SeparableConv1D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
            depthwise_regularizer=kernel_regularizer, pointwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, name=f"{self.name}_conv"
        )
        self.bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn")
        self.activation = get_activation(activation)

    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv(inputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.activation(outputs)
        return outputs


class SEModule(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size: int = 3,
                 strides: int = 1,
                 filters: int = 256,
                 activation: str = "silu",
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 **kwargs):
        super(SEModule, self).__init__(**kwargs)
        self.conv = ConvModule(
            kernel_size=kernel_size, strides=strides,
            filters=filters, activation=activation,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            name=f"{self.name}_conv_module"
        )
        self.activation = get_activation(activation)
        self.fc1 = tf.keras.layers.Dense(filters // 8, name=f"{self.name}_fc1")
        self.fc2 = tf.keras.layers.Dense(filters, name=f"{self.name}_fc2")

    def call(self, inputs, training=False, **kwargs):
        features, input_length = inputs
        outputs = self.conv(features, training=training)

        se = tf.reduce_sum(outputs, axis=1) / tf.expand_dims(tf.cast(input_length, dtype=outputs.dtype), axis=1)
        se = self.fc1(se, training=training)
        se = self.activation(se)
        se = self.fc2(se, training=training)
        se = self.activation(se)
        se = tf.nn.sigmoid(se)
        se = tf.expand_dims(se, axis=1)

        outputs = tf.multiply(outputs, se)
        return outputs


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 nlayers: int = 3,
                 kernel_size: int = 3,
                 filters: int = 256,
                 strides: int = 1,
                 residual: bool = True,
                 activation: str = 'silu',
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        self.dmodel = filters
        self.time_reduction_factor = strides

        self.convs = []
        for i in range(nlayers - 1):
            self.convs.append(
                ConvModule(
                    kernel_size=kernel_size, strides=1,
                    filters=filters, activation=activation,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                    name=f"{self.name}_conv_module_{i}"
                )
            )

        self.last_conv = ConvModule(
            kernel_size=kernel_size, strides=strides,
            filters=filters, activation=activation,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            name=f"{self.name}_conv_module_{nlayers - 1}"
        )

        self.se = SEModule(
            kernel_size=kernel_size, strides=1, filters=filters, activation=activation,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            name=f"{self.name}_se"
        )

        self.residual = None
        if residual:
            self.residual = ResConvModule(
                filters=filters, kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer, name=f"{self.name}_residual"
            )

        self.activation = get_activation(activation)

    def call(self, inputs, training=False, **kwargs):
        features, input_length = inputs
        outputs = features
        for conv in self.convs:
            outputs = conv(outputs, training=training)
        outputs = self.last_conv(outputs, training=training)
        input_length = tf.math.ceil(input_length / self.last_conv.strides[0])
        outputs = self.se([outputs, input_length], training=training)
        if self.residual is not None:
            res = self.residual(features, training=training)
            outputs = tf.add(outputs, res)
        outputs = self.activation(outputs)
        return outputs, input_length


class ContextNetEncoder(tf.keras.Model):
    def __init__(self,
                 blocks: List[dict] = [],
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 **kwargs):
        super(ContextNetEncoder, self).__init__(**kwargs)

        self.reshape = Reshape(name=f"{self.name}_reshape")

        self.blocks = []
        for config, i in enumerate(blocks):
            self.blocks.append(
                ConvBlock(**config, kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer, name=f"{self.name}_block_{i}")
            )

    def call(self, inputs, training=False, **kwargs):
        outputs, input_length = inputs
        outputs = self.reshape(outputs)
        for block in self.blocks:
            outputs, input_length = block([outputs, input_length], training=training)
        return outputs


class ContextNet(Transducer):
    def __init__(self,
                 vocabulary_size: int,
                 encoder_blocks: List[dict],
                 prediction_embed_dim: int = 512,
                 prediction_embed_dropout: int = 0,
                 prediction_num_rnns: int = 1,
                 prediction_rnn_units: int = 320,
                 prediction_rnn_type: str = "lstm",
                 prediction_rnn_implementation: int = 2,
                 prediction_layer_norm: bool = True,
                 prediction_projection_units: int = 0,
                 joint_dim: int = 1024,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name: str = "contextnet",
                 **kwargs):
        super(ContextNet, self).__init__(
            encoder=ContextNetEncoder(
                blocks=encoder_blocks,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{name}_encoder"
            ),
            vocabulary_size=vocabulary_size,
            embed_dim=prediction_embed_dim,
            embed_dropout=prediction_embed_dropout,
            num_rnns=prediction_num_rnns,
            rnn_units=prediction_rnn_units,
            rnn_type=prediction_rnn_type,
            rnn_implementation=prediction_rnn_implementation,
            layer_norm=prediction_layer_norm,
            projection_units=prediction_projection_units,
            joint_dim=joint_dim,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=name, **kwargs
        )
        self.dmodel = self.encoder.blocks[-1].dmodel
        self.time_reduction_factor = 1
        for block in self.encoder.blocks:
            self.time_reduction_factor += block.time_reduction_factor

    def call(self, inputs, training=False, **kwargs):
        """
        Transducer Model call function
        Args:
            features: audio features in shape [B, T, F, C]
            input_length: shape [B]
            predicted: predicted sequence of character ids, in shape [B, U]
            training: python boolean
            **kwargs: sth else

        Returns:
            `logits` with shape [B, T, U, vocab]
        """
        features, input_length, predicted, label_length = inputs
        enc = self.encoder([features, input_length], training=training, **kwargs)
        pred = self.predict_net([predicted, label_length], training=training, **kwargs)
        outputs = self.joint_net([enc, pred], training=training, **kwargs)
        return outputs

    def encoder_inference(self, features):
        """Infer function for encoder (or encoders)

        Args:
            features (tf.Tensor): features with shape [T, F, C]

        Returns:
            tf.Tensor: output of encoders with shape [T, E]
        """
        with tf.name_scope(f"{self.name}_encoder"):
            input_length = tf.expand_dims(tf.shape(features)[0], axis=0)
            outputs = tf.expand_dims(features, axis=0)
            outputs = self.encoder([outputs, input_length], training=False)
            return tf.squeeze(outputs, axis=0)
