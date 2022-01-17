import tensorflow as tf

from tensorflow_asr.utils import layer_util
from tensorflow_asr.models.layers.embedding import Embedding


class TransducerPrediction(tf.keras.Model):
    def __init__(
        self,
        vocabulary_size: int,
        embed_dim: int,
        embed_dropout: float = 0,
        num_rnns: int = 1,
        rnn_units: int = 512,
        rnn_type: str = "lstm",
        rnn_implementation: int = 2,
        layer_norm: bool = True,
        projection_units: int = 0,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="transducer_prediction",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.embed = Embedding(vocabulary_size, embed_dim, regularizer=kernel_regularizer, name=f"{name}_embedding")
        self.do = tf.keras.layers.Dropout(embed_dropout, name=f"{name}_dropout")
        # Initialize rnn layers
        RNN = layer_util.get_rnn(rnn_type)
        self.rnns = []
        for i in range(num_rnns):
            rnn = RNN(
                units=rnn_units,
                return_sequences=True,
                name=f"{name}_{rnn_type}_{i}",
                return_state=True,
                implementation=rnn_implementation,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
            if layer_norm:
                ln = tf.keras.layers.LayerNormalization(name=f"{name}_ln_{i}")
            else:
                ln = None
            if projection_units > 0:
                projection = tf.keras.layers.Dense(
                    projection_units,
                    name=f"{name}_projection_{i}",
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                )
            else:
                projection = None
            self.rnns.append({"rnn": rnn, "ln": ln, "projection": projection})

    def get_initial_state(self, batch_size: tf.Tensor = tf.constant(1)):
        """Get zeros states

        Returns:
            tf.Tensor: states having shape [num_rnns, 1 or 2, B, P]
        """
        states = []
        for rnn in self.rnns:
            states.append(tf.stack(rnn["rnn"].get_initial_state(tf.zeros([batch_size, 1, 1], dtype=tf.float32)), axis=0))
        return tf.stack(states, axis=0)

    def call(self, inputs, training=False, **kwargs):
        # inputs has shape [B, U]
        # use tf.gather_nd instead of tf.gather for tflite conversion
        outputs, prediction_length = inputs
        outputs = self.embed(outputs, training=training)
        outputs = self.do(outputs, training=training)
        for rnn in self.rnns:
            mask = tf.sequence_mask(prediction_length, maxlen=tf.shape(outputs)[1])
            outputs = rnn["rnn"](outputs, training=training, mask=mask)
            outputs = outputs[0]
            if rnn["ln"] is not None:
                outputs = rnn["ln"](outputs, training=training)
            if rnn["projection"] is not None:
                outputs = rnn["projection"](outputs, training=training)
        return outputs

    def recognize(self, inputs, states, tflite: bool = False):
        """Recognize function for prediction network

        Args:
            inputs (tf.Tensor): shape [1, 1]
            states (tf.Tensor): shape [num_lstms, 2, B, P]

        Returns:
            tf.Tensor: outputs with shape [1, 1, P]
            tf.Tensor: new states with shape [num_lstms, 2, 1, P]
        """
        if tflite:
            outputs = self.embed.recognize_tflite(inputs)
        else:
            outputs = self.embed(inputs, training=False)
        outputs = self.do(outputs, training=False)
        new_states = []
        for i, rnn in enumerate(self.rnns):
            outputs = rnn["rnn"](outputs, training=False, initial_state=tf.unstack(states[i], axis=0))
            new_states.append(tf.stack(outputs[1:]))
            outputs = outputs[0]
            if rnn["ln"] is not None:
                outputs = rnn["ln"](outputs, training=False)
            if rnn["projection"] is not None:
                outputs = rnn["projection"](outputs, training=False)
        return outputs, tf.stack(new_states, axis=0)

    def get_config(self):
        conf = self.embed.get_config()
        conf.update(self.do.get_config())
        for rnn in self.rnns:
            conf.update(rnn["rnn"].get_config())
            if rnn["ln"] is not None:
                conf.update(rnn["ln"].get_config())
            if rnn["projection"] is not None:
                conf.update(rnn["projection"].get_config())
        return conf
