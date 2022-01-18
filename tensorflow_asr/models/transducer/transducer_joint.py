import tensorflow as tf


class TransducerJointReshape(tf.keras.layers.Layer):
    def __init__(self, axis: int = 1, name="transducer_joint_reshape", **kwargs):
        super().__init__(name=name, trainable=False, **kwargs)
        self.axis = axis

    def call(self, inputs, repeats=None, **kwargs):
        outputs = tf.expand_dims(inputs, axis=self.axis)
        return tf.repeat(outputs, repeats=repeats, axis=self.axis)

    def get_config(self):
        conf = super(TransducerJointReshape, self).get_config()
        conf.update({"axis": self.axis})
        return conf


class TransducerJoint(tf.keras.Model):
    def __init__(
        self,
        vocabulary_size: int,
        joint_dim: int = 1024,
        activation: str = "tanh",
        prejoint_linear: bool = True,
        postjoint_linear: bool = False,
        joint_mode: str = "add",
        kernel_regularizer=None,
        bias_regularizer=None,
        name="tranducer_joint",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        activation = activation.lower()
        if activation == "linear":
            self.activation = tf.keras.layers.Activation(tf.keras.activation.linear, name=f"{name}_linear")
        elif activation == "relu":
            self.activation = tf.keras.layers.Activation(tf.nn.relu, name=f"{name}_relu")
        elif activation == "tanh":
            self.activation = tf.keras.layers.Activation(tf.nn.tanh, name=f"{name}_tanh")
        else:
            raise ValueError("activation must be either 'linear', 'relu' or 'tanh'")

        self.prejoint_linear = prejoint_linear
        self.postjoint_linear = postjoint_linear

        if self.prejoint_linear:
            self.ffn_enc = tf.keras.layers.Dense(
                joint_dim, name=f"{name}_enc", kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer
            )
            self.ffn_pred = tf.keras.layers.Dense(
                joint_dim, use_bias=False, name=f"{name}_pred", kernel_regularizer=kernel_regularizer
            )

        self.enc_reshape = TransducerJointReshape(axis=2, name=f"{name}_enc_reshape")
        self.pred_reshape = TransducerJointReshape(axis=1, name=f"{name}_pred_reshape")

        if joint_mode == "add":
            self.joint = tf.keras.layers.Add(name=f"{name}_add")
        elif joint_mode == "concat":
            self.joint = tf.keras.layers.Concatenate(name=f"{name}_concat")
        else:
            raise ValueError("joint_mode must be either 'add' or 'concat'")

        if self.postjoint_linear:
            self.ffn = tf.keras.layers.Dense(
                joint_dim, name=f"{name}_ffn", kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer
            )

        self.ffn_out = tf.keras.layers.Dense(
            vocabulary_size, name=f"{name}_vocab", kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer
        )

    def call(self, inputs, training=False, **kwargs):
        # enc has shape [B, T, E]
        # pred has shape [B, U, P]
        enc_out, pred_out = inputs
        if self.prejoint_linear:
            enc_out = self.ffn_enc(enc_out, training=training)  # [B, T, E] => [B, T, V]
            pred_out = self.ffn_pred(pred_out, training=training)  # [B, U, P] => [B, U, V]
        enc_out = self.enc_reshape(enc_out, repeats=tf.shape(pred_out)[1])
        pred_out = self.pred_reshape(pred_out, repeats=tf.shape(enc_out)[1])
        outputs = self.joint([enc_out, pred_out], training=training)
        if self.postjoint_linear:
            outputs = self.ffn(outputs, training=training)
        outputs = self.activation(outputs, training=training)  # => [B, T, U, V]
        outputs = self.ffn_out(outputs, training=training)
        return outputs

    def get_config(self):
        conf = self.ffn_enc.get_config()
        conf.update(self.ffn_pred.get_config())
        conf.update(self.ffn_out.get_config())
        conf.update(self.activation.get_config())
        conf.update(self.joint.get_config())
        conf.update({"prejoint_linear": self.prejoint_linear, "postjoint_linear": self.postjoint_linear})
        return conf
