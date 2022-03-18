from typing import Optional

import tensorflow as tf


class Scale(tf.keras.layers.Layer):
    """Scales the input by a trainable scalar weight.
    This is useful for applying ReZero to layers, which improves convergence
    speed. This implements the paper:
    ReZero is All You Need: Fast Convergence at Large Depth.
    (https://arxiv.org/pdf/2003.04887.pdf).
    """

    def __init__(
        self,
        initializer: tf.keras.initializers.Initializer = "ones",
        regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        **kwargs,
    ):
        """Initializes a scale layer.
        Args:
          initializer: A `str` of initializer for the scalar weight.
          regularizer: A `tf.keras.regularizers.Regularizer` for the scalar weight.
          **kwargs: Additional keyword arguments to be passed to this layer.
        Returns:
          An `tf.Tensor` of which should have the same shape as input.
        """
        super().__init__(**kwargs)

        self._initializer = initializer
        self._regularizer = regularizer

        self._scale = self.add_weight(
            name="scale",
            shape=[],
            dtype=self.dtype,
            initializer=self._initializer,
            regularizer=self._regularizer,
            trainable=True,
        )

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "initializer": self._initializer,
            "regularizer": self._regularizer,
        }
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        """Calls the layer with the given inputs."""
        scale = tf.cast(self._scale, inputs.dtype)
        return scale * inputs
