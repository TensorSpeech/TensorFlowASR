from typing import List

import tensorflow as tf


@tf.keras.utils.register_keras_serializable("tensorflow_asr.optimizers.regularizers")
class TimeDependentGaussianGradientNoise(tf.keras.regularizers.Regularizer):
    """
    Reference: https://openreview.net/pdf/ZY9xxQDMMu5Pk8ELfEz4.pdf
    """

    def __init__(
        self,
        mean: float = 0.0,
        eta: float = 1.0,  # {0.01, 0.3, 1.0}
        gamma: float = 0.55,
    ):
        self.mean = mean
        self.eta = eta
        self.gamma = gamma
        super().__init__()

    def noise(self, step: tf.Tensor, gradient: tf.Tensor):
        sigma_squared = self.eta / ((1 + tf.cast(step, dtype=gradient.dtype)) ** self.gamma)
        return tf.random.normal(mean=self.mean, stddev=tf.math.sqrt(sigma_squared), shape=tf.shape(gradient), dtype=gradient.dtype)

    def __call__(self, step: tf.Tensor, gradients: List[tf.Tensor]):
        """
        Apply gaussian noise with time dependent to gradients

        Parameters
        ----------
        step : tf.Tensor
            Training step
        gradients : List[tf.Tensor]
            Gradients calculated from optimizer

        Returns
        -------
        List[tf.Tensor]
            Noise added gradients
        """
        return list(tf.add(gradient, self.noise(step, gradient=gradient)) for gradient in gradients)

    def get_config(self):
        return {
            "mean": self.mean,
            "eta": self.eta,
            "gamma": self.gamma,
        }
