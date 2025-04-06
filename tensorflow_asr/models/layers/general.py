import keras
from keras.src import activations, backend

from tensorflow_asr.utils import math_util


class Dropout(keras.layers.Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape, seed, **kwargs)
        self.built = False


class Identity(keras.layers.Identity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.built = False


class Activation(keras.layers.Activation):
    def __init__(self, activation, **kwargs):
        super().__init__(activation, **kwargs)
        self.built = False


class Softmax(keras.layers.Softmax):
    """
    Softmax activation layer with better numerical stability to avoid Inf or NaN
    """

    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = math_util.masked_fill(
                inputs,
                mask=mask,
                value=math_util.large_compatible_negative_number(self.dtype),
            )
        if isinstance(self.axis, (tuple, list)):
            if len(self.axis) > 1:
                return backend.numpy.exp(inputs - backend.math.logsumexp(inputs, axis=self.axis, keepdims=True))
            return activations.softmax(inputs, axis=self.axis[0])
        return activations.softmax(inputs, axis=self.axis)
