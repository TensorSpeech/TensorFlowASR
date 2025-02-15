import keras


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
