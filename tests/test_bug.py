import keras


class Model(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense = keras.layers.Dense(10)
        self.mha = keras.layers.MultiHeadAttention(10, 10, output_shape=(100,))

    def call(self, inputs):
        x = self.dense(inputs)
        return self.mha(x, x, x)


model = Model()
model(keras.Input(shape=(10, 10)))
model.summary()
