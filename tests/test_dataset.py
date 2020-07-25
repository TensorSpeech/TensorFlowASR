import tensorflow as tf


def func():
    for i in range(100):
        yield ({
                   "name": i,
                   "value": i + 1
               }, {
                   "name": i,
                   "value": None
               })


dataset = tf.data.Dataset.from_generator(
    func,
    output_types=({
                      "name": tf.int32,
                      "value": tf.int32,
                  }, {
                      "name": tf.int32,
                      "value": tf.int32,
                  }
    ),
    output_shapes=({
                       "name": tf.TensorShape([]),
                       "value": tf.TensorShape([])
                   }, {
                       "name": tf.TensorShape([]),
                       "value": tf.TensorShape([None, None])
                   }
    )
)

print(dataset.element_spec)
