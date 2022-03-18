import tensorflow as tf


def test_dataset():
    a = [1, 2, 3, 4, 5, 6, 7]
    batch = 2
    ds = tf.data.Dataset.from_tensor_slices(a)
    ds = ds.cache()
    ds = ds.shuffle(3)
    ds = ds.repeat(3)
    ds = ds.batch(batch, drop_remainder=True)
    print(list(ds.as_numpy_iterator()))
