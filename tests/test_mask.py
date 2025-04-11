import tensorflow as tf

from tensorflow_asr.models.layers.multihead_attention import compute_streaming_mask


def test_mha_streaming_mask():
    mask = compute_streaming_mask(2, 2, tf.zeros([5, 8, 8]))
    print(mask)
    assert tf.reduce_all(
        tf.equal(
            mask,
            tf.constant(
                [
                    [
                        [True, True, False, False, False, False, False, False],
                        [True, True, False, False, False, False, False, False],
                        [True, True, True, True, False, False, False, False],
                        [True, True, True, True, False, False, False, False],
                        [False, False, True, True, True, True, False, False],
                        [False, False, True, True, True, True, False, False],
                        [False, False, False, False, True, True, True, True],
                        [False, False, False, False, True, True, True, True],
                    ]
                ]
            ),
        )
    ).numpy()

    mask = compute_streaming_mask(3, 3, tf.zeros([5, 14, 14]))
    print(mask)
    assert tf.reduce_all(
        tf.equal(
            mask,
            tf.constant(
                [
                    [
                        [True, True, True, False, False, False, False, False, False, False, False, False, False, False],
                        [True, True, True, False, False, False, False, False, False, False, False, False, False, False],
                        [True, True, True, False, False, False, False, False, False, False, False, False, False, False],
                        [True, True, True, True, True, True, False, False, False, False, False, False, False, False],
                        [True, True, True, True, True, True, False, False, False, False, False, False, False, False],
                        [True, True, True, True, True, True, False, False, False, False, False, False, False, False],
                        [False, False, False, True, True, True, True, True, True, False, False, False, False, False],
                        [False, False, False, True, True, True, True, True, True, False, False, False, False, False],
                        [False, False, False, True, True, True, True, True, True, False, False, False, False, False],
                        [False, False, False, False, False, False, True, True, True, True, True, True, False, False],
                        [False, False, False, False, False, False, True, True, True, True, True, True, False, False],
                        [False, False, False, False, False, False, True, True, True, True, True, True, False, False],
                        [False, False, False, False, False, False, False, False, False, True, True, True, True, True],
                        [False, False, False, False, False, False, False, False, False, True, True, True, True, True],
                    ]
                ]
            ),
        )
    ).numpy()
