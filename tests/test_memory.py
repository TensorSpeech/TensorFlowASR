"""
Transformer-XL segment-level recurrence (http://arxiv.org/abs/1901.02860).

The centrepiece is `test_streamed_segment_matches_full_sequence`: with the memory length equal
to the segment length, decoding the second segment with the first in memory must reproduce the
corresponding rows of a single full-length forward pass *exactly*. That is the property the
mechanism exists to provide, and it only holds if all of the following are right at once --
memory prepended to keys/values but not queries, the memory holding pre-projection hidden
states, the relative positional encoding spanning `memory_length + length`, and the attention
mask covering the memory. A shape-only test passes with any of those wrong.

Before the fix this path could not run at all: `MultiHeadAttention.build` passed a `batch_size`
kwarg that `Memory.__init__` does not accept, so every `memory_length` model raised at build.
`test_encoder_with_memory_builds_and_runs` pins that.
"""

import numpy as np
import pytest

from tensorflow_asr import tf
from tensorflow_asr.models.encoders.transformer import TransformerBlock, TransformerEncoder
from tensorflow_asr.models.layers.memory import Memory
from tensorflow_asr.models.layers.positional_encoding import RelativeSinusoidalPositionalEncoding

BATCH, LENGTH, DMODEL, NUM_HEADS, HEAD_SIZE = 1, 8, 8, 2, 4


def subsampling_config():
    # fresh each call: TransformerEncoder pops "type" out of the dict it is given
    return {
        "type": "conv2d",
        "filters": [8, 8],
        "kernels": [3, 3],
        "strides": [2, 2],
        "paddings": ["causal", "causal"],
        "norms": ["batch", "batch"],
        "activations": ["relu", "relu"],
    }


def relative_encoding(inputs, memory_length):
    layer = RelativeSinusoidalPositionalEncoding(interleave=True, memory_length=memory_length, causal=False)
    _, encoding = layer((inputs, tf.constant([inputs.shape[1]] * inputs.shape[0], tf.int32)), training=False)
    return encoding


# --------------------------------------------------------------------------- Memory layer


def test_memory_initial_state_is_zero_and_fully_masked():
    """An empty memory must be masked off, otherwise the first segment attends to zeros."""
    from keras.src import backend

    memory = Memory(memory_length=4, dmodel=DMODEL, name="m")
    state = memory.get_initial_state(BATCH)

    assert tuple(state.shape) == (BATCH, 4, DMODEL)
    assert np.all(state.numpy() == 0)
    assert not np.any(backend.get_keras_mask(state).numpy()), "initial memory must be masked out"


def test_memory_keeps_the_most_recent_frames():
    """New memory is the tail of [memory; inputs] -- oldest dropped, newest appended."""
    memory_length = 4
    memory = Memory(memory_length=memory_length, dmodel=1, name="m")
    previous = memory.get_initial_state(1)
    inputs = tf.reshape(tf.range(6, dtype=tf.float32), [1, 6, 1])

    extended, updated = memory(inputs, memories=previous, training=False)

    assert tuple(extended.shape) == (1, memory_length + 6, 1), "inputs must be prepended with the memory"
    # the last `memory_length` frames of the concatenation are the newest inputs
    assert np.array_equal(np.squeeze(updated.numpy()), [2.0, 3.0, 4.0, 5.0])


def test_memory_is_detached_from_the_graph_while_training():
    """`SG(.)` in eq. (3): no gradient flows into the cached segment."""
    memory = Memory(memory_length=2, dmodel=1, name="m")
    previous = tf.Variable([[[1.0], [2.0]]])
    inputs = tf.ones([1, 2, 1])

    with tf.GradientTape() as tape:
        extended, _ = memory(inputs, memories=previous, training=True)
        loss = tf.reduce_sum(extended)

    assert tape.gradient(loss, previous) is None, "memory must be stop-gradient during training"


# --------------------------------------------------------------------------- attention wiring


def test_memory_extends_keys_but_not_queries():
    """Queries stay on the current segment; only keys/values grow, so the output length holds."""
    block = TransformerBlock(dmodel=DMODEL, dff=16, num_heads=NUM_HEADS, head_size=HEAD_SIZE, memory_length=LENGTH, name="b")
    inputs = tf.random.normal([BATCH, LENGTH, DMODEL])
    block([inputs, relative_encoding(inputs, LENGTH)], training=False)

    outputs, states = block(
        [inputs, relative_encoding(inputs, LENGTH)],
        training=False,
        initial_state=block.get_initial_state(BATCH),
        return_states=True,
    )

    assert tuple(outputs.shape) == (BATCH, LENGTH, DMODEL), "memory must not add output frames"
    for name in ("key", "value"):
        assert tuple(states[name].shape) == (BATCH, LENGTH, DMODEL), f"{name} memory holds pre-projection hidden states"


def test_streamed_segment_matches_full_sequence():
    """
    The defining property of segment-level recurrence.

    With `memory_length == LENGTH` the memory after segment one is exactly segment one, so the
    second segment attends over the identical key set a full-length pass would use. The two
    must therefore agree to numerical noise.
    """
    memory_length = LENGTH
    block = TransformerBlock(dmodel=DMODEL, dff=16, num_heads=NUM_HEADS, head_size=HEAD_SIZE, memory_length=memory_length, name="b")
    first, second = tf.random.normal([BATCH, LENGTH, DMODEL]), tf.random.normal([BATCH, LENGTH, DMODEL])
    whole = tf.concat([first, second], axis=1)
    block([whole, relative_encoding(whole, 0)], training=False)

    full = block([whole, relative_encoding(whole, 0)], training=False, use_auto_mask=False)[0]

    _, states = block(
        [first, relative_encoding(first, memory_length)],
        training=False,
        use_auto_mask=False,
        initial_state=block.get_initial_state(BATCH),
        return_states=True,
    )
    streamed, _ = block(
        [second, relative_encoding(second, memory_length)],
        training=False,
        use_auto_mask=False,
        initial_state=states,
        return_states=True,
    )

    assert np.allclose(states["key"].numpy(), first.numpy(), atol=1e-5), "memory should be segment one verbatim"
    assert np.allclose(streamed.numpy(), full[:, LENGTH:, :].numpy(), atol=1e-4), "streamed segment diverged from the full pass"


def test_memory_changes_the_output():
    """Guards the reverse failure: silently dropping the memory would also 'pass' shape checks."""
    block = TransformerBlock(dmodel=DMODEL, dff=16, num_heads=NUM_HEADS, head_size=HEAD_SIZE, memory_length=LENGTH, name="b")
    first, second = tf.random.normal([BATCH, LENGTH, DMODEL]), tf.random.normal([BATCH, LENGTH, DMODEL])
    encoding = relative_encoding(second, LENGTH)
    block([first, encoding], training=False)

    empty = block.get_initial_state(BATCH)
    _, filled = block([first, encoding], training=False, use_auto_mask=False, initial_state=empty, return_states=True)

    without, _ = block([second, encoding], training=False, use_auto_mask=False, initial_state=empty, return_states=True)
    with_memory, _ = block([second, encoding], training=False, use_auto_mask=False, initial_state=filled, return_states=True)

    assert not np.allclose(without.numpy(), with_memory.numpy()), "a populated memory had no effect on the output"


@pytest.mark.parametrize("use_causal_mask", [False, True])
def test_masked_attention_covers_the_memory(use_causal_mask):
    """
    A causal or streaming mask is built against the unextended value, so it must be widened.

    Left alone it stays `LENGTH` wide while the scores become `memory_length + LENGTH` wide.
    Keras builds the causal mask from `row >= col`, which would hide the memory entirely --
    prepending the memory's own validity instead gives the `row + M >= col` the paper needs.
    """
    block = TransformerBlock(dmodel=DMODEL, dff=16, num_heads=NUM_HEADS, head_size=HEAD_SIZE, memory_length=LENGTH, name="b")
    inputs = tf.random.normal([BATCH, LENGTH, DMODEL])
    encoding = relative_encoding(inputs, LENGTH)
    block([inputs, encoding], training=False)

    outputs, _ = block(
        [inputs, encoding],
        training=False,
        use_auto_mask=True,
        use_causal_mask=use_causal_mask,
        initial_state=block.get_initial_state(BATCH),
        return_states=True,
    )

    assert tuple(outputs.shape) == (BATCH, LENGTH, DMODEL)
    assert bool(tf.reduce_all(tf.math.is_finite(outputs))), "masking produced non-finite values"


# --------------------------------------------------------------------------- encoder level


def test_encoder_with_memory_builds_and_runs():
    """Regression: `Memory` used to be constructed with a `batch_size` kwarg it never accepted."""
    memory_length = 6
    encoder = TransformerEncoder(
        subsampling=subsampling_config(),
        num_blocks=2,
        dmodel=DMODEL,
        dff=16,
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        memory_length=memory_length,
    )
    features, features_length = tf.random.normal([1, 40, 40, 1]), tf.constant([40], tf.int32)
    encoder((features, features_length), training=False)  # build before asking for state

    initial_state = encoder.get_initial_state(1)
    assert len(initial_state) == 2, "one memory per block"

    outputs, outputs_length, states = encoder.call_next(features, features_length, initial_state)
    tensors = tf.nest.flatten(states)

    assert tuple(outputs.shape)[0] == 1 and int(outputs_length[0]) == outputs.shape[1]
    assert tensors, "call_next returned no states despite memory_length being set"
    assert all(tuple(tensor.shape) == (1, memory_length, DMODEL) for tensor in tensors)


def test_encoder_memory_rolls_forward_across_calls():
    memory_length = 6
    encoder = TransformerEncoder(
        subsampling=subsampling_config(),
        num_blocks=1,
        dmodel=DMODEL,
        dff=16,
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        memory_length=memory_length,
    )
    features, features_length = tf.random.normal([1, 40, 40, 1]), tf.constant([40], tf.int32)
    encoder((features, features_length), training=False)

    _, _, first = encoder.call_next(features, features_length, encoder.get_initial_state(1))
    _, _, second = encoder.call_next(tf.random.normal([1, 40, 40, 1]), features_length, first)

    before, after = tf.nest.flatten(first)[0].numpy(), tf.nest.flatten(second)[0].numpy()
    assert not np.allclose(before, after), "memory did not advance between segments"


def test_encoder_without_memory_is_unchanged():
    """`memory_length=None` must keep returning a 3-tuple with no states."""
    encoder = TransformerEncoder(subsampling=subsampling_config(), num_blocks=1, dmodel=DMODEL, dff=16, num_heads=NUM_HEADS, head_size=HEAD_SIZE)
    features, features_length = tf.random.normal([1, 40, 40, 1]), tf.constant([40], tf.int32)

    outputs, outputs_length, states = encoder.call_next(features, features_length, None)

    assert states is None
    assert int(outputs_length[0]) == outputs.shape[1]
