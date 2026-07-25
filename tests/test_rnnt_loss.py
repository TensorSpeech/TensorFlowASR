"""
Tests for the pure-TensorFlow RNNT loss helper.

`compute_rnnt_loss_and_grad_helper` returns `-log P(labels | inputs)` marginalised over every
monotonic alignment, so it can be checked against a direct implementation of the forward
recursion. That reference is written straight from the definition here rather than reusing
anything from `tensorflow_asr`, which is what makes the comparison meaningful.

The previous version of this file imported the helper from `tensorflow_asr.losses.rnnt_loss`
(it now lives in `losses.impl.rnnt`), allocated a ~600 MB logits tensor and asserted nothing.
"""

import numpy as np
import pytest

from tensorflow_asr import tf
from tensorflow_asr.losses.impl.rnnt import compute_rnnt_loss_and_grad_helper

BLANK = 0


def reference_loss(logits, labels, nframes, nlabels, blank=BLANK):
    """
    Forward recursion over the RNNT lattice, in log space.

    From `(t, u)` a path either emits blank and advances the frame, or emits `labels[u]` and
    advances the label. Every path runs from `(0, 0)` to `(T - 1, U)` and then takes a final blank.
    """
    log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    alpha = np.full((nframes, nlabels + 1), -np.inf)
    alpha[0, 0] = 0.0
    for t in range(nframes):
        for u in range(nlabels + 1):
            if t == 0 and u == 0:
                continue
            paths = []
            if t > 0:
                paths.append(alpha[t - 1, u] + log_probs[t - 1, u, blank])
            if u > 0:
                paths.append(alpha[t, u - 1] + log_probs[t, u - 1, labels[u - 1]])
            alpha[t, u] = np.logaddexp.reduce(paths)
    return -(alpha[nframes - 1, nlabels] + log_probs[nframes - 1, nlabels, blank])


def random_batch(seed, batch_size, nframes, nlabels, vocab_size):
    rng = np.random.RandomState(seed)
    logits = (rng.randn(batch_size, nframes, nlabels + 1, vocab_size) * 1.5).astype(np.float32)
    labels = rng.randint(1, vocab_size, size=(batch_size, nlabels)).astype(np.int32)
    return logits, labels


def run_helper(logits, labels, nframes, nlabels):
    batch_size = logits.shape[0]
    return compute_rnnt_loss_and_grad_helper(
        logits=tf.convert_to_tensor(logits),
        labels=tf.convert_to_tensor(labels),
        label_length=tf.fill([batch_size], nlabels),
        logit_length=tf.fill([batch_size], nframes),
    )


@pytest.mark.parametrize("nframes,nlabels,vocab_size", [(3, 2, 4), (5, 3, 6), (4, 4, 5), (7, 2, 8)])
def test_loss_matches_forward_recursion(nframes, nlabels, vocab_size):
    logits, labels = random_batch(nframes * 31 + nlabels, 2, nframes, nlabels, vocab_size)
    loss, _ = run_helper(logits, labels, nframes, nlabels)
    for b in range(logits.shape[0]):
        expected = reference_loss(logits[b], labels[b], nframes, nlabels)
        assert np.isclose(float(loss[b]), expected, rtol=1e-4, atol=1e-4), f"utterance {b}: got {float(loss[b])}, want {expected}"


def test_gradient_shape_and_finiteness():
    nframes, nlabels, vocab_size = 6, 3, 7
    logits, labels = random_batch(99, 2, nframes, nlabels, vocab_size)
    loss, grad = run_helper(logits, labels, nframes, nlabels)
    assert loss.shape == (2,)
    assert grad.shape == logits.shape
    assert bool(tf.reduce_all(tf.math.is_finite(loss)))
    assert bool(tf.reduce_all(tf.math.is_finite(grad)))
    assert bool(tf.reduce_all(loss > 0)), "negative log-likelihood must be positive"


def test_gradient_matches_finite_differences():
    """The analytic gradient is the whole point of the helper, so check it numerically."""
    nframes, nlabels, vocab_size = 4, 2, 4
    logits, labels = random_batch(7, 1, nframes, nlabels, vocab_size)
    _, grad = run_helper(logits, labels, nframes, nlabels)
    grad = grad.numpy()[0]

    epsilon = 1e-2
    rng = np.random.RandomState(0)
    for _ in range(12):
        t, u, v = rng.randint(nframes), rng.randint(nlabels + 1), rng.randint(vocab_size)
        perturbed = logits.copy()
        perturbed[0, t, u, v] += epsilon
        up = reference_loss(perturbed[0], labels[0], nframes, nlabels)
        perturbed[0, t, u, v] -= 2 * epsilon
        down = reference_loss(perturbed[0], labels[0], nframes, nlabels)
        numerical = (up - down) / (2 * epsilon)
        assert np.isclose(grad[t, u, v], numerical, rtol=2e-2, atol=2e-3), f"grad[{t},{u},{v}] = {grad[t, u, v]}, finite difference = {numerical}"


def test_perfectly_confident_logits_give_near_zero_loss():
    """A model that puts all its mass on one valid alignment should pay almost nothing."""
    nframes, nlabels, vocab_size = 5, 3, 6
    labels = np.array([[1, 2, 3]], dtype=np.int32)
    logits = np.full((1, nframes, nlabels + 1, vocab_size), -10.0, dtype=np.float32)
    # emit every label on frame 0, then blank through the remaining frames
    for u in range(nlabels):
        logits[0, 0, u, labels[0, u]] = 10.0
    for t in range(nframes):
        logits[0, t, nlabels, BLANK] = 10.0

    loss, _ = run_helper(logits, labels, nframes, nlabels)
    assert float(loss[0]) < 0.1, f"expected a near-zero loss, got {float(loss[0])}"


def test_padded_batch_entries_are_independent():
    """Shorter utterances in a batch must not be affected by the padding around them."""
    nframes, nlabels, vocab_size = 6, 3, 6
    logits, labels = random_batch(2024, 1, nframes, nlabels, vocab_size)

    alone, _ = compute_rnnt_loss_and_grad_helper(
        logits=tf.convert_to_tensor(logits),
        labels=tf.convert_to_tensor(labels),
        label_length=tf.constant([nlabels - 1], tf.int32),
        logit_length=tf.constant([nframes - 2], tf.int32),
    )
    expected = reference_loss(logits[0], labels[0], nframes - 2, nlabels - 1)
    assert np.isclose(float(alone[0]), expected, rtol=1e-4, atol=1e-4)
