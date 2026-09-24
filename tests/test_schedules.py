"""
Tests for the learning rate schedules.

The plotting is commented out: `plt.show()` blocks on an interactive backend, so it hung the
suite here. It only appeared to work because `tests/test_layers.py` happened to be imported first
and forced a headless backend -- an ordering accident, not a fix.
"""

import numpy as np

from tensorflow_asr import tf
from tensorflow_asr.optimizers.schedules import CyclicTransformerSchedule, TransformerSchedule

TOTAL_STEPS = 100000


def evaluate(schedule, total_steps=TOTAL_STEPS):
    """
    Evaluate the schedule over every step in one call.

    Looping `schedule(i)` in python costs 100k eager dispatches, which is fast in isolation but
    degrades badly once the rest of the suite has built up TensorFlow state -- it was enough to
    stall the run. The schedules are elementwise, so a single tensor call gives identical values.
    """
    return schedule(tf.range(total_steps, dtype=tf.float32)).numpy()


def plot_schedule(learning_rates, title):  # pylint: disable=unused-argument
    """
    Learning rate plot, disabled. Uncomment for a manual run. It saves rather than shows, because
    `plt.show()` blocks on a GUI backend and would hang the suite.
    """
    # import matplotlib.pyplot as plt
    #
    # from tensorflow_asr.utils import plot_util
    #
    # figure = plt.figure()
    # plt.plot(learning_rates)
    # plt.title(title)
    # plt.savefig(plot_util.get_plot_path(title))
    # plt.close(figure)


def test_transformer_schedule():
    schedule = TransformerSchedule(dmodel=176, scale=10.0, warmup_steps=10000, max_lr="0.05/(176**0.5)", min_lr=None)
    learning_rates = evaluate(schedule)

    assert learning_rates.shape == (TOTAL_STEPS,)
    assert np.all(np.isfinite(learning_rates))
    assert np.all(learning_rates >= 0)
    # warmup ramps up, then the schedule decays
    assert learning_rates[5000] > learning_rates[0]
    assert learning_rates[-1] < learning_rates.max()
    assert 0 < int(np.argmax(learning_rates)) < TOTAL_STEPS
    # plot_schedule(learning_rates, "TransformerSchedule")


def test_cyclic_transformer_schedule():
    schedule = CyclicTransformerSchedule(dmodel=320, step_size=10000, warmup_steps=15000, max_lr=0.0025)
    learning_rates = evaluate(schedule)

    assert learning_rates.shape == (TOTAL_STEPS,)
    assert np.all(np.isfinite(learning_rates))
    assert np.all(learning_rates >= 0)
    assert learning_rates.max() <= 0.0025 + 1e-6, "cyclic schedule exceeded max_lr"
    # after warmup the rate oscillates rather than decaying monotonically
    after_warmup = learning_rates[15000:]
    direction_changes = np.sum(np.diff(np.sign(np.diff(after_warmup))) != 0)
    assert direction_changes > 1, "expected the cyclic schedule to turn more than once"
    # plot_schedule(learning_rates, "CyclicTransformerSchedule")
