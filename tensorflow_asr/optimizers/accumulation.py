"""
Gradient Accummulation for training TF2 custom training loop.
"""

from tensorflow_asr import tf


class GradientAccumulator:
    # We use the ON_READ synchronization policy so that no synchronization is
    # performed on assignment. To get the value, we call .value() which returns the
    # value on the current replica without synchronization.

    def __init__(self, ga_steps, name="ga"):
        self.name = name
        if ga_steps is None:
            raise ValueError("ga_steps must be defined")
        self._ga_steps = ga_steps

    @property
    def total_steps(self):
        return self._ga_steps

    def accumulate(self, gradients, per_ga_gradients):
        """Accumulates :obj:`gradients` on the current replica."""
        with tf.name_scope(self.name):
            return [None if x is None else x if y is None else x + y for x, y in zip(gradients, per_ga_gradients)]
