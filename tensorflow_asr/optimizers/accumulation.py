"""
Gradient Accummulation for training TF2 custom training loop.
Copy and modified from https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/optimizers/utils.py.
"""

import tensorflow as tf


class GradientAccumulator:
    # We use the ON_READ synchronization policy so that no synchronization is
    # performed on assignment. To get the value, we call .value() which returns the
    # value on the current replica without synchronization.

    def __init__(
        self,
        ga_steps,
        model: tf.keras.Model,
        name="ga",
    ):
        self.name = name
        if ga_steps is None:
            raise ValueError("ga_steps must be defined")
        if model.trainable_variables is None:
            raise ValueError("trainable_variables must be defined")
        self._ga_steps = ga_steps
        self._gradients = [None for _ in model.trainable_variables]
        self._gradient_inited = False

    @property
    def total_steps(self):
        return self._ga_steps

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        if not self._gradient_inited:
            raise ValueError("gradients are not initialized")
        return list(gradient.value() if gradient is not None else gradient for gradient in self._gradients)

    def accumulate(self, gradients):
        """Accumulates :obj:`gradients` on the current replica."""
        if not self._gradient_inited:
            for i, gradient in enumerate(gradients):
                if gradient is None:
                    continue
                self._gradients[i] = tf.Variable(
                    tf.zeros_like(gradient),
                    trainable=False,
                    synchronization=tf.VariableSynchronization.ON_READ,
                    aggregation=tf.VariableAggregation.NONE,
                    name=f"{self.name}_{i}",
                )
            self._gradient_inited = True
        for accum_gradient, gradient in zip(self._gradients, gradients):
            if gradient is not None and accum_gradient is not None:
                accum_gradient.assign_add(gradient, read_value=False)

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        for gradient in self._gradients:
            if gradient is not None:
                gradient.assign(tf.zeros_like(gradient), read_value=False)
