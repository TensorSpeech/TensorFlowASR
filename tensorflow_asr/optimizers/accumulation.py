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
        self._ga_steps = tf.constant(ga_steps, dtype=tf.int32)
        self._accum_step = tf.Variable(
            tf.constant(0, dtype=tf.int32),
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            name="accum_step",
        )
        self._gradients = [None for _ in model.trainable_variables]
        self._gradient_inited = False

    @property
    def step(self):
        """Number of accumulated steps."""
        return self._accum_step.value()

    @property
    def total_steps(self):
        return self._ga_steps

    @property
    def is_apply_step(self):
        return tf.equal(self.step, self.total_steps)

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        if not self._gradient_inited:
            raise ValueError("gradients are not initialized")
        return tf.cond(  # zeros gradients so that apply_gradient has no effect
            self.is_apply_step,
            lambda: [gradient.value() if gradient is not None else gradient for gradient in self._gradients],
            lambda: [tf.zeros_like(gradient) if gradient is not None else gradient for gradient in self._gradients],
        )

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
        self._accum_step.assign_add(1)

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        self._accum_step.assign(0)
        for gradient in self._gradients:
            if gradient is not None:
                gradient.assign(tf.zeros_like(gradient), read_value=False)
