"""
Gradient Accummulation for training TF2 custom training loop.
"""

from keras.src.optimizers.base_optimizer import BaseOptimizer

from tensorflow_asr import tf


class GradientAccumulator:
    # We use the ON_READ synchronization policy so that no synchronization is
    # performed on assignment. To get the value, we call .value() which returns the
    # value on the current replica without synchronization.

    def __init__(self, ga_steps, optimizer: BaseOptimizer, name="ga"):
        self.name = name
        if ga_steps is None:
            raise ValueError("ga_steps must be defined")
        self._ga_steps = ga_steps
        self._optimizer = optimizer
        self._accumulated_gradients = []
        self.built = False

    def build(self, variables):
        if not self._optimizer.built:
            self._optimizer.build(variables)
        for i, variable in enumerate(variables):
            self._accumulated_gradients.append(
                self._optimizer.add_variable_from_reference(
                    variable,
                    name="gradient_accumulator",
                )
            )
        self.built = True

    @property
    def total_steps(self):
        return self._ga_steps

    # def is_apply_step(self, step):
    #     return tf.math.equal(step % self._ga_steps, 0)

    def reset(self):
        for g_acc in self._accumulated_gradients:
            g_acc.assign(tf.zeros(g_acc.shape, dtype=g_acc.dtype))

    def _get_acc_grads(self, trainable_variables):
        # `trainable_variables` might have been filtered in previous
        # processing steps, so we need to ensure the correct mapping between
        # `self._accumulated_gradients` and `trainable_variables`
        acc_grads = [self._accumulated_gradients[self._optimizer._get_variable_index(v)] for v in trainable_variables]
        return acc_grads

    def accumulate(self, grads, trainable_variables):
        """Accumulates :obj:`gradients` on the current replica."""
        if not self.built:
            self.build(trainable_variables)
        # return [None if x is None else x if y is None else x + y for x, y in zip(gradients, per_ga_gradients)]
        acc_grads = self._get_acc_grads(trainable_variables)
        new_g_accs = [(g + acc_g) for g, acc_g in zip(grads, acc_grads)]
        for n_g_acc, g_acc in zip(new_g_accs, acc_grads):
            g_acc.assign(n_g_acc)

    def gradients(self, grads, trainable_variables):
        """Gets the gradients for the apply step."""
        if not self.built:
            self.build(trainable_variables)
        acc_grads = self._get_acc_grads(trainable_variables)
        grads = [(g + acc_g) / self._ga_steps for g, acc_g in zip(grads, acc_grads)]
        return grads
