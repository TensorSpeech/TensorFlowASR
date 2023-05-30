# Copyright 2020 Huy Le Nguyen (@nglehuy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


@tf.keras.utils.register_keras_serializable("tensorflow_asr.optimizers.schedules")
class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, dmodel, initial_lr=1.0, warmup_steps=4000, max_lr=None, min_lr=None):
        super().__init__()
        self.dmodel = tf.convert_to_tensor(dmodel, dtype=tf.float32)
        self.initial_lr = tf.convert_to_tensor(initial_lr, dtype=tf.float32)
        self.warmup_steps = tf.convert_to_tensor(warmup_steps, dtype=tf.float32)
        self.max_lr = max_lr
        self.min_lr = min_lr

    def __call__(self, step):
        # lr = (d_model^-0.5) * min(step^-0.5, step*(warm_up^-1.5))
        step = tf.cast(step, dtype=tf.float32)
        lr = (self.dmodel**-0.5) * tf.math.minimum(step**-0.5, step * (self.warmup_steps**-1.5))
        lr = self.initial_lr * lr
        if self.max_lr is not None:
            lr = tf.math.minimum(self.max_lr, lr)
        if self.min_lr is not None:
            lr = tf.math.maximum(self.min_lr, lr)
        return lr

    def get_config(self):
        return {
            "dmodel": self.dmodel,
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
        }


@tf.keras.utils.register_keras_serializable("tensorflow_asr.optimizers.schedules")
class BoundExponentialDecay(tf.keras.optimizers.schedules.ExponentialDecay):
    def __init__(self, min_lr=0.0, **kwargs):
        super().__init__(**kwargs)
        self.min_lr = min_lr

    def __call__(self, step):
        with tf.name_scope(self.name or "ExponentialDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            decay_rate = tf.cast(self.decay_rate, dtype)

            global_step_recomp = tf.cast(step, dtype)
            p = global_step_recomp / decay_steps
            if self.staircase:
                p = tf.math.floor(p)
            new_lr = tf.multiply(initial_learning_rate, tf.pow(decay_rate, p), name=name)
            return tf.maximum(self.min_lr, new_lr)

    def get_config(self):
        return {
            "min_lr": self.min_lr,
        }


@tf.keras.utils.register_keras_serializable("tensorflow_asr.optimizers.schedules")
class CyclicTransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """This callback implements a cyclical learning rate policy (CLR) to the square
    root decay generally used to train transformers.
    The method cycles the learning rate around the square root decay LR with an amplitude
    equal to the target LR with a given period.
    # Arguments
        d_model: The dimension of the transformer model.
        warmup_steps: Warm up steps where the LR increases linearly.
            Default to 4000 steps.
        max_lr: Maximum value of the learning rate reachable.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.

    It is inspired from the paper:
    # References
      - [Cyclical Learning Rates for Training Neural Networks](
      https://arxiv.org/abs/1506.01186)
    """

    def __init__(self, dmodel, warmup_steps=4000, max_lr=None, step_size=None):
        """Applies triangular cyclic to the square root decay learning rate.
        Args:
        d_model: Model dimension
        warmup_steps: Warm up steps where the LR increases linearly.
        max_lr: The maximum LR.
        step_size: The size of the cyclic triangular half cycle.
        """
        super().__init__()
        self.dmodel = tf.cast(dmodel, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.max_lr = tf.cast(max_lr, tf.float32)
        self.step_size = tf.cast(step_size, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = step * (self.warmup_steps**-1.5)
        lr = 2 * tf.math.rsqrt(step)
        lr = tf.math.rsqrt(self.dmodel) * tf.math.minimum(lr, warmup)
        lr = tf.math.minimum(self.max_lr, lr)
        cycle = tf.math.floor(1 + step / (2 * self.step_size))
        x = tf.math.abs(step / self.step_size - 2 * cycle + 1)
        lr = lr * (0.5 + tf.math.maximum(0.0, x))
        lr = tf.math.minimum(self.max_lr, tf.math.minimum(lr, warmup))
        return lr

    def get_config(self):
        return {
            "dmodel": self.dmodel,
            "warmup_steps": self.warmup_steps,
            "max_lr": self.max_lr,
            "step_size": self.step_size,
        }
