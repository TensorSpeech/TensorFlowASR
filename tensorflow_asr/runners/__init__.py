# Copyright 2020 Huy Le Nguyen (@usimarit)
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

import os
import tensorflow as tf


def save_from_checkpoint(func,
                         outdir: str,
                         max_to_keep: int = 10,
                         **kwargs):
    """
    Function to save models from latest saved checkpoint
    Args:
        func: function takes inputs as **kwargs and performs when checkpoint is found
        outdir: logging directory
        max_to_keep: number of checkpoints to keep
        **kwargs: contains built models, optimizers
    """
    steps = tf.Variable(0, dtype=tf.int64)  # Step must be int64
    epochs = tf.Variable(1)
    checkpoint_dir = os.path.join(outdir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"checkpoint directory not found: {checkpoint_dir}")
    ckpt = tf.train.Checkpoint(steps=steps, epochs=epochs, **kwargs)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=max_to_keep)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        func(**kwargs)
    else:
        raise ValueError("no lastest checkpoint found")
