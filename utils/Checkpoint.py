from __future__ import absolute_import

import tensorflow as tf


class Checkpoint(tf.keras.callbacks.Callback):
  def __init__(self, ckpt_manager):
    self.ckpt_manager = ckpt_manager
    super().__init__()

  def on_epoch_end(self, epoch, logs=None):
    self.ckpt_manager.save()
    print(f"Saved checkpoint at epoch {epoch + 1}")
