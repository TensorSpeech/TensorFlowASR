from __future__ import absolute_import

import tensorflow as tf


class Checkpoint(tf.keras.callbacks.Callback):
  def __init__(self, ckpt_manager):
    self.ckpt_manager = ckpt_manager
    super().__init__()

  def on_epoch_end(self, epoch, logs=None):
    print(f"Save checkpoint at epoch {epoch}")
    self.ckpt_manager.save()
