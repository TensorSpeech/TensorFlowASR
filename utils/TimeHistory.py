from __future__ import absolute_import

import time
import tensorflow as tf


class TimeHistory(tf.keras.callbacks.Callback):
  def __init__(self, filename):
    self.filename = filename
    self.times = []
    self.total_time = None
    super().__init__()

  def on_train_begin(self, logs={}):
    self.total_time = time.time()

  def on_train_end(self, logs={}):
    self.total_time = time.time() - self.total_time
    with open(self.filename, "a") as f:
      f.write(f"Total: {self.total_time}\n")

  def on_epoch_begin(self, epoch, logs={}):
    self.epoch_time_start = time.time()

  def on_epoch_end(self, epoch, logs={}):
    self.times.append(time.time() - self.epoch_time_start)
    with open(self.filename, "a") as f:
      f.write(f"{epoch}\t{self.times[-1]}\n")
