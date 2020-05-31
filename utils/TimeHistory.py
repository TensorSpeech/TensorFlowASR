from __future__ import absolute_import

import time
import tensorflow as tf


class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, filename):
        self.filename = filename
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = int(time.time() - self.epoch_time_start)
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(f"{epoch + 1}\t{duration}\n")
