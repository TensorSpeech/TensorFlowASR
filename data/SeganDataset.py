from __future__ import absolute_import

import tensorflow as tf

class Dataset:
  def __init__(self, data_path, mode="train", train_sort=False):
    self.data_path = data_path
    self.mode = mode
    self.train_sort = train_sort