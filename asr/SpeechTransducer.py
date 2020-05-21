from __future__ import absolute_import

import os
import gc
import sys
import time
import tensorflow as tf

from featurizers.TextFeaturizer import TextFeaturizer
from decoders.CTCDecoders import create_decoder
from models.Transducer import create_transducer_model
from utils.Utils import get_asr_config, check_key_in_dict, bytes_to_string, wer, cer

class SpeechTransducer:
  def __init__(self, configs_path, noise_filter):
    self.configs = get_asr_config(configs_path)
    self.text_featurizer = TextFeaturizer(self.configs["vocabulary_file_path"])
    self.decoder = create_decoder(decoder_config=self.configs["decoder"],
                                  index_to_token=self.text_featurizer.index_to_token,
                                  num_classes=self.text_featurizer.num_classes,
                                  vocab_array=self.text_featurizer.vocab_array)
    self.model, self.optimizer = create_transducer_model(num_classes=self.text_featurizer.num_classes,
                                                         last_activation=self.configs["last_activation"],
                                                         base_model=self.configs["base_model"],
                                                         streaming_size=self.configs["streaming_size"],
                                                         speech_conf=self.configs["speech_conf"])
    self.noise_filter = noise_filter
    self.writer = None

  def _create_checkpoints(self, model):
    if not self.configs["checkpoint_dir"]:
      raise ValueError("Must set checkpoint_dir")
    if not os.path.exists(self.configs["checkpoint_dir"]):
      os.makedirs(self.configs["checkpoint_dir"])
    self.ckpt = tf.train.Checkpoint(model=model,
                                    optimizer=self.optimizer)
    self.ckpt_manager = tf.train.CheckpointManager(
      self.ckpt, self.configs["checkpoint_dir"], max_to_keep=None)

  @tf.function
  def train(self, model, dataset, optimizer, loss, num_classes, epoch, num_epochs):
    pass
