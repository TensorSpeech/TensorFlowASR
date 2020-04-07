from __future__ import absolute_import

import multiprocessing
import os
import tensorflow as tf
import numpy as np
from utils.Utils import check_key_in_dict
from ctc_decoders import Scorer
from ctc_decoders import ctc_beam_search_decoder_batch


class Decoder:
  def __init__(self, index_to_token):
    self.index_to_token = index_to_token
    # tensorflow.org/api_docs/python/tf/keras/backend/ctc_decode
    # default blank index is -1
    self.blank_index = -1

  def convert_to_string(self, decoded):
    # Remove blank indices
    def map_cvrt(elem):
      elem = np.array(elem)
      elem = elem[elem != self.blank_index]
      return ''.join([self.index_to_token[i] for i in elem])

    # Convert to string
    return tf.map_fn(map_cvrt, decoded, dtype=tf.string)

  def decode(self, probs, input_length):
    pass


class GreedyDecoder(Decoder):
  """ Decode the best guess from probs using greedy algorithm """

  def decode(self, probs, input_length):
    # probs.shape = [batch_size, time_steps, num_classes]
    decoded = tf.keras.backend.ctc_decode(y_pred=probs,
                                          input_length=input_length,
                                          greedy=True)
    # decoded shape = [batch_size, decoded index]
    decoded = decoded[0]
    return self.convert_to_string(decoded)


class BeamSearchDecoder(Decoder):
  """ Decode probs using beam search algorithm """

  def __init__(self, index_to_token, beam_width=1024, lm_path=None,
               alpha=None, beta=None, vocab_array=None):
    super().__init__(index_to_token)
    self.beam_width = beam_width
    self.lm_path = lm_path
    self.alpha = alpha
    self.beta = beta
    self.vocab_array = vocab_array
    self.num_cpus = multiprocessing.cpu_count()
    if self.lm_path:
      assert self.alpha and self.beta and self.vocab_array, \
        "alpha, beta and vocab_array must be specified"
      self.scorer = Scorer(self.alpha, self.beta, model_path=self.lm_path,
                           vocabulary=self.vocab_array)

  def decode(self, probs, input_length):
    # probs.shape = [batch_size, time_steps, num_classes]
    if self.lm_path:
      decoded = ctc_beam_search_decoder_batch(probs.numpy(), self.vocab_array,
                                              beam_size=self.beam_width,
                                              num_processes=self.num_cpus,
                                              ext_scoring_func=self.scorer)
      for idx, value in enumerate(decoded):
        _, text = [v for v in zip(*value)]
        decoded[idx] = text[0]

      return tf.convert_to_tensor(decoded)

    decoded = tf.keras.backend.ctc_decode(y_pred=probs,
                                          input_length=input_length,
                                          greedy=False,
                                          beam_width=self.beam_width)
    # decoded shape = [batch_size, top_path=1, decoded index]
    # get the first object of the list of top-path objects
    decoded = decoded[0]
    return self.convert_to_string(decoded)


def create_decoder(decoder_config, index_to_token, vocab_array):
  check_key_in_dict(decoder_config, keys=["name"])
  if decoder_config["name"] == "beamsearch":
    check_key_in_dict(decoder_config, keys=["beam_width"])
    if "lm_path" in decoder_config.keys():
      check_key_in_dict(decoder_config, keys=["alpha", "beta"])
      decoder = BeamSearchDecoder(
        index_to_token=index_to_token,
        beam_width=decoder_config["beam_width"],
        lm_path=os.path.expanduser(decoder_config["lm_path"]),
        alpha=decoder_config["alpha"],
        beta=decoder_config["beta"],
        vocab_array=vocab_array)
    else:
      decoder = BeamSearchDecoder(index_to_token=index_to_token,
                                  beam_width=decoder_config["beam_width"])
  elif decoder_config["name"] == "greedy":
    decoder = GreedyDecoder(index_to_token=index_to_token)
  else:
    raise ValueError("'decoder' value must be either 'beamsearch',\
                         'beamsearch_lm' or 'greedy'")
  return decoder
