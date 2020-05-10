from __future__ import absolute_import

import multiprocessing
import os
import tensorflow as tf
import numpy as np
from utils.Utils import check_key_in_dict, bytes_to_string
from ctc_decoders import Scorer
from ctc_decoders import ctc_beam_search_decoder_batch, ctc_greedy_decoder


class Decoder:
  def __init__(self, index_to_token, num_classes, vocab_array=None):
    self.index_to_token = index_to_token
    # tensorflow.org/api_docs/python/tf/keras/backend/ctc_decode
    # default blank index is -1
    self.blank_index = -1
    self.num_cpus = multiprocessing.cpu_count()
    self.vocab_array = vocab_array
    self.num_classes = num_classes

  def convert_to_string(self, decoded):
    # Remove blank indices
    def map_cvrt(elem):
      elem = elem.numpy()
      elem = elem[elem != self.blank_index]
      elem = elem[elem != self.num_classes - 1]
      return ''.join([self.index_to_token[i] for i in elem])

    # Convert to string
    decoded = tf.map_fn(map_cvrt, decoded, dtype=tf.string)
    return bytes_to_string(decoded.numpy())

  def convert_to_string_single(self, decoded: np.ndarray):
    decoded = decoded[decoded != self.blank_index]
    decoded = decoded[decoded != self.num_classes - 1]
    return ''.join([self.index_to_token[i] for i in decoded])

  def __call__(self, probs, input_length, last_activation="linear"):
    if last_activation != "softmax":
      probs = tf.nn.softmax(probs)
    decoded = tf.py_function(self.decode, inp=[probs, input_length],
                             Tout=tf.string)
    return decoded

  def decode(self, probs, input_length):
    pass


class GreedyDecoder(Decoder):
  """ Decode the best guess from probs using greedy algorithm """

  def map_fn(self, splited_logits):
    _d = []
    for value in splited_logits:
      _d.append(ctc_greedy_decoder(probs_seq=value,
                                   vocabulary=self.vocab_array))
    return _d

  def decode(self, probs, input_length):
    # probs.shape = [batch_size, time_steps, num_classes]
    # decoded = tf.keras.backend.ctc_decode(y_pred=probs,
    #                                       input_length=input_length,
    #                                       greedy=True)
    # # decoded shape = [batch_size, decoded index]
    # decoded = decoded[0]
    # return self.convert_to_string(decoded)

    undecoded = np.array_split(probs.numpy(), self.num_cpus)

    with multiprocessing.Pool(self.num_cpus) as pool:
      decoded = pool.map(self.map_fn, undecoded)

    return np.concatenate(decoded)


class BeamSearchDecoder(Decoder):
  """ Decode probs using beam search algorithm """

  def __init__(self, index_to_token, num_classes, beam_width=1024, lm_path=None,
               alpha=None, beta=None, vocab_array=None):
    super().__init__(index_to_token, num_classes, vocab_array)
    self.beam_width = beam_width
    self.lm_path = lm_path
    self.alpha = alpha
    self.beta = beta
    if self.lm_path:
      assert self.alpha and self.beta and self.vocab_array, \
        "alpha, beta and vocab_array must be specified"
      self.scorer = Scorer(self.alpha, self.beta, model_path=self.lm_path,
                           vocabulary=self.vocab_array)
    else:
      self.scorer = None

  def decode(self, probs, input_length):
    # probs.shape = [batch_size, time_steps, num_classes]
    decoded = ctc_beam_search_decoder_batch(probs_split=probs.numpy(),
                                            vocabulary=self.vocab_array,
                                            beam_size=self.beam_width,
                                            num_processes=self.num_cpus,
                                            ext_scoring_func=self.scorer)

    for idx, value in enumerate(decoded):
      _, text = [v for v in zip(*value)]
      decoded[idx] = text[0]

    return tf.convert_to_tensor(decoded, tf.string)

  # decoded = tf.keras.backend.ctc_decode(y_pred=probs,
  #                                       input_length=input_length,
  #                                       greedy=False,
  #                                       beam_width=self.beam_width)
  # # decoded shape = [batch_size, top_path=1, decoded index]
  # # get the first object of the list of top-path objects
  # decoded = decoded[0]
  # print(decoded)
  # return self.convert_to_string(decoded)


def create_decoder(decoder_config, index_to_token, num_classes, vocab_array):
  check_key_in_dict(decoder_config, keys=["name"])
  if decoder_config["name"] == "beamsearch":
    check_key_in_dict(decoder_config, keys=["beam_width"])
    if decoder_config.get("lm_path", None) is not None:
      check_key_in_dict(decoder_config, keys=["alpha", "beta"])
      decoder = BeamSearchDecoder(
        index_to_token=index_to_token,
        num_classes=num_classes,
        beam_width=decoder_config["beam_width"],
        lm_path=os.path.expanduser(decoder_config["lm_path"]),
        alpha=decoder_config["alpha"],
        beta=decoder_config["beta"],
        vocab_array=vocab_array)
    else:
      decoder = BeamSearchDecoder(index_to_token=index_to_token,
                                  num_classes=num_classes,
                                  beam_width=decoder_config["beam_width"],
                                  vocab_array=vocab_array)
  elif decoder_config["name"] == "greedy":
    decoder = GreedyDecoder(index_to_token=index_to_token,
                            num_classes=num_classes,
                            vocab_array=vocab_array)
  else:
    raise ValueError("'decoder' value must be either 'beamsearch',\
                         'beamsearch_lm' or 'greedy'")
  return decoder
