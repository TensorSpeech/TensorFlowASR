from __future__ import absolute_import

import multiprocessing
import tensorflow as tf
import numpy as np
from utils.Utils import check_key_in_dict, preprocess_paths
from ctc_decoders import Scorer
from ctc_decoders import ctc_beam_search_decoder_batch, ctc_greedy_decoder
from decoders.Decoder import Decoder


class CTCGreedyDecoder(Decoder):
    """ Decode the best guess from probs using greedy algorithm """

    def map_fn(self, splited_logits):
        _d = []
        for value in splited_logits:
            _d.append(ctc_greedy_decoder(probs_seq=value, vocabulary=self.text_featurizer.vocab_array))
        return _d

    def decode(self, probs, input_length):
        undecoded = np.array_split(probs.numpy(), self.num_cpus)

        with multiprocessing.Pool(self.num_cpus) as pool:
            decoded = pool.map(self.map_fn, undecoded)

        return np.concatenate(decoded)


class CTCBeamSearchDecoder(Decoder):
    """ Decode probs using beam search algorithm """

    def __init__(self, text_featurizer, beam_width=1024, lm_path=None,
                 alpha=None, beta=None):
        super().__init__(text_featurizer)
        self.beam_width = beam_width
        self.lm_path = lm_path
        self.alpha = alpha
        self.beta = beta
        if self.lm_path:
            assert self.alpha is not None and self.beta is not None, \
                "alpha, beta and vocab_array must be specified"
            self.scorer = Scorer(self.alpha, self.beta, model_path=self.lm_path,
                                 vocabulary=self.text_featurizer.vocab_array)
        else:
            self.scorer = None

    def decode(self, probs, input_length):
        # probs.shape = [batch_size, time_steps, num_classes]
        decoded = ctc_beam_search_decoder_batch(probs_split=probs.numpy(),
                                                vocabulary=self.text_featurizer.vocab_array,
                                                beam_size=self.beam_width,
                                                num_processes=self.num_cpus,
                                                ext_scoring_func=self.scorer)

        for idx, value in enumerate(decoded):
            _, text = [v for v in zip(*value)]
            decoded[idx] = text[0]

        return tf.convert_to_tensor(decoded, tf.string)


def create_ctc_decoder(decoder_config, text_featurizer):
    check_key_in_dict(decoder_config, keys=["name"])
    if decoder_config["name"] == "beamsearch":
        check_key_in_dict(decoder_config, keys=["beam_width"])
        if decoder_config.get("lm_path", None) is not None:
            check_key_in_dict(decoder_config, keys=["alpha", "beta"])
            decoder = CTCBeamSearchDecoder(text_featurizer=text_featurizer,
                                           beam_width=decoder_config["beam_width"],
                                           lm_path=preprocess_paths(decoder_config["lm_path"]),
                                           alpha=decoder_config["alpha"],
                                           beta=decoder_config["beta"])

        else:
            decoder = CTCBeamSearchDecoder(text_featurizer=text_featurizer,
                                           beam_width=decoder_config["beam_width"])

    elif decoder_config["name"] == "greedy":
        decoder = CTCGreedyDecoder(text_featurizer)
    else:
        raise ValueError("'decoder' value must be either 'beamsearch' or 'greedy'")
    return decoder
