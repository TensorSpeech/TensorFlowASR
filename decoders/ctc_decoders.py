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
from __future__ import absolute_import

import multiprocessing
import tensorflow as tf
import numpy as np

from featurizers.text_featurizers import TextFeaturizer
from utils.utils import check_key_in_dict, preprocess_paths
from ctc_decoders import Scorer
from ctc_decoders import ctc_beam_search_decoder_batch, ctc_greedy_decoder
from decoders.decoder import Decoder


class CTCDecoder(Decoder):
    """ Decoder for ctc models """

    def __init__(self,
                 decoder_config: dict,
                 text_featurizer: TextFeaturizer):
        check_key_in_dict(decoder_config, keys=["name"])
        assert decoder_config["name"] in ["greedy", "beamsearch"], \
            f"Decoder's name must be either 'greedy' or 'beamsearch': {decoder_config['name']}"
        self.name = decoder_config["name"]
        super(CTCDecoder, self).__init__(decoder_config, text_featurizer)

    def set_decoding_func(self):
        if self.decoder_config["name"] == "greedy":
            return self._perform_greedy

        check_key_in_dict(self.decoder_config, keys=["beam_width"])
        if self.decoder_config.get("lm_path", None) is not None:
            self.name = f"{self.decoder_config['name']}_lm"  # Update name
            self.decoder_config["lm_path"] = preprocess_paths(self.decoder_config["lm_path"])
            check_key_in_dict(self.decoder_config, keys=["alpha", "beta"])
            self.scorer = Scorer(self.decoder_config["alpha"], self.decoder_config["beta"],
                                 model_path=self.decoder_config["lm_path"],
                                 vocabulary=self.text_featurizer.vocab_array)
        else:
            self.scorer = None
        return self._perform_beamsearch

    def greedy_map_fn(self, splited_logits):
        _d = []
        for value in splited_logits:
            _d.append(ctc_greedy_decoder(probs_seq=value, vocabulary=self.text_featurizer.vocab_array))
        return _d

    def _perform_greedy(self, probs, *args, **kwargs):
        undecoded = np.array_split(probs.numpy(), self.num_cpus)

        with multiprocessing.Pool(self.num_cpus) as pool:
            decoded = pool.map(self.greedy_map_fn, undecoded)

        return np.concatenate(decoded)

    def _perform_beamsearch(self, probs, *args, **kwargs):
        decoded = ctc_beam_search_decoder_batch(probs_split=probs.numpy(),
                                                vocabulary=self.text_featurizer.vocab_array,
                                                beam_size=self.decoder_config["beam_width"],
                                                num_processes=self.num_cpus,
                                                ext_scoring_func=self.scorer)

        for idx, value in enumerate(decoded):
            _, text = [v for v in zip(*value)]
            decoded[idx] = text[0]

        return tf.convert_to_tensor(decoded, tf.string)
