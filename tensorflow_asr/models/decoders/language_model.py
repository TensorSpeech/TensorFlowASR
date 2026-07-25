# Copyright 2020 Huy Le Nguyen (@nglehuy)
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
"""Language model interface for shallow fusion into transducer beam search"""

from tensorflow_asr import keras, tf
from tensorflow_asr.models.base_layer import Layer


@keras.utils.register_keras_serializable(package=__name__)
class LanguageModel(Layer):
    """
    Stateful language model scored token by token alongside the transducer during beam search.

    Ref: "Pushing the Limits of Beam Search Decoding for Transducer-based ASR models",
         L. Grigoryan et al., Interspeech 2025, https://arxiv.org/abs/2506.00185

    Subclasses only have to provide `get_initial_state` and `score`; the beam search owns the
    per-hypothesis bookkeeping. Two constraints come from where this gets called:

    1. `score` runs inside the decoder's `tf.while_loop`, once per step on all `B * W` hypotheses
       at once. It must be pure TensorFlow -- no python-side lookups, no `numpy` -- otherwise the
       decoder stops being convertible to TFLite / XLA.
    2. States are a single tensor with the batch on axis 0 and any trailing rank. The beam search
       re-orders them on axis 0 whenever hypotheses are recombined, exactly as it does for the
       prediction network states, so anything that is not expressible as one tensor (a python
       dict of arrays, a ragged trie) will not survive the loop.

    Because states are threaded through the beam rather than kept internally, the same instance is
    safe to use for every utterance in a batch and across calls.
    """

    def get_initial_state(self, batch_size: int) -> tf.Tensor:
        """
        Start-of-sentence states.

        Parameters
        ----------
        batch_size : int

        Returns
        -------
        tf.Tensor, shape [B, ...]
        """
        raise NotImplementedError()

    def score(self, previous_tokens: tf.Tensor, previous_states: tf.Tensor):
        """
        Distribution over the next token given the previously emitted one.

        Parameters
        ----------
        previous_tokens : tf.Tensor, shape [B, 1]
            Last non-blank label emitted by each hypothesis
        previous_states : tf.Tensor, shape [B, ...]
            States as returned by `get_initial_state` or by a previous call to `score`

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor], shapes ([B, V], [B, ...])
            Log-probabilities over the *transducer* vocabulary, and the updated states.
            `V` must match the transducer vocabulary size and the indices must agree with the
            tokenizer. The blank column is never read by the decoder, so its value is free.
        """
        raise NotImplementedError()
