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
"""Language model interface for fusion into transducer beam search"""

from tensorflow_asr import keras, tf


@keras.utils.register_keras_serializable(package=__name__)
class LanguageModel(keras.Model):
    """
    Language model over the transducer's own vocabulary, trained on text and scored during decoding.

    Ref: "Pushing the Limits of Beam Search Decoding for Transducer-based ASR models",
         L. Grigoryan et al., Interspeech 2025, https://arxiv.org/abs/2506.00185

    It is a full `keras.Model` rather than a layer because it is trained on its own, by
    `scripts/train_lm.py`, and saved to its own h5. Two keys of the top-level `lm_config` point at
    one, and either may carry a `weights` key with the h5 to load:

    - `external_config`, the LM shallow fused into the scores (weighted by
      `decoder_config.lm_alpha`).
    - `internal_config`, the low-order LM that LODR subtracts (weighted by
      `decoder_config.lm_beta`).

    Subclasses implement two forward passes, because training and decoding want different shapes:

    - `call(tokens)` -- **training**. Teacher forced over a whole sequence at once, `[B, U]` in,
      `[B, U, V]` out. This is what `fit` runs, so it is the keras-idiomatic `call`.
    - `call_next(previous_tokens, previous_states)` -- **decoding**. One step, carrying state,
      `[B, 1]` in, `[B, V]` out. The beam search calls this. The name matches what the rest of the
      repository already uses for a single stateful step (`TransducerPrediction.call_next`,
      `Encoder.call_next`, `Transducer.call_next`).

    Three constraints come from where `call_next` is used:

    1. It runs inside the decoder's `tf.while_loop`, once per step on all `B * W` hypotheses at
       once. It must be pure TensorFlow -- no python-side lookups, no `numpy` -- otherwise the
       decoder stops being convertible to TFLite / XLA.
    2. States are a single tensor with the batch on axis 0 and any trailing rank. The beam search
       re-orders them on axis 0 whenever hypotheses are recombined, exactly as it does for the
       prediction network states, so anything that is not expressible as one tensor (a python dict
       of arrays, a ragged trie) will not survive the loop.
    3. Weights must exist before decoding starts. `make()` builds them; a model that first built
       inside the loop body would be creating variables inside a `tf.while_loop`, which graph mode
       rejects.

    Because states are threaded through the beam rather than kept internally, the same instance is
    safe to use for every utterance in a batch and across calls.

    Token conventions, shared with the decoder:

    - The blank index doubles as **start of sentence**. The beam holds `last_token = blank` until a
      hypothesis emits its first label, so training must feed `[blank, t_1, ..., t_{n-1}]` to
      predict `[t_1, ..., t_n]`. `shift_tokens` does that.
    - There is no end-of-sentence symbol. A hypothesis ends when the encoder frames run out, never
      on an emitted token, so predicting one would be wasted capacity.
    - Outputs are log-probabilities over the *transducer* vocabulary with matching indices. The
      blank column is never read by the decoder, so its value is free.
    """

    def make(self, batch_size=1, max_length=8):
        """
        Build the weights, so `load_weights` and decoding have something to fill.

        Shapes do not matter beyond making the graph concrete -- the model is used at whatever
        length the caller has -- so the defaults are deliberately tiny.
        """
        self(keras.Input(shape=[max_length], batch_size=batch_size, dtype=tf.int32), training=False)

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

    def call(self, tokens: tf.Tensor, training=False):
        """
        Teacher-forced forward pass over whole sequences, used for training.

        Parameters
        ----------
        tokens : tf.Tensor, shape [B, U], dtype int32
            Input tokens, already shifted so that position `u` holds the token *before* the one to
            be predicted at `u`. See `shift_tokens`.

        Returns
        -------
        tf.Tensor, shape [B, U, V]
            Log-probabilities of the next token at each position.
        """
        raise NotImplementedError()

    def call_next(self, previous_tokens: tf.Tensor, previous_states: tf.Tensor):
        """
        One decoding step: distribution over the next token given the previously emitted one.

        Parameters
        ----------
        previous_tokens : tf.Tensor, shape [B, 1]
            Last non-blank label emitted by each hypothesis
        previous_states : tf.Tensor, shape [B, ...]
            States as returned by `get_initial_state` or by a previous call to this method

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor], shapes ([B, V], [B, ...])
            Log-probabilities over the transducer vocabulary, and the updated states.
        """
        raise NotImplementedError()


def shift_tokens(tokens: tf.Tensor, blank: int) -> tf.Tensor:
    """
    Turn target tokens into teacher-forcing inputs: `[t1, t2, t3] -> [blank, t1, t2]`.

    Blank stands in for start of sentence, matching how the beam search seeds `last_token`, so the
    first position is trained to predict what actually starts an utterance.
    """
    batch_size = tf.shape(tokens)[0]
    start = tf.fill([batch_size, 1], tf.cast(blank, tokens.dtype))
    return tf.concat([start, tokens[:, :-1]], axis=1)
