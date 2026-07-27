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
"""LSTM language model for shallow fusion into transducer beam search"""

from tensorflow_asr import keras, tf
from tensorflow_asr.models.lm.language_model import LanguageModel


@keras.utils.register_keras_serializable(package=__name__)
class LSTMLanguageModel(LanguageModel):
    """
    Stacked LSTM language model over the transducer's own vocabulary.

    The defaults reproduce the external LM of the ILME paper
    (https://arxiv.org/abs/2011.01991): two 2048-unit LSTM layers over a 512-dimensional embedding,
    with the input and output embeddings tied -- 58M parameters at their 3999 word-pieces. LODR
    (https://arxiv.org/abs/2203.16776) uses an 87M Transformer instead, and the k2/icefall LODR
    recipe a 3-layer 2048-unit RNN; all three are the same shape of model at the same order of
    size, so this covers the published setups.

    **The vocabulary must be the transducer's own.** `call_next` returns log-probabilities indexed
    against it, so the LM has to be trained over the same tokenizer -- the same SentencePiece or
    wordpiece model, the same merges, the same integer indices. This is why a pretrained LM from
    another toolkit cannot be dropped in even after converting its weights, and why
    `scripts/train_lm.py` tokenizes with the tokenizer built from your own config.

    Tied embeddings are the reason for the projection layer. Tying means reusing the [V, E]
    embedding matrix as the output layer, which requires the thing being projected to be E-wide,
    but the last LSTM is `units`-wide. So a `units -> embed_dim` projection sits between them, as
    in the paper. With `tie_embeddings=False` the projection is dropped and an ordinary
    `units -> V` dense layer is used, which is bigger and usually slightly worse.

    Decoding state is `[B, nlayers, 2, units]` -- one `(h, c)` pair per layer, stacked into the
    single tensor the beam search requires, in the same layout the transducer's own prediction
    network uses.

    A note on size. At these defaults this is 56M parameters evaluated once per beam step, on all
    `B * W` hypotheses. Transducer decoding is a long sequence of small steps, so that cost is paid
    hundreds of times per utterance and it dominates. Start smaller -- `units=512, nlayers=2` is a
    reasonable first try -- and only grow it if the WER pays for the time.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        units: int = 2048,
        nlayers: int = 2,
        tie_embeddings: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.units = units
        self.nlayers = nlayers
        self.tie_embeddings = tie_embeddings
        self.dropout = dropout

        self.embedding = keras.layers.Embedding(vocab_size, embed_dim, name="embedding", dtype=self.dtype)
        # `return_sequences` for the training pass over a whole transcript, `return_state` for the
        # one-step decoding pass -- the same layer object serves both, so the weights are shared
        # without any copying between the two paths.
        self.lstms = [
            keras.layers.LSTM(
                units,
                return_sequences=True,
                return_state=True,
                dropout=dropout,
                name=f"lstm_{i}",
                dtype=self.dtype,
            )
            for i in range(nlayers)
        ]
        if tie_embeddings:
            self.projection = keras.layers.Dense(embed_dim, name="projection", dtype=self.dtype)
            self.output_bias = self.add_weight(name="output_bias", shape=[vocab_size], initializer="zeros", dtype="float32")
            self.output_dense = None
        else:
            self.projection = None
            self.output_bias = None
            self.output_dense = keras.layers.Dense(vocab_size, name="logits", dtype=self.dtype)

    def get_initial_state(self, batch_size):
        return tf.zeros([batch_size, self.nlayers, 2, self.units], dtype=self.compute_dtype)

    def _log_probs(self, outputs, training=False):
        """Shared head: project, apply the output layer, normalise. `outputs` is [..., units]."""
        if self.tie_embeddings:
            outputs = self.projection(outputs, training=training)  # [..., E]
            # einsum rather than `tf.matmul(..., transpose_b=True)`: the embedding matrix is rank 2
            # and `outputs` is rank 3 during training, which matmul will not broadcast over.
            logits = tf.einsum("...e,ve->...v", outputs, tf.cast(self.embedding.embeddings, outputs.dtype))
            logits = logits + tf.cast(self.output_bias, logits.dtype)
        else:
            logits = self.output_dense(outputs, training=training)
        return tf.nn.log_softmax(tf.cast(logits, tf.float32))

    def call(self, tokens, training=False):
        outputs = self.embedding(tokens)  # [B, U] => [B, U, E]
        for lstm in self.lstms:
            outputs = lstm(outputs, training=training)[0]  # (sequences, h, c) -> sequences
        return self._log_probs(outputs, training=training)  # [B, U, V]

    def call_next(self, previous_tokens, previous_states):
        outputs = self.embedding(previous_tokens)  # [B, 1] => [B, 1, E]
        new_states = []
        for i, lstm in enumerate(self.lstms):
            state = previous_states[:, i]  # [B, 2, units]
            outputs, memory, carry = lstm(outputs, initial_state=[state[:, 0], state[:, 1]])
            new_states.append(tf.stack([memory, carry], axis=1))  # [B, 2, units]
        log_probs = self._log_probs(outputs)  # [B, 1, V]
        return tf.squeeze(log_probs, axis=1), tf.stack(new_states, axis=1)  # [B, V], [B, nlayers, 2, units]

    def compute_output_shape(self, tokens_shape):
        return (*tokens_shape, self.vocab_size)

    def get_config(self):
        return {
            **super().get_config(),
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "units": self.units,
            "nlayers": self.nlayers,
            "tie_embeddings": self.tie_embeddings,
            "dropout": self.dropout,
        }
