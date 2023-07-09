# Copyright 2023 Huy Le Nguyen (@nglehuy)
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


import tensorflow as tf

from tensorflow_asr.datasets import ASRDataset
from tensorflow_asr.metrics.error_rates import ErrorRate
from tensorflow_asr.tokenizers import Tokenizer


def compute_wer(decode, target, dtype=tf.float32):
    decode = tf.strings.split(decode)
    target = tf.strings.split(target)
    distances = tf.cast(tf.edit_distance(decode.to_sparse(), target.to_sparse(), normalize=False), dtype)  # [B]
    lengths = tf.cast(target.row_lengths(axis=1), dtype)  # [B]
    return distances, lengths


def compute_cer(decode, target, dtype=tf.float32):
    decode = tf.strings.bytes_split(decode)  # [B, N]
    target = tf.strings.bytes_split(target)  # [B, M]
    distances = tf.cast(tf.edit_distance(decode.to_sparse(), target.to_sparse(), normalize=False), dtype)  # [B]
    lengths = tf.cast(target.row_lengths(axis=1), dtype)  # [B]
    return distances, lengths


class TestLogger(tf.keras.callbacks.Callback):
    def __init__(self, tokenizer: Tokenizer, eval_dataset: ASRDataset):
        super().__init__()
        self.tokenizer = tokenizer

    def set_model(self, model):
        super().set_model(model)
        self.model.add_custom_metric(ErrorRate(name="wer"))
        self.model.add_custom_metric(ErrorRate(name="cer"))
        self._make_update_state()

    def _make_update_state(self):
        def update_state(wer, cer):
            def update(wer, cer):
                self.model._tfasr_metrics["wer"].update_state(wer)
                self.model._tfasr_metrics["cer"].update_state(cer)

            return self.model.distribute_strategy.run(update, args=(wer, cer))

        self.update_state = tf.function(update_state)

    def on_test_batch_end(self, batch, logs=None):
        if logs is None:
            return

        predictions = logs.pop("predictions")
        transcripts = self.tokenizer.detokenize(predictions.pop("_tokens"))
        targets = self.tokenizer.detokenize(predictions.pop("_labels"))

        wer = compute_wer(transcripts, targets)
        cer = compute_cer(transcripts, targets)
        self.update_state(wer, cer)


class PredictLogger(tf.keras.callbacks.Callback):
    def __init__(self, tokenizer: Tokenizer, test_dataset: ASRDataset, output_file_path: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.output_file_path = output_file_path

    def on_predict_begin(self, logs=None):
        self.index = 0
        self.output_file = tf.io.gfile.GFile(self.output_file_path, mode="w")
        self.output_file.write("\t".join(("GROUND_TRUTH", "GREEDY", "BEAM_SEARCH")) + "\n")  # header

    def on_predict_batch_end(self, batch, logs=None):
        if logs is None:
            return

        predictions = logs.pop("predictions")
        transcripts = self.tokenizer.detokenize(predictions.pop("_tokens"))
        beam_transcripts = self.tokenizer.detokenize(predictions.pop("_beam_tokens"))
        targets = self.tokenizer.detokenize(predictions.pop("_labels"))

        for i, item in enumerate(zip(targets.numpy(), transcripts.numpy(), beam_transcripts.numpy()), start=self.index):
            groundtruth, greedy, beam = [x.decode("utf-8") for x in item]
            path = self.test_dataset.entries[i][0]
            line = "\t".join((path, groundtruth, greedy, beam)) + "\n"
            self.output_file.write(line)
            self.index += 1

    def on_predict_end(self, logs=None):
        self.index = 0
        self.output_file.close()
