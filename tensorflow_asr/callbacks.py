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

import importlib

import numpy as np
import tensorflow as tf

from tensorflow_asr.datasets import ASRDataset
from tensorflow_asr.utils import file_util
from tensorflow_asr.utils.env_util import KERAS_SRC

serialization_lib = importlib.import_module(f"{KERAS_SRC}.saving.serialization_lib")


@tf.keras.utils.register_keras_serializable("tensorflow_asr.callbacks")
class TestLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.wer = {"numer": 0, "denom": 0}
        self.cer = {"numer": 0, "denom": 0}

    @staticmethod
    def compute_wer(decode, target, dtype=tf.float32):
        decode = tf.strings.split(decode)
        target = tf.strings.split(target)
        distances = tf.cast(tf.edit_distance(decode.to_sparse(), target.to_sparse(), normalize=False), dtype)  # [B]
        lengths = tf.cast(target.row_lengths(axis=1), dtype)  # [B]
        return distances, lengths

    @staticmethod
    def compute_cer(decode, target, dtype=tf.float32):
        decode = tf.strings.bytes_split(decode)  # [B, N]
        target = tf.strings.bytes_split(target)  # [B, M]
        distances = tf.cast(tf.edit_distance(decode.to_sparse(), target.to_sparse(), normalize=False), dtype)  # [B]
        lengths = tf.cast(target.row_lengths(axis=1), dtype)  # [B]
        return distances, lengths

    def on_test_batch_end(self, batch, logs=None):
        if logs is None:
            return

        predictions = logs.pop("predictions")
        if predictions is None:
            return

        transcripts = self.model.tokenizer.detokenize(predictions.pop("_tokens"))
        targets = self.model.tokenizer.detokenize(predictions.pop("_labels"))

        wer_numer, wer_denom = tf.nest.map_structure(tf.reduce_sum, TestLogger.compute_wer(transcripts, targets))
        cer_numer, cer_denom = tf.nest.map_structure(tf.reduce_sum, TestLogger.compute_cer(transcripts, targets))

        self.wer["numer"] += wer_numer.numpy()
        self.wer["denom"] += wer_denom.numpy()
        self.cer["numer"] += cer_numer.numpy()
        self.cer["denom"] += cer_denom.numpy()

    def on_test_end(self, logs=None):
        logs = logs or {}
        logs["wer"] = np.divide(self.wer["numer"], self.wer["denom"])  # handled nan
        logs["cer"] = np.divide(self.cer["numer"], self.cer["denom"])
        return logs

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable("tensorflow_asr.callbacks")
class PredictLogger(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset: ASRDataset, output_file_path: str):
        super().__init__()
        self.test_dataset = test_dataset
        self.output_file_path = output_file_path

    def on_predict_begin(self, logs=None):
        self.index = 0
        self.output_file = tf.io.gfile.GFile(self.output_file_path, mode="w")
        self.output_file.write("\t".join(("PATH", "GROUND_TRUTH", "GREEDY", "BEAM_SEARCH")) + "\n")  # header

    def on_predict_batch_end(self, batch, logs=None):
        if logs is None:
            return

        predictions = logs.pop("outputs", None)
        if predictions is None:
            return

        transcripts = self.model.tokenizer.detokenize(predictions.pop("_tokens"))
        beam_transcripts = self.model.tokenizer.detokenize(predictions.pop("_beam_tokens"))
        targets = self.model.tokenizer.detokenize(predictions.pop("_labels"))

        for i, item in enumerate(zip(targets.numpy(), transcripts.numpy(), beam_transcripts.numpy()), start=self.index):
            groundtruth, greedy, beam = [x.decode("utf-8") for x in item]
            path = self.test_dataset.entries[i][0]
            line = "\t".join((path, groundtruth, greedy, beam)) + "\n"
            self.output_file.write(line)
            self.index += 1

    def on_predict_end(self, logs=None):
        self.index = 0
        self.output_file.close()

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable("tensorflow_asr.callbacks")
class TensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(
        self,
        log_dir="logs",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
        **kwargs,
    ):
        log_dir = file_util.preprocess_paths(log_dir, isdir=True)
        super().__init__(
            log_dir,
            histogram_freq,
            write_graph,
            write_images,
            write_steps_per_second,
            update_freq,
            profile_batch,
            embeddings_freq,
            embeddings_metadata,
            **kwargs,
        )

    def on_train_batch_end(self, batch, logs=None):
        train_logs = dict((logs or {}).items())
        train_logs = self._collect_learning_rate(train_logs)
        return super().on_train_batch_end(batch, train_logs)

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable("tensorflow_asr.callbacks")
class TerminateOnNaN(tf.keras.callbacks.TerminateOnNaN):
    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable("tensorflow_asr.callbacks")
class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(
        self,
        filepath,
        monitor: str = "val_loss",
        verbose: int = 0,
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: str = "auto",
        save_freq="epoch",
        options=None,
        initial_value_threshold=None,
        **kwargs,
    ):
        filepath = file_util.preprocess_paths(filepath)
        if options is not None:
            options = tf.train.CheckpointOptions(**options)
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, options, initial_value_threshold, **kwargs)

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable("tensorflow_asr.callbacks")
class BackupAndRestore(tf.keras.callbacks.BackupAndRestore):
    def __init__(
        self,
        backup_dir,
        save_freq="epoch",
        delete_checkpoint=True,
        save_before_preemption=False,
    ):
        backup_dir = file_util.preprocess_paths(backup_dir, isdir=True)
        super().__init__(backup_dir, save_freq, delete_checkpoint, save_before_preemption)

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable("tensorflow_asr.callbacks")
class EarlyStopping(tf.keras.callbacks.EarlyStopping):
    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def deserialize(callback_config):
    if isinstance(callback_config, list):
        return [serialization_lib.deserialize_keras_object(c) for c in callback_config]
    return serialization_lib.deserialize_keras_object(callback_config)
