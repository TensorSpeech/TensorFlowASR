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

import logging
import os
import shutil
from http import HTTPStatus

import numpy as np
from keras.src.saving import serialization_lib

from tensorflow_asr import keras, tf
from tensorflow_asr.datasets import ASRDataset
from tensorflow_asr.utils import file_util

logger = logging.getLogger(__name__)


@keras.utils.register_keras_serializable(package=__name__)
class TestLogger(keras.callbacks.Callback):
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


@keras.utils.register_keras_serializable(package=__name__)
class PredictLogger(keras.callbacks.Callback):
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


@keras.utils.register_keras_serializable(package=__name__)
class TensorBoard(keras.callbacks.TensorBoard):
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
        self._profile_batch = profile_batch

    def on_train_batch_end(self, batch, logs=None):
        train_logs = dict((logs or {}).items())
        train_logs = self._collect_learning_rate(train_logs)
        return super().on_train_batch_end(batch, train_logs)

    def get_config(self):
        return {
            "log_dir": self.log_dir,
            "histogram_freq": self.histogram_freq,
            "write_graph": self.write_graph,
            "write_images": self.write_images,
            "write_steps_per_second": self.write_steps_per_second,
            "update_freq": self.update_freq,
            "profile_batch": self._profile_batch,
            "embeddings_freq": self.embeddings_freq,
            "embeddings_metadata": self.embeddings_metadata,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable(package=__name__)
class TerminateOnNaN(keras.callbacks.TerminateOnNaN):
    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable(package=__name__)
class ModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
    ):
        filepath = file_util.preprocess_paths(filepath)
        self._mode = mode
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)

    def get_config(self):
        return {
            "filepath": self.filepath,
            "monitor": self.monitor,
            "verbose": self.verbose,
            "save_best_only": self.save_best_only,
            "save_weights_only": self.save_weights_only,
            "mode": self._mode,
            "save_freq": self.save_freq,
            "initial_value_threshold": self.best,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable(package=__name__)
class BackupAndRestore(keras.callbacks.BackupAndRestore):
    def __init__(
        self,
        backup_dir,
        save_freq="epoch",
        double_checkpoint=True,
        delete_checkpoint=False,
    ):
        backup_dir = file_util.preprocess_paths(backup_dir, isdir=True)
        super().__init__(backup_dir=backup_dir, save_freq=save_freq, double_checkpoint=double_checkpoint, delete_checkpoint=delete_checkpoint)

    def get_config(self):
        return {
            "backup_dir": self.backup_dir,
            "save_freq": self.save_freq,
            "delete_checkpoint": self.delete_checkpoint,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable(package=__name__)
class EarlyStopping(keras.callbacks.EarlyStopping):
    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
    ):
        super().__init__(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights, start_from_epoch)
        self._mode = mode

    def get_config(self):
        return {
            "monitor": self.monitor,
            "min_delta": self.min_delta,
            "patience": self.patience,
            "verbose": self.verbose,
            "mode": self._mode,
            "baseline": self.baseline,
            "restore_best_weights": self.restore_best_weights,
            "start_from_epoch": self.start_from_epoch,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable(package=__name__)
class KaggleModelBackupAndRestore(BackupAndRestore):
    def __init__(
        self,
        model_handle: str,
        model_dir: str,
        save_freq="epoch",
    ):
        backup_dir = os.path.join(model_dir, "states")
        super().__init__(backup_dir, save_freq=save_freq, double_checkpoint=True, delete_checkpoint=False)

        try:
            os.environ["TQDM_DISABLE"] = "1"
            os.environ["DISABLE_KAGGLE_CACHE"] = "true"

            import kagglehub  # pylint: disable=import-outside-toplevel,unused-import

            logging.getLogger("kagglehub").disabled = True
            logging.getLogger("kagglehub").handlers.clear()

            self._api = kagglehub  # use option 2,3 to authenticate kaggle: https://github.com/Kaggle/kagglehub?tab=readme-ov-file#option-2-read-credentials-from-environment-variables pylint: disable=line-too-long

        except ImportError as e:
            raise ImportError("Kaggle library is not installed. Please install it via `pip install '.[kaggle]'`.") from e

        self._model_handle = model_handle
        self._model_dir = file_util.preprocess_paths(model_dir, isdir=True)
        if file_util.is_cloud_path(model_dir):
            raise ValueError(f"Model dir must be local path for Kaggle backup and restore. Received: {model_dir}")
        self.save_freq = save_freq
        if save_freq != "epoch" and not isinstance(save_freq, int):
            raise ValueError(
                "Invalid value for argument `save_freq`. " f"Received: save_freq={save_freq}. " "Expected either 'epoch' or an integer value."
            )

        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self._current_epoch = 0

    def _restore_kaggle(self):
        if os.path.exists(self._weights_path) and os.path.exists(self._training_metadata_path):
            return

        from kagglehub.exceptions import KaggleApiHTTPError  # pylint: disable=import-outside-toplevel

        try:
            cached_path = self._api.model_download(handle=self._model_handle, force_download=True)
            logger.info(f"Restoring model from '{cached_path}'...")
            has_version = False
            try:
                has_version = int(os.path.basename(cached_path))
            except:  # pylint: disable=bare-except
                pass
            if not has_version:
                latest_version = None
                for x in os.listdir(cached_path):
                    try:
                        latest_version = max(filter(None, (latest_version, int(x))))
                    except:  # pylint: disable=bare-except
                        pass
                if not latest_version:
                    logger.info(f"Model '{self._model_handle}' does not have any version. Skipping restore...")
                    return
                cached_path = os.path.join(cached_path, str(latest_version))
            shutil.copytree(cached_path, self._model_dir, ignore_dangling_symlinks=True, dirs_exist_ok=True)
            shutil.rmtree(cached_path)
            logger.info(f"Model restored to '{self._model_dir}'")
        except KaggleApiHTTPError as e:
            if e.response is not None and (e.response.status_code in (HTTPStatus.NOT_FOUND, HTTPStatus.FORBIDDEN)):
                logger.info(
                    f"Model '{self._model_handle}' does not exist or access is forbidden. It will be auto-create on saving. Skipping restore..."
                )

    def _backup_kaggle(self, logs, notes: str):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                return  # Don't save this epoch if loss is NaN or Inf
        self._api.model_upload(handle=self._model_handle, local_model_dir=self._model_dir, version_notes=notes, ignore_patterns=[".DS_Store"])

    def on_train_begin(self, logs=None):
        self._restore_kaggle()
        super().on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        self._current_epoch = epoch + 1
        self._last_batch_seen = 0
        if self.save_freq == "epoch":
            self._save_model()
            self._backup_kaggle(logs, notes=f"Backed up model at epoch {self._current_epoch}")

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self._save_model()
            self._backup_kaggle(logs, notes=f"Backed up model at batch {batch}")

    def get_config(self):
        return {
            "model_handle": self._model_handle,
            "model_dir": self._model_dir,
            "save_freq": self.save_freq,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def deserialize(callback_config):
    if isinstance(callback_config, list):
        return [serialization_lib.deserialize_keras_object(c) for c in callback_config]
    return serialization_lib.deserialize_keras_object(callback_config)
