# This implementation is inspired from
# https://github.com/dathudeptrai/TensorflowTTS/blob/master/tensorflow_tts/trainers/base_trainer.py
# Copyright 2020 Minh Nguyen (@dathudeptrai) Copyright 2020 Huy Le Nguyen (@usimarit)
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
import abc
import os
from tqdm import tqdm
from colorama import Fore

import numpy as np
import tensorflow as tf

from ..utils.utils import preprocess_paths, get_num_batches, bytes_to_string
from ..utils.metrics import ErrorRate, wer, cer


class BaseRunner(metaclass=abc.ABCMeta):
    """ Customized runner module for all models """

    def __init__(self, config: dict):
        """
        running_config:
            batch_size: 8
            num_epochs:          20
            outdir:              ...
            log_interval_steps:  200
            eval_interval_steps: 200
            save_interval_steps: 200
        """
        self.config = config
        self.config["outdir"] = preprocess_paths(self.config["outdir"])
        # Writers
        self.writers = {
            "train": tf.summary.create_file_writer(
                os.path.join(self.config["outdir"], "tensorboard", "train")),
            "eval": tf.summary.create_file_writer(
                os.path.join(self.config["outdir"], "tensorboard", "eval"))
        }

    def add_writer(self, stage: str):
        self.writers[stage] = tf.summary.create_file_writer(
            os.path.join(self.config["outdir"], "tensorboard", stage))

    def _write_to_tensorboard(self,
                              list_metrics: dict,
                              step: any,
                              stage: str = "train"):
        """Write variables to tensorboard."""
        writer = self.writers.get(stage, None)

        if writer is None:
            raise ValueError(f"Missing writer for stage {stage}")

        with writer.as_default():
            for key, value in list_metrics.items():
                if isinstance(value, tf.keras.metrics.Metric):
                    tf.summary.scalar(key, value.result(), step=step)
                else:
                    tf.summary.scalar(key, value, step=step)
                writer.flush()


class BaseTrainer(BaseRunner):
    """Customized trainer module for all models."""

    def __init__(self,
                 config: dict,
                 strategy=None):
        """
        Args:
            config: the 'learning_config' part in YAML config file
        """
        # Configurations
        super(BaseTrainer, self).__init__(config)
        self.set_strategy(strategy)
        # Steps and Epochs start from 0
        self.steps = tf.Variable(0, dtype=tf.int64)  # Step must be int64 to use tf.summary
        self.train_steps_per_epoch = None
        self.eval_steps_per_epoch = None
        # Dataset
        self.train_data_loader = None
        self.eval_data_loader = None

        with self.strategy.scope():
            self.set_train_metrics()
            self.set_eval_metrics()

    @property
    def total_train_steps(self):
        if self.train_steps_per_epoch is None: return None
        return self.config["num_epochs"] * self.train_steps_per_epoch

    @property
    def epochs(self):
        if self.train_steps_per_epoch is None: return 1
        return (self.steps.numpy() // self.train_steps_per_epoch) + 1

    # -------------------------------- GET SET -------------------------------------

    @abc.abstractmethod
    def set_train_metrics(self):
        self.train_metrics = {}
        raise NotImplementedError()

    @abc.abstractmethod
    def set_eval_metrics(self):
        self.eval_metrics = {}
        raise NotImplementedError()

    def set_strategy(self, strategy=None):
        if strategy is None:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            self.strategy = tf.distribute.OneDeviceStrategy("/GPU:0") if gpus else \
                tf.distribute.OneDeviceStrategy("/CPU:0")
        else:
            self.strategy = strategy

    def set_train_data_loader(self, train_dataset, train_bs=None):
        """ Set train data loader (MUST). """
        if not train_bs: train_bs = self.config["batch_size"]
        self.global_batch_size = train_bs * self.strategy.num_replicas_in_sync
        self.train_data = train_dataset.create(self.global_batch_size)
        self.train_data_loader = self.strategy.experimental_distribute_dataset(self.train_data)
        self.train_steps_per_epoch = train_dataset.total_steps

    def set_eval_data_loader(self, eval_dataset, eval_bs=None):
        """ Set eval data loader (MUST).
        Eval batch might be significantly greater than train batch """
        if eval_dataset is None:
            self.eval_data = None
            self.eval_data_loader = None
            return
        if not eval_bs: eval_bs = self.config["batch_size"]
        self.eval_data = eval_dataset.create(eval_bs * self.strategy.num_replicas_in_sync)
        self.eval_data_loader = self.strategy.experimental_distribute_dataset(self.eval_data)
        self.eval_steps_per_epoch = eval_dataset.total_steps

    # -------------------------------- CHECKPOINTS -------------------------------------

    def create_checkpoint_manager(self, max_to_keep=10, **kwargs):
        """Create checkpoint management."""
        with self.strategy.scope():
            self.ckpt = tf.train.Checkpoint(steps=self.steps, **kwargs)
            checkpoint_dir = os.path.join(self.config["outdir"], "checkpoints")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            self.ckpt_manager = tf.train.CheckpointManager(
                self.ckpt, checkpoint_dir, max_to_keep=max_to_keep)

    def save_checkpoint(self):
        """Save checkpoint."""
        with self.strategy.scope():
            self.ckpt_manager.save()
            self.train_progbar.set_postfix_str("Successfully Saved Checkpoint")

    def load_checkpoint(self):
        """Load checkpoint."""
        with self.strategy.scope():
            if self.ckpt_manager.latest_checkpoint:
                self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

    def save_model_weights(self):
        """ Save the latest model's weights at each save_interval_steps """
        pass

    # -------------------------------- RUNNING -------------------------------------

    def _finished(self):
        if self.train_steps_per_epoch is None:
            return False
        return self.steps.numpy() >= self.total_train_steps

    def run(self):
        """Run training."""
        if self.steps.numpy() > 0: tf.print("Resume training ...")

        self.train_progbar = tqdm(
            initial=self.steps.numpy(), unit="batch", total=self.total_train_steps,
            position=0, leave=True,
            bar_format="{desc} |%s{bar:20}%s{r_bar}" % (Fore.GREEN, Fore.RESET),
            desc="[Train]"
        )

        while not self._finished():
            self._train_epoch()

        # save when training is done
        self.save_checkpoint()
        self.save_model_weights()

        self.train_progbar.close()
        print("> Finish training")

    def _train_epoch(self):
        """Train model one epoch."""
        train_iterator = iter(self.train_data_loader)
        train_steps = 0
        while True:
            try:
                self._train_function(train_iterator)  # Run train step
            except StopIteration:
                break
            except tf.errors.OutOfRangeError:
                break
            except Exception as e:
                raise e

            # Update steps
            self.steps.assign_add(1)
            self.train_progbar.update(1)
            train_steps += 1

            # Run save checkpoint
            self._check_save_interval()

            # Print epoch info
            self.train_progbar.set_description_str(
                f"[Train] [Epoch {self.epochs}/{self.config['num_epochs']}]")

            # Print train info to progress bar
            self._print_train_metrics(self.train_progbar)

            # Run logging
            self._check_log_interval()

            # Run evaluation
            self._check_eval_interval()

        self.train_steps_per_epoch = train_steps
        self.train_progbar.total = self.total_train_steps
        self.train_progbar.refresh()

    @tf.function
    def _train_function(self, iterator):
        batch = next(iterator)
        self.strategy.run(self._train_step, args=(batch,))

    @abc.abstractmethod
    def _train_step(self, batch):
        """ One step training. Does not return anything"""
        raise NotImplementedError()

    def _eval_epoch(self):
        """One epoch evaluation."""
        if not self.eval_data_loader: return

        print("\n> Start evaluation ...")

        for metric in self.eval_metrics.keys():
            self.eval_metrics[metric].reset_states()

        eval_progbar = tqdm(
            initial=0, total=self.eval_steps_per_epoch, unit="batch",
            position=0, leave=True,
            bar_format="{desc} |%s{bar:20}%s{r_bar}" % (Fore.BLUE, Fore.RESET),
            desc=f"[Eval] [Step {self.steps.numpy()}]"
        )
        eval_iterator = iter(self.eval_data_loader)
        eval_steps = 0

        while True:
            try:
                self._eval_function(eval_iterator)  # Run eval step
            except StopIteration:
                break
            except tf.errors.OutOfRangeError:
                break
            except Exception as e:
                raise e

            # Update steps
            eval_progbar.update(1)
            eval_steps += 1

            # Print eval info to progress bar
            self._print_eval_metrics(eval_progbar)

        self.eval_steps_per_epoch = eval_steps
        eval_progbar.close()
        # Write to tensorboard
        self._write_to_tensorboard(self.eval_metrics, self.steps, stage="eval")

        print("> End evaluation ...")

    @tf.function
    def _eval_function(self, iterator):
        batch = next(iterator)
        self.strategy.run(self._eval_step, args=(batch,))

    @abc.abstractmethod
    def _eval_step(self, batch):
        """One eval step. Does not return anything"""
        raise NotImplementedError()

    @abc.abstractmethod
    def compile(self, *args, **kwargs):
        """ Function to initialize models and optimizers """
        raise NotImplementedError()

    def fit(self, train_dataset, eval_dataset=None, train_bs=None, eval_bs=None):
        """ Function run start training, including executing "run" func """
        self.set_train_data_loader(train_dataset, train_bs)
        self.set_eval_data_loader(eval_dataset, eval_bs)
        self.load_checkpoint()
        self.run()

    # -------------------------------- LOGGING -------------------------------------

    def _check_log_interval(self):
        """Save log interval."""
        if self.steps % self.config["log_interval_steps"] == 0 or \
                self.steps >= self.total_train_steps:
            self._write_to_tensorboard(self.train_metrics, self.steps, stage="train")
            """Reset train metrics after save it to tensorboard."""
            for metric in self.train_metrics.keys():
                self.train_metrics[metric].reset_states()

    def _check_save_interval(self):
        """Save log interval."""
        if self.steps % self.config["save_interval_steps"] == 0 or \
                self.steps >= self.total_train_steps:
            self.save_checkpoint()
            self.save_model_weights()

    def _check_eval_interval(self):
        """Save log interval."""
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    # -------------------------------- UTILS -------------------------------------

    def _print_train_metrics(self, progbar):
        result_dict = {}
        for key, value in self.train_metrics.items():
            result_dict[f"{key}"] = str(value.result().numpy())
        progbar.set_postfix(result_dict)

    def _print_eval_metrics(self, progbar):
        result_dict = {}
        for key, value in self.eval_metrics.items():
            result_dict[f"{key}"] = str(value.result().numpy())
        progbar.set_postfix(result_dict)

    # -------------------------------- END -------------------------------------


class BaseTester(BaseRunner):
    """ Customized tester module for all models
    This tester model will write results to test.tsv file in outdir
    After writing finished, it will calculate testing metrics
    """

    def __init__(self, config: dict, output_name: str = "test"):
        super(BaseTester, self).__init__(config)
        self.test_data_loader = None
        self.processed_records = 0

        self.output_file_path = os.path.join(self.config["outdir"], f"{output_name}.tsv")
        self.test_metrics = {
            "beam_wer": ErrorRate(func=wer, name="test_beam_wer", dtype=tf.float32),
            "beam_cer": ErrorRate(func=cer, name="test_beam_cer", dtype=tf.float32),
            "beam_lm_wer": ErrorRate(func=wer, name="test_beam_lm_wer", dtype=tf.float32),
            "beam_lm_cer": ErrorRate(func=cer, name="test_beam_lm_cer", dtype=tf.float32),
            "greed_wer": ErrorRate(func=wer, name="test_greed_wer", dtype=tf.float32),
            "greed_cer": ErrorRate(func=cer, name="test_greed_cer", dtype=tf.float32)
        }

    def set_output_file(self):
        if os.path.exists(self.output_file_path):
            with open(self.output_file_path, "r", encoding="utf-8") as out:
                self.processed_records = get_num_batches(
                    len(out.read().splitlines()) - 1,
                    batch_size=1
                )
        else:
            with open(self.output_file_path, "w") as out:
                out.write("PATH\tGROUNDTRUTH\tGREEDY\tBEAMSEARCH\tBEAMSEARCHLM\n")

    def set_test_data_loader(self, test_dataset):
        """Set train data loader (MUST)."""
        self.test_data_loader = test_dataset.create(batch_size=1)

    # -------------------------------- RUNNING -------------------------------------

    def compile(self, trained_model: tf.keras.Model):
        """ Set loaded trained model """
        self.model = trained_model

    def run(self, test_dataset):
        self.set_output_file()
        self.set_test_data_loader(test_dataset)
        self._test_epoch()
        self._finish()

    def _test_epoch(self):
        if self.processed_records > 0:
            self.test_data_loader = self.test_data_loader.skip(self.processed_records)
        progbar = tqdm(initial=self.processed_records, total=None,
                       unit="batch", position=0, desc="[Test]")
        test_iter = iter(self.test_data_loader)
        while True:
            try:
                decoded = self._test_function(test_iter)
            except StopIteration:
                break
            except tf.errors.OutOfRangeError:
                break

            decoded = [d.numpy() for d in decoded]
            self._append_to_file(*decoded)
            progbar.update(1)

        progbar.close()

    @tf.function
    def _test_function(self, iterator):
        batch = next(iterator)
        return self._test_step(batch)

    @tf.function(experimental_relax_shapes=True)
    def _test_step(self, batch):
        """
        One testing step
        Args:
            batch: a step fed from test dataset

        Returns:
            (file_paths, groundtruth, greedy, beamsearch, beamsearch_lm) each has shape [B]
        """
        file_paths, features, _, labels, _, _ = batch

        labels = self.model.text_featurizer.iextract(labels)
        greed_pred = self.model.recognize(features)
        if self.model.text_featurizer.decoder_config["beam_width"] > 0:
            beam_pred = self.model.recognize_beam(features=features, lm=False)
            beam_lm_pred = self.model.recognize_beam(features=features, lm=True)
        else:
            beam_pred = beam_lm_pred = tf.constant([""], dtype=tf.string)

        return file_paths, labels, greed_pred, beam_pred, beam_lm_pred

    # -------------------------------- UTILS -------------------------------------

    def _finish(self):
        tf.print("\n> Calculating evaluation metrics ...")
        with open(self.output_file_path, "r", encoding="utf-8") as out:
            lines = out.read().splitlines()
            lines = lines[1:]  # skip header

        for line in lines:
            line = line.split("\t")
            labels, greed_pred, beam_pred, beam_lm_pred = line[1], line[2], line[3], line[4]
            labels = tf.convert_to_tensor([labels], dtype=tf.string)
            greed_pred = tf.convert_to_tensor([greed_pred], dtype=tf.string)
            beam_pred = tf.convert_to_tensor([beam_pred], dtype=tf.string)
            beam_lm_pred = tf.convert_to_tensor([beam_lm_pred], dtype=tf.string)
            # Update metrics
            self.test_metrics["greed_wer"].update_state(greed_pred, labels)
            self.test_metrics["greed_cer"].update_state(greed_pred, labels)
            self.test_metrics["beam_wer"].update_state(beam_pred, labels)
            self.test_metrics["beam_cer"].update_state(beam_pred, labels)
            self.test_metrics["beam_lm_wer"].update_state(beam_lm_pred, labels)
            self.test_metrics["beam_lm_cer"].update_state(beam_lm_pred, labels)

        tf.print("Test results:")
        tf.print("G_WER =", self.test_metrics["greed_wer"].result())
        tf.print("G_CER =", self.test_metrics["greed_cer"].result())
        tf.print("B_WER =", self.test_metrics["beam_wer"].result())
        tf.print("B_CER =", self.test_metrics["beam_cer"].result())
        tf.print("BLM_WER =", self.test_metrics["beam_lm_wer"].result())
        tf.print("BLM_CER =", self.test_metrics["beam_lm_cer"].result())

    def _append_to_file(self,
                        file_path: np.ndarray,
                        groundtruth: np.ndarray,
                        greedy: np.ndarray,
                        beamsearch: np.ndarray,
                        beamsearch_lm: np.ndarray):
        file_path = bytes_to_string(file_path)
        groundtruth = bytes_to_string(groundtruth)
        greedy = bytes_to_string(greedy)
        beamsearch = bytes_to_string(beamsearch)
        beamsearch_lm = bytes_to_string(beamsearch_lm)
        with open(self.output_file_path, "a", encoding="utf-8") as out:
            for i, path in enumerate(file_path):
                line = f"{groundtruth[i]}\t{greedy[i]}\t{beamsearch[i]}\t{beamsearch_lm[i]}"
                out.write(f"{path.strip()}\t{line}\n")

    # -------------------------------- END -------------------------------------
