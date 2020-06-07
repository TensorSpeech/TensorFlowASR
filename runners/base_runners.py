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

from __future__ import absolute_import

import abc
import os
import time

import tensorflow as tf
from tqdm import tqdm

from utils.utils import preprocess_paths, update_total


class BaseRunner(metaclass=abc.ABCMeta):
    """ Customized runner module for all models """

    def __init__(self, config: dict):
        """
        config: {
            "num_epochs": int,
            "outdir": str,
            "eval_interval_steps": int,
            "save_interval_steps": int
        }
        """
        self.config = config
        self.config["outdir"] = preprocess_paths(self.config["outdir"])
        # Writer
        self.writer = tf.summary.create_file_writer(
            os.path.join(config["outdir"], "tensorboard")
        )

    def _write_to_tensorboard(self,
                              list_metrics: dict,
                              step: any,
                              stage: str = "train"):
        """Write variables to tensorboard."""
        with self.writer.as_default():
            for key, value in list_metrics.items():
                tf.summary.scalar(stage + "/" + key, value.result(), step=step)
                self.writer.flush()


class BaseTrainer(BaseRunner):
    """Customized trainer module for all models."""

    def __init__(self, config: dict):
        # Configurations
        super(BaseTrainer, self).__init__(config)
        # Steps and Epochs start from 0
        self.steps = tf.Variable(0, dtype=tf.int64)
        self.epochs = tf.Variable(0)
        self.train_steps_per_epoch = tf.Variable(0, dtype=tf.int64)
        self.finish_training = tf.Variable(False, dtype=tf.bool)
        # Time metric
        self.time_metrics = {
            "training_hours": tf.keras.metrics.Sum("training_hours", dtype=tf.float32)
        }
        # Strategy
        self.mirror_strategy = tf.distribute.MirroredStrategy()
        # Checkpoints
        self.ckpt = None
        self.ckpt_manager = None
        # Datasets
        self.train_data_loader = None

    def set_train_data_loader(self, train_dataset):
        """Set train data loader (MUST)."""
        self.train_data_loader = train_dataset

    def get_train_data_loader(self):
        """Get train data loader."""
        return self.train_data_loader

    def set_eval_data_loader(self, eval_dataset):
        """Set eval data loader (MUST)."""
        self.eval_data_loader = eval_dataset

    def get_eval_data_loader(self):
        """Get eval data loader."""
        return self.eval_data_loader

    def run(self):
        """Run training."""
        if self.steps.numpy() > 0: tf.print("Resume training ...")

        self.tqdm = tqdm(initial=self.steps.numpy(), total=None, desc="[train]", unit="step")

        while True:
            if self.finish_training.numpy(): break
            start = time.time()
            self._train_epoch()
            self.time_metrics["training_hours"].update_state((time.time() - start) / 3600)
            self._write_to_tensorboard(self.time_metrics, self.steps, stage="train")
            update_total(self.tqdm, self.config["num_epochs"] * self.train_steps_per_epoch.numpy())

        self.tqdm.close()

        tf.print("Finish training.")

    def create_checkpoint_manager(self,
                                  max_to_keep=10,
                                  **kwargs):
        """Create checkpoint management."""
        self.ckpt = tf.train.Checkpoint(steps=self.steps, epochs=self.epochs, **kwargs)
        checkpoint_dir = os.path.join(self.config["outdir"], "checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_dir, max_to_keep=max_to_keep)

    def save_checkpoint(self):
        """Save checkpoint."""
        tf.py_function(self.ckpt_manager.save, [], [tf.string])
        tf.print("Successfully saved checkpoint at", self.steps, "steps.")

    def load_checkpoint(self):
        """Load checkpoint."""
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

    @tf.function
    def _train_epoch(self):
        """Train model one epoch."""
        for spe, batch in self.train_data_loader.enumerate(start=1):
            # one step training
            self._train_step(batch)

            # Update steps
            self.steps.assign_add(1)
            self.train_steps_per_epoch.assign(spe)
            tf.py_function(lambda: self.tqdm.update(1), [], [])

            # check interval, must pass updated steps due to unrecognized updated self attributes
            self._check_log_interval()
            self._check_save_interval()
            self._check_eval_interval()

        # Update
        self.epochs.assign_add(1)
        if self.steps >= self.config["num_epochs"] * self.train_steps_per_epoch: self.finish_training.assign(True)
        # Logging
        tf.print("Finish epochs", self.epochs, "at steps", self.steps, "with", self.train_steps_per_epoch, "steps per epoch")

    @abc.abstractmethod
    def _train_step(self, batch):
        """One step training."""
        pass

    @abc.abstractmethod
    def _eval_epoch(self):
        """One epoch evaluation."""
        pass

    @abc.abstractmethod
    def _eval_step(self, batch):
        """One eval step."""
        pass

    @abc.abstractmethod
    def compile(self, *args, **kwargs):
        """ Function to initialize models and optimizers """
        pass

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """ Function run start training, including executing "run" func """
        pass

    @abc.abstractmethod
    def _exec_log_interval(self):
        """Save log interval."""
        pass

    def _check_log_interval(self):
        """Save log interval."""
        if tf.logical_or(
                tf.equal(tf.math.mod(self.steps, self.config["log_interval_steps"]), 0),
                self.finish_training):
            self._exec_log_interval()

    def _check_eval_interval(self):
        """Save interval checkpoint."""
        if tf.logical_or(
                tf.equal(tf.math.mod(self.steps, self.config["eval_interval_steps"]), 0),
                self.finish_training):
            self._eval_epoch()

    def _check_save_interval(self):
        """Save interval checkpoint."""
        if tf.logical_or(
                tf.equal(tf.math.mod(self.steps, self.config["save_interval_steps"]), 0),
                self.finish_training):
            self.save_checkpoint()


class BaseLoader(metaclass=abc.ABCMeta):
    """ Based class for loading saved model """

    def __init__(self,
                 saved_path: str,
                 yaml_arch_path: str,
                 from_weights: bool = False):
        self.saved_path = preprocess_paths(saved_path)
        self.yaml_arch_path = preprocess_paths(yaml_arch_path)
        self.from_weights = from_weights

    def load_model(self):
        try:
            self.model = tf.saved_model.load(self.saved_path)
        except Exception as e:
            raise Exception(e)

    def load_model_from_weights(self):
        try:
            import yaml
            with open(preprocess_paths(self.yaml_arch_path), "r", encoding="utf-8") as arch:
                self.model = tf.keras.models.model_from_yaml(arch.read())
            self.model.load_weights(self.saved_path)
        except Exception as e:
            raise Exception(e)

    def compile(self):
        if self.from_weights:
            self.load_model_from_weights()
        else:
            self.load_model()


class BaseTester(BaseLoader, BaseRunner):
    """ Customized tester module for all models """

    def __init__(self,
                 config: dict,
                 saved_path: str,
                 yaml_arch_path: str,
                 from_weights: bool = False):
        BaseLoader.__init__(self, saved_path, yaml_arch_path, from_weights)
        BaseRunner.__init__(self, config)
        self.test_data_loader = None

    def set_test_data_loader(self, test_dataset):
        """Set train data loader (MUST)."""
        self.test_data_loader = tqdm(test_dataset, desc="[test]")

    def get_test_data_loader(self):
        """Get train data loader."""
        return self.test_data_loader

    def run(self, test_dataset):
        self.set_test_data_loader(test_dataset)
        """ Run testing """
        self.test_steps_per_epoch = 0
        for self.test_steps_per_epoch, batch in enumerate(self.test_data_loader, 1):
            self._test_step(batch)
            # Print postfix
            self._post_process_step()

        tf.print("Finished testing with", self.test_steps_per_epoch, "steps per epoch.")
        self.finish()

    @abc.abstractmethod
    def _post_process_step(self):
        pass

    @abc.abstractmethod
    def _test_step(self, batch):
        """ One testing step"""
        pass

    @abc.abstractmethod
    def finish(self):
        pass


class BaseInferencer(BaseLoader):
    """ Customized inferencer module for all models """

    @abc.abstractmethod
    def preprocess(self, *args, **kwargs):
        """ Preprocessing stage """
        pass

    @abc.abstractmethod
    def postprocess(self, *args, **kwargs):
        """ Postprocessing stage """
        pass

    @abc.abstractmethod
    def infer(self, inputs):
        """ Function for infering result """
        pass
