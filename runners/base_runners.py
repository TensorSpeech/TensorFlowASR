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

import os
import abc
import time
import logging
from tqdm import tqdm

import tensorflow as tf

from utils.utils import preprocess_paths


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

    def __init__(self,
                 train_steps_per_epoch: int,
                 config: dict):
        # Configurations
        # train_steps_per_epoch = math.ceil(num_samples / batch_size)
        super(BaseTrainer, self).__init__(config)
        # Steps and Epochs start from 0
        self.steps = tf.Variable(0, dtype=tf.int64)
        self.epochs = tf.Variable(0, dtype=tf.int32)
        self.train_steps_per_epoch = train_steps_per_epoch
        self.max_global_steps = int(config["num_epochs"]) * train_steps_per_epoch
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
        self.tqdm = tqdm(initial=self.steps.numpy(), total=self.max_global_steps, desc="[train]")
        while self.steps.numpy() < self.max_global_steps:
            self._train_epoch()

        self.tqdm.close()
        logging.info("Finish training.")

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
        self.ckpt_manager.save()

    def load_checkpoint(self):
        """Load checkpoint."""
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

    def _update_time_metrics(self, start, end):
        self.time_metrics["training_hours"].update_state((end - start) / 3600)

    def _train_epoch(self):
        """Train model one epoch."""
        for batch in self.train_data_loader:
            # one step training
            start = time.time()
            self._train_step(batch)
            self._post_train_step()

            # Update steps
            self.steps.assign_add(1)
            self.tqdm.update(1)

            # check interval
            self._check_log_interval()
            self._check_eval_interval()
            self._check_save_interval()
            self._update_time_metrics(start, time.time())
            self._check_time_interval()

        # Update
        self.epochs.assign_add(1)
        logging.info(f"(Steps: {self.steps.numpy()}) "
                     f"Finished {self.epochs.numpy()} epoch training "
                     f"({self.train_steps_per_epoch} steps per epoch).")

    def _post_train_step(self):
        pass

    @abc.abstractmethod
    def _eval_epoch(self):
        """One epoch evaluation."""
        pass

    @abc.abstractmethod
    def _train_step(self, batch):
        """One step training."""
        pass

    @abc.abstractmethod
    def _eval_step(self, batch):
        """One eval step."""
        pass

    @abc.abstractmethod
    def compile(self, model, optimizer):
        pass

    @abc.abstractmethod
    def fit(self, train_dataset, eval_dataset, max_to_keep=10):
        pass

    @abc.abstractmethod
    def _check_log_interval(self):
        """Save log interval."""
        pass

    def _check_eval_interval(self):
        """Evaluation interval step."""
        if (self.steps % self.config["eval_interval_steps"] == 0) \
                or (self.steps.numpy() >= self.max_global_steps):
            self._eval_epoch()

    def _check_save_interval(self):
        """Save interval checkpoint."""
        if (self.steps % self.config["save_interval_steps"] == 0) \
                or (self.steps.numpy() >= self.max_global_steps):
            self.save_checkpoint()
            logging.info(f"Successfully saved checkpoint @ {self.steps.numpy()} steps.")

    def _check_time_interval(self):
        if (self.steps % self.config["time_interval_steps"] == 0) \
                or (self.steps.numpy() >= self.max_global_steps):
            self._write_to_tensorboard(self.time_metrics, self.steps, stage="train")


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

        logging.info(f"Finished testing ({self.test_steps_per_epoch} steps per epoch).")
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
