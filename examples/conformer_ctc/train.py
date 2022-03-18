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

from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()

import math
import os

import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.helpers import dataset_helpers, featurizer_helpers
from tensorflow_asr.models.ctc.conformer import Conformer
from tensorflow_asr.optimizers.schedules import TransformerSchedule
from tensorflow_asr.utils import cli_util, file_util

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config_wp.j2")


def main(
    config_path: str = DEFAULT_YAML,
    tfrecords: bool = False,
    bs: int = None,
    spx: int = 1,
    devices: list = None,
    mxp: bool = False,
    pretrained: str = None,
    jit_compile: bool = False,
    ga_steps: int = None,
):
    env_util.setup_seed()
    strategy = env_util.setup_strategy(devices)
    env_util.setup_mxp(mxp=mxp)

    config = Config(config_path)

    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(config=config)

    train_dataset, eval_dataset = dataset_helpers.prepare_training_datasets(
        config=config,
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        tfrecords=tfrecords,
    )

    train_data_loader, eval_data_loader, global_batch_size = dataset_helpers.prepare_training_data_loaders(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        strategy=strategy,
        batch_size=bs,
    )

    with strategy.scope():
        conformer = Conformer(**config.model_config, vocab_size=text_featurizer.num_classes)
        conformer.make(speech_featurizer.shape, batch_size=global_batch_size)
        if pretrained:
            conformer.load_weights(pretrained, by_name=file_util.is_hdf5_filepath(pretrained), skip_mismatch=True)
        optimizer = tf.keras.optimizers.Adam(
            TransformerSchedule(
                d_model=conformer.dmodel,
                warmup_steps=config.learning_config.learning_rate_config.get("warmup_steps"),
                max_lr=(config.learning_config.learning_rate_config.get("max_lr_numerator") / math.sqrt(conformer.dmodel)),
            ),
            **config.learning_config.optimizer_config,
        )
        conformer.compile(
            optimizer=optimizer,
            steps_per_execution=spx,
            blank=text_featurizer.blank,
            jit_compile=jit_compile,
            mxp=mxp,
            ga_steps=ga_steps,
        )
        conformer.summary()

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(**config.learning_config.running_config.checkpoint),
        tf.keras.callbacks.BackupAndRestore(**config.learning_config.running_config.backup_and_restore),
        tf.keras.callbacks.TensorBoard(**config.learning_config.running_config.tensorboard),
    ]

    conformer.fit(
        train_data_loader,
        epochs=config.learning_config.running_config.num_epochs,
        validation_data=eval_data_loader,
        callbacks=callbacks,
        steps_per_epoch=train_dataset.total_steps,
        validation_steps=eval_dataset.total_steps if eval_data_loader else None,
    )


if __name__ == "__main__":
    cli_util.run(main)
