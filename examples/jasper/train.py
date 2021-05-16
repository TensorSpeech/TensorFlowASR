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

import os
import argparse
from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="Jasper Training")

parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")

parser.add_argument("--tfrecords", default=False, action="store_true", help="Whether to use tfrecords")

parser.add_argument("--sentence_piece", default=False, action="store_true", help="Whether to use `SentencePiece` model")

parser.add_argument("--subwords", default=False, action="store_true", help="Use subwords")

parser.add_argument("--bs", type=int, default=None, help="Batch size per replica")

parser.add_argument("--spx", type=int, default=1, help="Steps per execution for maximizing performance")

parser.add_argument("--metadata", type=str, default=None, help="Path to file containing metadata")

parser.add_argument("--static_length", default=False, action="store_true", help="Use static lengths")

parser.add_argument("--devices", type=int, nargs="*", default=[0], help="Devices' ids to apply distributed training")

parser.add_argument("--mxp", default=False, action="store_true", help="Enable mixed precision")

parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

strategy = env_util.setup_strategy(args.devices)

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets import asr_dataset
from tensorflow_asr.featurizers import speech_featurizers, text_featurizers
from tensorflow_asr.models.ctc.jasper import Jasper

config = Config(args.config)
speech_featurizer = speech_featurizers.TFSpeechFeaturizer(config.speech_config)

if args.sentence_piece:
    logger.info("Loading SentencePiece model ...")
    text_featurizer = text_featurizers.SentencePieceFeaturizer(config.decoder_config)
elif args.subwords:
    logger.info("Loading subwords ...")
    text_featurizer = text_featurizers.SubwordFeaturizer(config.decoder_config)
else:
    logger.info("Use characters ...")
    text_featurizer = text_featurizers.CharFeaturizer(config.decoder_config)

if args.tfrecords:
    train_dataset = asr_dataset.ASRTFRecordDataset(
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        **vars(config.learning_config.train_dataset_config),
        indefinite=True
    )
    eval_dataset = asr_dataset.ASRTFRecordDataset(
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        **vars(config.learning_config.eval_dataset_config),
        indefinite=True
    )
else:
    train_dataset = asr_dataset.ASRSliceDataset(
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        **vars(config.learning_config.train_dataset_config),
        indefinite=True
    )
    eval_dataset = asr_dataset.ASRSliceDataset(
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        **vars(config.learning_config.eval_dataset_config),
        indefinite=True
    )

train_dataset.load_metadata(args.metadata)
eval_dataset.load_metadata(args.metadata)

if not args.static_length:
    speech_featurizer.reset_length()
    text_featurizer.reset_length()

global_batch_size = args.bs or config.learning_config.running_config.batch_size
global_batch_size *= strategy.num_replicas_in_sync

train_data_loader = train_dataset.create(global_batch_size)
eval_data_loader = eval_dataset.create(global_batch_size)

with strategy.scope():
    # build model
    jasper = Jasper(**config.model_config, vocabulary_size=text_featurizer.num_classes)
    jasper.make(speech_featurizer.shape, batch_size=global_batch_size)
    if args.pretrained:
        jasper.load_weights(args.pretrained, by_name=True, skip_mismatch=True)
    jasper.summary(line_length=100)
    jasper.compile(
        optimizer=config.learning_config.optimizer_config,
        experimental_steps_per_execution=args.spx,
        global_batch_size=global_batch_size,
        blank=text_featurizer.blank
    )

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(**config.learning_config.running_config.checkpoint),
    tf.keras.callbacks.experimental.BackupAndRestore(config.learning_config.running_config.states_dir),
    tf.keras.callbacks.TensorBoard(**config.learning_config.running_config.tensorboard)
]

jasper.fit(
    train_data_loader,
    epochs=config.learning_config.running_config.num_epochs,
    validation_data=eval_data_loader,
    callbacks=callbacks,
    steps_per_epoch=train_dataset.total_steps,
    validation_steps=eval_dataset.total_steps if eval_data_loader else None
)
