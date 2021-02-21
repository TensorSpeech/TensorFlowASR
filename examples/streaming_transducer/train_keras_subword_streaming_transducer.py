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
from tensorflow_asr.utils import setup_environment, setup_strategy

setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="Conformer Training")

parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")

parser.add_argument("--max_ckpts", type=int, default=10, help="Max number of checkpoints to keep")

parser.add_argument("--tfrecords", default=False, action="store_true", help="Whether to use tfrecords")

parser.add_argument("--tbs", type=int, default=None, help="Train batch size per replica")

parser.add_argument("--ebs", type=int, default=None, help="Evaluation batch size per replica")

parser.add_argument("--spx", type=int, default=1, help="Steps per execution for maximizing performance")

parser.add_argument("--metadata_prefix", type=str, default=None, help="Path to file containing metadata")

parser.add_argument("--devices", type=int, nargs="*", default=[0], help="Devices' ids to apply distributed training")

parser.add_argument("--mxp", default=False, action="store_true", help="Enable mixed precision")

parser.add_argument("--subwords", type=str, default=None, help="Path to file that stores generated subwords")

parser.add_argument("--subwords_corpus", nargs="*", type=str, default=[], help="Transcript files for generating subwords")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

strategy = setup_strategy(args.devices)

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.keras import ASRTFRecordDatasetKeras, ASRSliceDatasetKeras
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import SubwordFeaturizer
from tensorflow_asr.models.keras.streaming_transducer import StreamingTransducer

config = Config(args.config)
speech_featurizer = TFSpeechFeaturizer(config.speech_config)

if args.subwords and os.path.exists(args.subwords):
    print("Loading subwords ...")
    text_featurizer = SubwordFeaturizer.load_from_file(config.decoder_config, args.subwords)
else:
    print("Generating subwords ...")
    text_featurizer = SubwordFeaturizer.build_from_corpus(
        config.decoder_config,
        corpus_files=args.subwords_corpus
    )
    text_featurizer.save_to_file(args.subwords)

if args.tfrecords:
    train_dataset = ASRTFRecordDatasetKeras(
        speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
        **vars(config.learning_config.train_dataset_config),
        indefinite=True
    )
    eval_dataset = ASRTFRecordDatasetKeras(
        speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
        **vars(config.learning_config.eval_dataset_config)
    )
    # Update metadata calculated from both train and eval datasets
    train_dataset.load_metadata(args.metadata_prefix)
    eval_dataset.load_metadata(args.metadata_prefix)
    # Use dynamic length
    speech_featurizer.reset_length()
    text_featurizer.reset_length()
else:
    train_dataset = ASRSliceDatasetKeras(
        speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
        **vars(config.learning_config.train_dataset_config),
        indefinite=True
    )
    eval_dataset = ASRSliceDatasetKeras(
        speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
        **vars(config.learning_config.eval_dataset_config),
        indefinite=True
    )

global_batch_size = config.learning_config.running_config.batch_size
global_batch_size *= strategy.num_replicas_in_sync

train_data_loader = train_dataset.create(global_batch_size)
eval_data_loader = eval_dataset.create(global_batch_size)

with strategy.scope():
    # build model
    streaming_transducer = StreamingTransducer(
        **config.model_config,
        vocabulary_size=text_featurizer.num_classes
    )
    streaming_transducer._build(speech_featurizer.shape)
    streaming_transducer.summary(line_length=150)

    optimizer = tf.keras.optimizers.get(config.learning_config.optimizer_config)

    streaming_transducer.compile(
        optimizer=optimizer,
        experimental_steps_per_execution=args.spx,
        global_batch_size=global_batch_size,
        blank=text_featurizer.blank
    )

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(**config.learning_config.running_config.checkpoint),
    tf.keras.callbacks.experimental.BackupAndRestore(config.learning_config.running_config.states_dir),
    tf.keras.callbacks.TensorBoard(**config.learning_config.running_config.tensorboard)
]

streaming_transducer.fit(
    train_data_loader, epochs=config.learning_config.running_config.num_epochs,
    validation_data=eval_data_loader, callbacks=callbacks,
    steps_per_epoch=train_dataset.total_steps, validation_steps=eval_dataset.total_steps
)
