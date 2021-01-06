# Copyright 2021 M. Yusuf Sarıgöz (@monatis)
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
import math
import argparse
from tensorflow_asr.utils import setup_environment, setup_tpu

setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="Conformer Training")

parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")

parser.add_argument("--max_ckpts", type=int, default=10, help="Max number of checkpoints to keep")

parser.add_argument("--tfrecords_shards", type=int, default=16, help="Number of tfrecords shards")

parser.add_argument("--sentence_piece", default=False, action="store_true", help="Whether to use `SentencePiece` model")

parser.add_argument("--bs", type=int, default=2, help="Common training and evaluation batch size per TPU core")

parser.add_argument("--tpu_address", type=str, default=None, help="TPU address. Leave None on Colab")

parser.add_argument("--max_lengths_path", type=str, default="~", help="Path to file containing max lengths. Will be computed if not exists")

parser.add_argument("--mxp", default=False, action="store_true", help="Enable mixed precision")

parser.add_argument("--cache", default=False, action="store_true", help="Enable caching for dataset")

parser.add_argument("--subwords", type=str, default=None, help="Path to file that stores generated subwords")

parser.add_argument("--subwords_corpus", nargs="*", type=str, default=[], help="Transcript files for generating subwords")

parser.add_argument("--bfs", type=int, default=100, help="Buffer size for shuffling")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.asr_dataset import ASRTFRecordDataset
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import SubwordFeaturizer, SentencePieceFeaturizer
from tensorflow_asr.runners.transducer_runners import TransducerTrainer
from tensorflow_asr.models.conformer import Conformer
from tensorflow_asr.optimizers.schedules import TransformerSchedule

config = Config(args.config, learning=True)
speech_featurizer = TFSpeechFeaturizer(config.speech_config)

if args.sentence_piece:
    print("Loading SentencePiece model ...")
    text_featurizer = SentencePieceFeaturizer.load_from_file(config.decoder_config, args.subwords)
elif args.subwords and os.path.exists(args.subwords):
    print("Loading subwords ...")
    text_featurizer = SubwordFeaturizer.load_from_file(config.decoder_config, args.subwords)
else:
    print("Generating subwords ...")
    text_featurizer = SubwordFeaturizer.build_from_corpus(
        config.decoder_config,
        corpus_files=args.subwords_corpus
    )
    text_featurizer.save_to_file(args.subwords)

train_dataset = ASRTFRecordDataset(
    data_paths=config.learning_config.dataset_config.train_paths,
    tfrecords_dir=config.learning_config.dataset_config.tfrecords_dir,
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    augmentations=config.learning_config.augmentations,
    tfrecords_shards=args.tfrecords_shards,
    stage="train", cache=args.cache,
    shuffle=True, buffer_size=args.bfs, enable_tpu=True,
)
train_dataset.compute_max_lengths(args.max_lengths_path)

eval_dataset = ASRTFRecordDataset(
    data_paths=config.learning_config.dataset_config.eval_paths,
    tfrecords_dir=config.learning_config.dataset_config.tfrecords_dir,
    tfrecords_shards=args.tfrecords_shards,
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    stage="eval", cache=args.cache,
    shuffle=True, buffer_size=args.bfs, enable_tpu=True,
)
eval_dataset.compute_max_lengths(args.max_lengths_path)

# Get maximum(max_input_length, max_label_length, max_prediction_length) from attributes of train and eval datasets
# and set back the greater values to both datasets.
# Finally, build the model with these static shapes.
max_input_length = max(train_dataset.max_input_length, eval_dataset.max_input_length)
max_label_length = max(train_dataset.max_label_length, eval_dataset.max_label_length)
max_prediction_length = max(train_dataset.max_prediction_length, eval_dataset.max_prediction_length)
train_dataset.max_input_length, train_dataset.max_label_length, train_dataset.max_prediction_length = max_input_length, max_label_length, max_prediction_length
eval_dataset.max_input_length, eval_dataset.max_label_length, eval_dataset.max_prediction_length = max_input_length, max_label_length, max_prediction_length
input_shape = speech_featurizer.shape
input_shape[0] = max_input_length

strategy = setup_tpu(args.tpu_address)

conformer_trainer = TransducerTrainer(
    config=config.learning_config.running_config,
    text_featurizer=text_featurizer, strategy=strategy
)

with conformer_trainer.strategy.scope():
    # build model
    conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
    conformer._build(input_shape, prediction_max_length=max_prediction_length, batch_size=args.bs)
    conformer.summary(line_length=120)

    optimizer = tf.keras.optimizers.Adam(
        TransformerSchedule(
            d_model=conformer.dmodel,
            warmup_steps=config.learning_config.optimizer_config["warmup_steps"],
            max_lr=(0.05 / math.sqrt(conformer.dmodel))
        ),
        beta_1=config.learning_config.optimizer_config["beta1"],
        beta_2=config.learning_config.optimizer_config["beta2"],
        epsilon=config.learning_config.optimizer_config["epsilon"]
    )

conformer_trainer.compile(model=conformer, optimizer=optimizer,
                          max_to_keep=args.max_ckpts)

conformer_trainer.fit(train_dataset, eval_dataset, train_bs=args.bs, eval_bs=args.bs)

