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
from __future__ import absolute_import

import os

import tensorflow as tf

import configs.user_config as config
from configs.user_config import UserConfig
from datasets.asr_dataset import ASRTFRecordDataset
from decoders.ctc_decoders import CTCDecoder
from featurizers.speech_featurizers import SpeechFeaturizer
from featurizers.text_featurizers import TextFeaturizer
from runners.ctc_runners import CTCTrainer, CTCTester

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(config.__file__)), "default_ctc.yml")


def main(parser):
    modes = ["train", "test", "create_tfrecords"]

    parser.add_argument("--mode", "-m", type=str, default=None,
                        help=f"Mode in {modes}")

    parser.add_argument("--config", "-c", type=str, default=DEFAULT_YAML,
                        help="The file path of model configuration file")

    parser.add_argument("--arch", "-a", type=str, default=None,
                        help="Path to the yaml architecture to be exported")

    parser.add_argument("--export", "-e", type=str, default=None,
                        help="Path to the model file to be exported")

    parser.add_argument("--mixed_precision", type=bool, default=False,
                        help="Whether to use mixed precision training")

    parser.add_argument("--from_weights", type=bool, default=False,
                        help="Whether to save or load only weights")

    def run(args):
        assert args.mode in modes, f"Mode must in {modes}"

        config = UserConfig(DEFAULT_YAML, args.config, learning=True)
        speech_featurizer = SpeechFeaturizer(config["speech_config"])
        text_featurizer = TextFeaturizer(config["decoder_config"]["vocabulary"])
        decoder = CTCDecoder(config["decoder_config"], text_featurizer)

        if args.mode == "train":
            tf.random.set_seed(2020)

            if args.mixed_precision:
                policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
                tf.keras.mixed_precision.experimental.set_policy(policy)
                print("Enabled mixed precision training")

            train_dataset = ASRTFRecordDataset(
                config["learning_config"]["dataset_config"]["train_paths"],
                config["learning_config"]["dataset_config"]["tfrecords_dir"],
                speech_featurizer, text_featurizer, "train",
                augmentations=config["learning_config"]["augmentations"], shuffle=True,
            ).create(config["learning_config"]["dataset_config"]["batch_size"]).take(5)

            eval_dataset = ASRTFRecordDataset(
                config["learning_config"]["dataset_config"]["eval_paths"],
                config["learning_config"]["dataset_config"]["tfrecords_dir"],
                speech_featurizer, text_featurizer, "eval", shuffle=False
            ).create(config["learning_config"]["dataset_config"]["batch_size"])

            ctc_trainer = CTCTrainer(speech_featurizer, text_featurizer, decoder,
                                     config["learning_config"]["running_config"], args.mixed_precision)
            ctc_trainer.compile(config["model_config"], config["learning_config"]["optimizer"])

            try:
                ctc_trainer.fit(train_dataset, eval_dataset, max_to_keep=10)
            except KeyboardInterrupt:
                ctc_trainer.save_checkpoint()

            if args.export:
                if args.from_weight:
                    ctc_trainer.model.save_weights(args.export)
                else:
                    ctc_trainer.model.save(args.export)

        elif args.mode == "test":
            tf.random.set_seed(0)
            assert args.export is not None

            test_dataset = ASRTFRecordDataset(
                config["learning_config"]["dataset_config"]["test_paths"],
                config["learning_config"]["dataset_config"]["tfrecords_dir"],
                speech_featurizer, text_featurizer, "test", shuffle=False
            ).create(config["learning_config"]["dataset_config"]["batch_size"])

            ctc_tester = CTCTester(text_featurizer, decoder, config["learning_config"]["running_config"],
                                   args.export, args.arch, from_weights=args.from_weights)
            ctc_tester.compile()
            ctc_tester.run(test_dataset)

        else:
            ASRTFRecordDataset(
                config["learning_config"]["dataset_config"]["train_paths"],
                config["learning_config"]["dataset_config"]["tfrecords_dir"],
                speech_featurizer, text_featurizer, "train",
                augmentations=config["learning_config"]["augmentations"], shuffle=True,
            ).create_tfrecords()

            ASRTFRecordDataset(
                config["learning_config"]["dataset_config"]["eval_paths"],
                config["learning_config"]["dataset_config"]["tfrecords_dir"],
                speech_featurizer, text_featurizer, "eval", shuffle=False
            ).create_tfrecords()

            ASRTFRecordDataset(
                config["learning_config"]["dataset_config"]["test_paths"],
                config["learning_config"]["dataset_config"]["tfrecords_dir"],
                speech_featurizer, text_featurizer, "test", shuffle=False
            ).create_tfrecords()

    return run
