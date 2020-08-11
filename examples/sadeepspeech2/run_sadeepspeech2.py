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
from tiramisu_asr.utils import setup_environment

setup_environment()
import tensorflow as tf

from tiramisu_asr.configs.user_config import UserConfig
from tiramisu_asr.datasets.asr_dataset import ASRTFRecordDataset, ASRSliceDataset
from tiramisu_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tiramisu_asr.featurizers.text_featurizers import TextFeaturizer
from tiramisu_asr.runners.ctc_runners import CTCTrainer
from tiramisu_asr.runners.base_runners import BaseTester
from ctc_decoders import Scorer
from model import SelfAttentionDS2
from optimizer import create_optimizer

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")


def main():
    modes = ["train", "test"]

    tf.keras.backend.clear_session()

    parser = argparse.ArgumentParser(prog="Self Attention DS2")

    parser.add_argument("--mode", "-m", type=str, default=None,
                        help=f"Mode in {modes}")

    parser.add_argument("--config", "-c", type=str, default=DEFAULT_YAML,
                        help="The file path of model configuration file")

    parser.add_argument("--mixed_precision", type=bool, default=False,
                        help="Whether to use mixed precision training")

    parser.add_argument("--max_ckpts", type=int, default=10,
                        help="Max number of checkpoints to keep")

    parser.add_argument("--saved", type=str, default=None,
                        help="Path to saved model")

    parser.add_argument("--tfrecords", type=bool, default=False,
                        help="Whether to use tfrecords")

    parser.add_argument("--eval_train_ratio", type=int, default=1,
                        help="ratio between train batch size and eval batch size")

    def run(args):
        assert args.mode in modes, f"Mode must in {modes}"

        config = UserConfig(DEFAULT_YAML, args.config, learning=True)
        speech_featurizer = TFSpeechFeaturizer(config["speech_config"])
        text_featurizer = TextFeaturizer(config["decoder_config"])

        if args.mode == "train":
            tf.random.set_seed(2020)

            if args.mixed_precision:
                policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
                tf.keras.mixed_precision.experimental.set_policy(policy)
                print("Enabled mixed precision training")

            ctc_trainer = CTCTrainer(speech_featurizer, text_featurizer,
                                     config["learning_config"]["running_config"],
                                     args.mixed_precision)

            if args.tfrecords:
                train_dataset = ASRTFRecordDataset(
                    config["learning_config"]["dataset_config"]["train_paths"],
                    config["learning_config"]["dataset_config"]["tfrecords_dir"],
                    speech_featurizer, text_featurizer, "train",
                    augmentations=config["learning_config"]["augmentations"], shuffle=True,
                )
                eval_dataset = ASRTFRecordDataset(
                    config["learning_config"]["dataset_config"]["eval_paths"],
                    config["learning_config"]["dataset_config"]["tfrecords_dir"],
                    speech_featurizer, text_featurizer, "eval", shuffle=False
                )
            else:
                train_dataset = ASRSliceDataset(
                    stage="train", speech_featurizer=speech_featurizer,
                    text_featurizer=text_featurizer,
                    data_paths=config["learning_config"]["dataset_config"]["train_paths"],
                    augmentations=config["learning_config"]["augmentations"], shuffle=True,
                )
                eval_dataset = ASRSliceDataset(
                    stage="eval", speech_featurizer=speech_featurizer,
                    text_featurizer=text_featurizer,
                    data_paths=config["learning_config"]["dataset_config"]["eval_paths"],
                    shuffle=False
                )

            # Build DS2 model
            with ctc_trainer.strategy.scope():
                satt_ds2_model = SelfAttentionDS2(
                    input_shape=speech_featurizer.compute_feature_shape(),
                    arch_config=config["model_config"],
                    num_classes=text_featurizer.num_classes
                )
                satt_ds2_model._build(speech_featurizer.compute_feature_shape())
                satt_ds2_model.summary(line_length=150)
                optimizer = create_optimizer(
                    name=config["learning_config"]["optimizer_config"]["name"],
                    d_model=config["model_config"]["att"]["head_size"],
                    **config["learning_config"]["optimizer_config"]["config"]
                )
            # Compile
            ctc_trainer.compile(satt_ds2_model, optimizer, max_to_keep=args.max_ckpts)

            ctc_trainer.fit(train_dataset, eval_dataset, args.eval_train_ratio)

        elif args.mode == "test":
            tf.random.set_seed(0)
            assert args.saved

            text_featurizer.add_scorer(
                Scorer(**text_featurizer.decoder_config["lm_config"],
                       vocabulary=text_featurizer.vocab_array))

            # Build DS2 model
            satt_ds2_model = SelfAttentionDS2(
                input_shape=speech_featurizer.compute_feature_shape(),
                arch_config=config["model_config"],
                num_classes=text_featurizer.num_classes
            )
            satt_ds2_model._build(speech_featurizer.compute_feature_shape())
            satt_ds2_model.summary(line_length=150)
            satt_ds2_model.load_weights(args.saved)
            satt_ds2_model.add_featurizers(speech_featurizer, text_featurizer)

            if args.tfrecords:
                test_dataset = ASRTFRecordDataset(
                    config["learning_config"]["dataset_config"]["test_paths"],
                    config["learning_config"]["dataset_config"]["tfrecords_dir"],
                    speech_featurizer, text_featurizer, "test",
                    augmentations=config["learning_config"]["augmentations"], shuffle=False
                )
            else:
                test_dataset = ASRSliceDataset(
                    stage="test", speech_featurizer=speech_featurizer,
                    text_featurizer=text_featurizer,
                    data_paths=config["learning_config"]["dataset_config"]["test_paths"],
                    augmentations=config["learning_config"]["augmentations"], shuffle=False
                )

            ctc_tester = BaseTester(config=config["learning_config"]["running_config"])
            ctc_tester.compile(satt_ds2_model)
            ctc_tester.run(test_dataset)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
