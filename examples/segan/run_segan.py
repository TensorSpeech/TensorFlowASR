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

from tiramisu_asr.runners.segan_runners import SeganTrainer, SeganTester
from tiramisu_asr.datasets.segan_dataset import SeganDataset
from tiramisu_asr.configs.user_config import UserConfig

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")


def main():
    modes = ["train", "test", "save_from_checkpoint"]

    parser = argparse.ArgumentParser(prog="SEGAN")

    parser.add_argument("--mode", "-m", type=str, default="train",
                        help=f"Mode in {modes}")

    parser.add_argument("--config", "-c", type=str, default=DEFAULT_YAML,
                        help="The file path of model configuration file")

    parser.add_argument("--export", "-e", type=str, default=None,
                        help="Path to the model file to be exported")

    parser.add_argument("--mixed_precision", type=bool, default=False,
                        help="Whether to use mixed precision training")

    parser.add_argument("--from_weights", type=bool, default=False,
                        help="Whether to save or load only weights")

    parser.add_argument("--max_ckpts", type=int, default=10,
                        help="Max number of checkpoints to keep")

    parser.add_argument("--saved_model", type=str, default=None,
                        help="Path to saved model")

    def run(args):
        assert args.mode in modes, f"Mode must in {modes}"

        config = UserConfig(DEFAULT_YAML, args.config, learning=True)

        if args.mode == "train":
            tf.random.set_seed(2020)

            if args.mixed_precision:
                policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
                tf.keras.mixed_precision.experimental.set_policy(policy)
                print("Enabled mixed precision training")

            dataset = SeganDataset(
                "train", config["learning_config"]["dataset_config"]["train_paths"],
                config["learning_config"]["dataset_config"]["noise_config"],
                config["speech_config"], shuffle=True
            )

            segan_trainer = SeganTrainer(
                config["speech_config"],
                config["learning_config"]["running_config"],
                args.mixed_precision
            )

            segan_trainer.compile(config["model_config"],
                                  config["learning_config"]["optimizer_config"],
                                  max_to_keep=args.max_ckpts)
            segan_trainer.fit(train_dataset=dataset)

            if args.export:
                if args.from_weights:
                    segan_trainer.generator.save_weights(args.export)
                else:
                    segan_trainer.generator.save(args.export)
        elif args.mode == "test":
            tf.random.set_seed(0)
            assert args.export

            dataset = SeganDataset(
                "test", config["learning_config"]["dataset_config"]["test_paths"],
                config["learning_config"]["dataset_config"]["noise_config"],
                config["speech_config"], shuffle=False).create_test()

            segan_tester = SeganTester(
                config["speech_config"], config["learning_config"]["running_config"],
                args.export, from_weights=args.from_weights)

            segan_tester.compile(config["model_config"])
            segan_tester.run(dataset)

        else:
            assert args.export
            segan_trainer = SeganTrainer(
                config["speech_config"],
                config["learning_config"]["running_config"], args.mixed_precision)
            segan_trainer.compile(config["model_config"],
                                  config["learning_config"]["optimizer_config"])
            segan_trainer.load_checkpoint()

            if args.from_weights:
                segan_trainer.generator.save_weights(args.export)
            else:
                segan_trainer.generator.save(args.export)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
