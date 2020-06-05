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

import tensorflow as tf


def main(parser):
    modes = ["train", "test", "infer", "save", "save_from_checkpoint",
             "convert_to_tflite", "load_tflite"]

    parser.add_argument("--mode", "-m", type=str, default="train",
                        help=f"Mode in {modes}")

    parser.add_argument("--config", "-c", type=str, default=None,
                        help="The file path of model configuration file")

    parser.add_argument("--input_file_path", "-i", type=str, default=None,
                        help="Path to input file")

    parser.add_argument("--export_file", "-e", type=str, default=None,
                        help="Path to the model file to be exported")

    parser.add_argument("--output_file_path", "-o", type=str, default=None,
                        help="Path to output file")

    def run(args):
        assert args.mode in modes, f"Mode must in {modes}"

        if args.mode == "train":
            tf.random.set_seed(2020)
        else:
            tf.random.set_seed(0)

    return run
