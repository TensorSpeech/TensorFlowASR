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
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from .scripts.run_ctc import main as main_ctc
from .scripts.run_segan import main as main_segan
from .scripts.create_tfrecords import main as main_stt_tfrecords

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,",
              len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def main():
    tf.keras.backend.clear_session()

    parser = argparse.ArgumentParser(prog="ASR")

    subparsers = parser.add_subparsers(help='Commands for training or testing models')

    parser_tfrecords = subparsers.add_parser("stt_tfrecords", help="Create stt tfrecords")
    run_stt_tfrecords = main_stt_tfrecords(parser_tfrecords)
    parser_tfrecords.set_defaults(func=run_stt_tfrecords)

    # Create parser for ctc
    parser_ctc = subparsers.add_parser("ctc", help="Run ctc model")
    run_ctc = main_ctc(parser_ctc)
    parser_ctc.set_defaults(func=run_ctc)

    # Create parser for segan
    parser_segan = subparsers.add_parser("segan", help="Run segan model")
    run_segan = main_segan(parser_segan)
    parser_segan.set_defaults(func=run_segan)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
