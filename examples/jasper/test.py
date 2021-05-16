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
from tqdm import tqdm
import argparse
from tensorflow_asr.utils import env_util, file_util

logger = env_util.setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="Jasper Testing")

parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")

parser.add_argument("--saved", type=str, default=None, help="Path to saved model")

parser.add_argument("--mxp", default=False, action="store_true", help="Enable mixed precision")

parser.add_argument("--bs", type=int, default=None, help="Test batch size")

parser.add_argument("--sentence_piece", default=False, action="store_true", help="Whether to use `SentencePiece` model")

parser.add_argument("--subwords", default=False, action="store_true", help="Use subwords")

parser.add_argument("--device", type=int, default=0, help="Device's id to run test on")

parser.add_argument("--cpu", default=False, action="store_true", help="Whether to only use cpu")

parser.add_argument("--output", type=str, default="test.tsv", help="Result filepath")

args = parser.parse_args()

assert args.saved

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

env_util.setup_devices([args.device], cpu=args.cpu)

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.asr_dataset import ASRSliceDataset
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import SubwordFeaturizer, SentencePieceFeaturizer, CharFeaturizer
from tensorflow_asr.models.ctc.jasper import Jasper
from tensorflow_asr.utils import app_util

config = Config(args.config)
speech_featurizer = TFSpeechFeaturizer(config.speech_config)

if args.sentence_piece:
    logger.info("Use SentencePiece ...")
    text_featurizer = SentencePieceFeaturizer(config.decoder_config)
elif args.subwords:
    logger.info("Use subwords ...")
    text_featurizer = SubwordFeaturizer(config.decoder_config)
else:
    logger.info("Use characters ...")
    text_featurizer = CharFeaturizer(config.decoder_config)

tf.random.set_seed(0)

test_dataset = ASRSliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    **vars(config.learning_config.test_dataset_config)
)

# build model
jasper = Jasper(**config.model_config, vocabulary_size=text_featurizer.num_classes)
jasper.make(speech_featurizer.shape)
jasper.load_weights(args.saved, by_name=True)
jasper.summary(line_length=100)
jasper.add_featurizers(speech_featurizer, text_featurizer)

batch_size = args.bs or config.learning_config.running_config.batch_size
test_data_loader = test_dataset.create(batch_size)

with file_util.save_file(file_util.preprocess_paths(args.output)) as filepath:
    overwrite = True
    if tf.io.gfile.exists(filepath):
        overwrite = input(f"Overwrite existing result file {filepath} ? (y/n): ").lower() == "y"
    if overwrite:
        results = jasper.predict(test_data_loader, verbose=1)
        logger.info(f"Saving result to {args.output} ...")
        with open(filepath, "w") as openfile:
            openfile.write("PATH\tDURATION\tGROUNDTRUTH\tGREEDY\tBEAMSEARCH\n")
            progbar = tqdm(total=test_dataset.total_steps, unit="batch")
            for i, pred in enumerate(results):
                groundtruth, greedy, beamsearch = [x.decode('utf-8') for x in pred]
                path, duration, _ = test_dataset.entries[i]
                openfile.write(f"{path}\t{duration}\t{groundtruth}\t{greedy}\t{beamsearch}\n")
                progbar.update(1)
            progbar.close()
    app_util.evaluate_results(filepath)
