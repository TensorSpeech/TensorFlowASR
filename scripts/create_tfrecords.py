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

import argparse
from tensorflow_asr.utils.utils import preprocess_paths
from tensorflow_asr.datasets.asr_dataset import ASRTFRecordDataset

modes = ["train", "eval", "test"]

parser = argparse.ArgumentParser(prog="TFRecords Creation")

parser.add_argument("--mode", "-m", type=str, default=None, help=f"Mode in {modes}")

parser.add_argument("--tfrecords_dir", type=str, default=None, help="Directory to tfrecords")

parser.add_argument("--tfrecords_shards", type=int, default=16, help="Number of tfrecords shards")

parser.add_argument("--shuffle", default=False, action="store_true", help="Shuffle data or not")

parser.add_argument("transcripts", nargs="+", type=str, default=None, help="Paths to transcript files")

args = parser.parse_args()

assert args.mode in modes, f"Mode must in {modes}"

transcripts = preprocess_paths(args.transcripts)
tfrecords_dir = preprocess_paths(args.tfrecords_dir)

ASRTFRecordDataset(
    data_paths=transcripts, tfrecords_dir=tfrecords_dir,
    speech_featurizer=None, text_featurizer=None,
    stage=args.mode, shuffle=args.shuffle, tfrecords_shards=args.tfrecords_shards
).create_tfrecords()
