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

from utils.utils import preprocess_paths
from datasets.asr_dataset import ASRTFRecordDataset


def main(parser):
    modes = ["train", "eval", "test"]

    parser.add_argument("--mode", "-m", type=str, default=None, help=f"Mode in {modes}")

    parser.add_argument("--tfrecords_dir", type=str, default=None, help="Directory to tfrecords")

    parser.add_argument("transcripts", nargs="+", type=str, default=None, help="Paths to transcript files")

    def run(args):
        assert args.mode in modes, f"Mode must in {modes}"

        transcripts = preprocess_paths(args.transcripts)
        tfrecords_dir = preprocess_paths(args.tfrecords_dir)

        if args.mode == "train":
            ASRTFRecordDataset(transcripts, tfrecords_dir, None, None, args.mode, shuffle=True).create_tfrecords()
        else:
            ASRTFRecordDataset(transcripts, tfrecords_dir, None, None, args.mode, shuffle=False).create_tfrecords()

    return run
