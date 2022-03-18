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

from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.asr_dataset import ASRTFRecordDataset
from tensorflow_asr.helpers import featurizer_helpers
from tensorflow_asr.utils import cli_util, file_util


def main(
    *transcripts,
    mode: str = None,
    config_path: str = None,
    tfrecords_dir: str = None,
    tfrecords_shards: int = 16,
    shuffle: bool = True,
):
    data_paths = file_util.preprocess_paths(transcripts)
    tfrecords_dir = file_util.preprocess_paths(tfrecords_dir, isdir=True)
    logger.info(f"Create tfrecords to directory: {tfrecords_dir}")

    config = Config(config_path)

    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(config=config)

    tfrecord_dataset = ASRTFRecordDataset(
        data_paths=data_paths,
        tfrecords_dir=tfrecords_dir,
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        stage=mode,
        shuffle=shuffle,
        tfrecords_shards=tfrecords_shards,
    )
    tfrecord_dataset.create_tfrecords()


if __name__ == "__main__":
    cli_util.run(main)
