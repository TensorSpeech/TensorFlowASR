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

env_util.setup_environment()

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.asr_dataset import ASRDataset
from tensorflow_asr.helpers import featurizer_helpers
from tensorflow_asr.utils import cli_util, file_util


def main(
    *transcripts,
    stage: str = "train",
    config_path: str = None,
    metadata: str = None,
):
    transcripts = file_util.preprocess_paths(transcripts)

    config = Config(config_path)

    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(config=config)

    dataset = ASRDataset(
        data_paths=transcripts,
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        stage=stage,
        shuffle=False,
    )

    dataset.update_metadata(metadata)


if __name__ == "__main__":
    cli_util.run(main)
