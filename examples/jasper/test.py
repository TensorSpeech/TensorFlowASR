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
import fire
from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()
import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.models.ctc.jasper import Jasper
from tensorflow_asr.helpers import exec_helpers, dataset_helpers, featurizer_helpers

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")


def main(
    config: str = DEFAULT_YAML,
    saved: str = None,
    mxp: bool = False,
    bs: int = None,
    sentence_piece: bool = False,
    subwords: bool = False,
    device: int = 0,
    cpu: bool = False,
    output: str = "test.tsv",
):
    assert saved and output
    tf.random.set_seed(0)
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": mxp})
    env_util.setup_devices([device], cpu=cpu)

    config = Config(config)

    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(
        config=config,
        subwords=subwords,
        sentence_piece=sentence_piece,
    )

    jasper = Jasper(**config.model_config, vocabulary_size=text_featurizer.num_classes)
    jasper.make(speech_featurizer.shape)
    jasper.load_weights(saved, by_name=True)
    jasper.summary(line_length=100)
    jasper.add_featurizers(speech_featurizer, text_featurizer)

    test_dataset = dataset_helpers.prepare_testing_datasets(
        config=config, speech_featurizer=speech_featurizer, text_featurizer=text_featurizer
    )
    batch_size = bs or config.learning_config.running_config.batch_size
    test_data_loader = test_dataset.create(batch_size)

    exec_helpers.run_testing(model=jasper, test_dataset=test_dataset, test_data_loader=test_data_loader, output=output)


if __name__ == "__main__":
    fire.Fire(main)
