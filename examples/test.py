# Copyright 2023 Huy Le Nguyen (@nglehuy)
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


from tensorflow_asr import tf  # import to aid logging messages
from tensorflow_asr import dataset
from tensorflow_asr.config import Config
from tensorflow_asr.featurizers import text_featurizers
from tensorflow_asr.helpers import exec_helpers
from tensorflow_asr.utils import cli_util, env_util, file_util


def main(
    config_path: str,
    dataset_type: str,
    h5: str = None,
    mxp: str = "none",
    bs: int = None,
    device: int = 0,
    cpu: bool = False,
    output: str = "test.tsv",
):
    assert h5 and output
    tf.keras.backend.clear_session()
    env_util.setup_seed()
    env_util.setup_devices([device], cpu=cpu)
    env_util.setup_mxp(mxp=mxp)

    config = Config(config_path)
    batch_size = bs or config.learning_config.running_config.batch_size

    text_featurizer = text_featurizers.get(config)

    test_dataset = dataset.get(text_featurizer=text_featurizer, dataset_config=config.data_config.test_dataset_config, dataset_type=dataset_type)
    test_data_loader = test_dataset.create(batch_size)

    model = tf.keras.models.model_from_config(config.model_config)
    model.make(batch_size=batch_size)
    model.load_weights(h5, by_name=file_util.is_hdf5_filepath(h5))
    model.summary()
    model.add_featurizers(text_featurizer)

    exec_helpers.run_testing(model=model, test_dataset=test_dataset, test_data_loader=test_data_loader, output=output)


if __name__ == "__main__":
    cli_util.run(main)
