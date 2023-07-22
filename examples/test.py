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


import json
import os

from tensorflow_asr import datasets, tf, tokenizers  # import to aid logging messages
from tensorflow_asr.callbacks import PredictLogger
from tensorflow_asr.configs import Config
from tensorflow_asr.utils import app_util, cli_util, env_util, file_util

logger = tf.get_logger()


def main(
    config_path: str,
    dataset_type: str,
    datadir: str,
    h5: str = None,
    mxp: str = "none",
    bs: int = 1,
    device: int = 0,
    cpu: bool = False,
    jit_compile: bool = False,
    output: str = "test.tsv",
    repodir: str = os.path.realpath(os.path.join(os.path.dirname(__file__), "..")),
):
    assert h5 and output
    output = file_util.preprocess_paths(output)

    env_util.setup_seed()
    env_util.setup_devices([device], cpu=cpu)
    env_util.setup_mxp(mxp=mxp)

    config = Config(config_path, training=False, repodir=repodir, datadir=datadir)
    batch_size = bs

    tokenizer = tokenizers.get(config)

    test_dataset = datasets.get(tokenizer=tokenizer, dataset_config=config.data_config.test_dataset_config, dataset_type=dataset_type)
    test_data_loader = test_dataset.create(batch_size)

    model: tf.keras.Model = tf.keras.models.model_from_config(config.model_config)
    model.make(batch_size=batch_size)
    model.load_weights(h5, by_name=file_util.is_hdf5_filepath(h5), skip_mismatch=True)
    model.jit_compile = jit_compile
    model.summary()

    overwrite = True
    if tf.io.gfile.exists(output):
        while overwrite not in ["yes", "no"]:
            overwrite = input(f"File {output} exists, overwrite? (yes/no): ").lower()
        overwrite = overwrite == "yes"

    if overwrite:
        with file_util.save_file(output) as output_file_path:
            model.predict(
                test_data_loader,
                verbose=1,
                callbacks=[
                    PredictLogger(tokenizer=tokenizer, test_dataset=test_dataset, output_file_path=output_file_path),
                ],
            )

    evaluation_outputs = app_util.evaluate_hypotheses(output)
    logger.info(json.dumps(evaluation_outputs, indent=2))


if __name__ == "__main__":
    cli_util.run(main)
