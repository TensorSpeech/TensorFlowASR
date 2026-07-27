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


import logging
import os

from tensorflow_asr import datasets, tf, tokenizers  # import to aid logging messages
from tensorflow_asr.callbacks import PredictLogger
from tensorflow_asr.configs import Config
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.utils import app_util, cli_util, env_util, file_util, keras_util

logger = logging.getLogger(__name__)


def main(
    config_path: str,
    dataset_type: str,
    datadir: str,
    outputdir: str,
    h5: str = None,
    lm_h5: str = None,
    internal_lm_h5: str = None,
    mxp: str = "none",
    bs: int = 1,
    jit_compile: bool = False,
    device_type: str = "cpu",
    devices: list = None,
    tpu_address: str = None,
    tpu_vm: bool = False,
    repodir: str = os.getcwd(),
):
    """
    Evaluate a checkpoint over the test datasets.

    Parameters
    ----------
    lm_h5 : str
        Weights of the external language model fused into beam search, from
        `tensorflow_asr train_lm --target=external`. Optional, and only read when
        `lm_config.external_config` describes one. Without it that model keeps its initial
        weights, which is never what you want outside a test.
    internal_lm_h5 : str
        Same, for the low-order language model LODR subtracts
        (`train_lm --target=internal`, `lm_config.internal_config`). Only read when
        `decoder_config.lm_type` is "lodr".
    device_type : str
        "cpu" (default), "gpu" or "tpu". Decoding defaults to the CPU on purpose: the greedy and
        beam loops are hundreds of tiny sequential steps, so an accelerator spends more time moving
        each step on and off the device than the arithmetic saves. On Apple silicon with the
        `apple` extra installed, a measured decode was ~230x slower on the GPU than the CPU, and
        `jit_compile=True` fails there. Pass `--device-type=gpu` if your accelerator does win --
        worth timing before assuming it. On that measured decode: 256s on the accelerator, 1.2s on
        the CPU.
    """
    outputdir = file_util.preprocess_paths(outputdir, isdir=True)
    checkpoint_name = os.path.splitext(os.path.basename(h5))[0]

    # Devices first: the visible list locks as soon as anything touches TensorFlow, and
    # `setup_seed` is enough to do it.
    env_util.setup_strategy(device_type=device_type, devices=devices, tpu_address=tpu_address, tpu_vm=tpu_vm)
    env_util.setup_seed()
    env_util.setup_mxp(mxp=mxp)

    config = Config(config_path, training=False, repodir=repodir, datadir=datadir)
    batch_size = bs

    tokenizer = tokenizers.get(config)
    tokenizer.make()

    logger.info(f"Configs: {str(config)}")

    model: BaseModel = keras_util.model_from_config(config.model_config)
    model.tokenizer = tokenizer
    model.make_lm(config.lm_config, lm_weights=lm_h5, internal_lm_weights=internal_lm_h5)  # no-op unless lm_config sets a model
    app_util.validate_lm(model, config.decoder_config, lm_h5=lm_h5, internal_lm_h5=internal_lm_h5)
    model.make(batch_size=batch_size)
    model.load_weights(h5, skip_mismatch=False)
    model.jit_compile = jit_compile
    model.summary()

    for test_data_config in config.data_config.test_dataset_configs:
        if not test_data_config.name:
            raise ValueError("Test dataset name must be provided")
        logger.info(f"Testing dataset: {test_data_config.name}")

        output = os.path.join(outputdir, f"{test_data_config.name}-{checkpoint_name}.tsv")

        test_dataset = datasets.get(tokenizer=tokenizer, dataset_config=test_data_config, dataset_type=dataset_type)
        test_data_loader = test_dataset.create(batch_size)

        overwrite = True
        if tf.io.gfile.exists(output):
            while overwrite not in ["yes", "no"]:
                overwrite = input(f"File {output} exists, overwrite? (yes/no): ").lower()
            overwrite = overwrite == "yes"

        if overwrite:
            with file_util.save_file(output) as output_file_path, env_util.device_scope(device_type):
                model.predict(
                    test_data_loader,
                    verbose=1,
                    callbacks=[
                        PredictLogger(test_dataset=test_dataset, output_file_path=output_file_path),
                    ],
                )

        evaluation_outputs = app_util.evaluate_hypotheses(output)
        logger.info(f"Results:\n{evaluation_outputs.to_markdown()}")


if __name__ == "__main__":
    cli_util.run(main)
