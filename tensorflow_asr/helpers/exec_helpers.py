# Copyright 2022 Huy Le Nguyen (@usimarit)
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

import tensorflow as tf
from tqdm import tqdm

from tensorflow_asr.datasets.asr_dataset import ASRSliceDataset
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.utils import app_util, file_util

logger = tf.get_logger()


def run_testing(
    model: BaseModel,
    test_dataset: ASRSliceDataset,
    test_data_loader: tf.data.Dataset,
    output: str,
):
    with file_util.save_file(file_util.preprocess_paths(output)) as filepath:
        overwrite = True
        if tf.io.gfile.exists(filepath):
            overwrite = input(f"Overwrite existing result file {filepath} ? (y/n): ").lower() == "y"
        if overwrite:
            results = model.predict(test_data_loader, verbose=1)
            logger.info(f"Saving result to {output} ...")
            with tf.io.gfile.GFile(filepath, "w") as openfile:
                openfile.write("PATH\tDURATION\tGROUNDTRUTH\tGREEDY\tBEAMSEARCH\n")
                progbar = tqdm(total=test_dataset.total_steps, unit="batch")
                for i, pred in enumerate(results):
                    groundtruth, greedy, beamsearch = [x.decode("utf-8") for x in pred]
                    path, duration, _ = test_dataset.entries[i]
                    openfile.write(f"{path}\t{duration}\t{groundtruth}\t{greedy}\t{beamsearch}\n")
                    progbar.update(1)
                progbar.close()
        app_util.evaluate_results(filepath)


def convert_tflite(
    model: BaseModel,
    output: str,
):
    concrete_func = model.make_tflite_function().get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    output = file_util.preprocess_paths(output)
    with open(output, "wb") as tflite_out:
        tflite_out.write(tflite_model)
