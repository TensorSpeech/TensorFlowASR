# pylint: disable=not-callable
# Copyright 2020 Huy Le Nguyen (@nglehuy)
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

from tensorflow_asr.metrics.error_rates import ErrorRate
from tensorflow_asr.utils import file_util, metric_util

logger = tf.get_logger()


def evaluate_results(
    filepath: str,
):
    logger.info(f"Evaluating result from {filepath} ...")
    metrics = {
        "greedy_wer": ErrorRate(metric_util.tf_wer, name="greedy_wer", dtype=tf.float32),
        "greedy_cer": ErrorRate(metric_util.tf_cer, name="greedy_cer", dtype=tf.float32),
        "beamsearch_wer": ErrorRate(metric_util.tf_wer, name="beamsearch_wer", dtype=tf.float32),
        "beamsearch_cer": ErrorRate(metric_util.tf_cer, name="beamsearch_cer", dtype=tf.float32),
    }
    with file_util.read_file(filepath) as path:
        with open(path, "r", encoding="utf-8") as openfile:
            lines = openfile.read().splitlines()
            lines = lines[1:]  # skip header
    for eachline in tqdm(lines):
        _, _, groundtruth, greedy, beamsearch = eachline.split("\t")
        groundtruth = tf.convert_to_tensor([groundtruth], dtype=tf.string)
        greedy = tf.convert_to_tensor([greedy], dtype=tf.string)
        beamsearch = tf.convert_to_tensor([beamsearch], dtype=tf.string)
        metrics["greedy_wer"].update_state(decode=greedy, target=groundtruth)
        metrics["greedy_cer"].update_state(decode=greedy, target=groundtruth)
        metrics["beamsearch_wer"].update_state(decode=beamsearch, target=groundtruth)
        metrics["beamsearch_cer"].update_state(decode=beamsearch, target=groundtruth)
    for key, value in metrics.items():
        logger.info(f"{key}: {value.result().numpy()}")
