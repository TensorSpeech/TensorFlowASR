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

import jiwer
import tensorflow as tf
import tensorflow_text as tf_text
from tqdm import tqdm

from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.tokenizers import Tokenizer
from tensorflow_asr.utils import file_util, math_util

logger = tf.get_logger()


def evaluate_hypotheses(filepath: str):
    """
    Compute wer, cer, mer, wil, wip for given lists of greedy and beamsearch hypotheses

    Parameters
    ----------
    filepath : str
        Output tsv file path for the predictions

    Returns
    -------
    dict
        {"greedy": {wer, cer, mer, wil, wip}, "beam": {wer, cer, mer, wil, wip}}
        The results are original, NOT multiplied with 100.
    """
    logger.info(f"Reading file {filepath} ...")
    reference, greedy_hypothesis, beam_hypothesis = [], [], []
    with file_util.read_file(filepath) as path:
        with tf.io.gfile.GFile(path, "r") as openfile:
            lines = openfile.read().splitlines()
            lines = lines[1:]  # skip header
            for eachline in tqdm(lines):
                _, groundtruth, greedy, beamsearch = eachline.split("\t")
                reference.append(groundtruth)
                greedy_hypothesis.append(greedy)
                beam_hypothesis.append(beamsearch)

    logger.info("Evaluating greedy results ...")
    greedy_wordoutput = jiwer.process_words(reference=reference, hypothesis=greedy_hypothesis)
    greedy_charoutput = jiwer.process_characters(reference=reference, hypothesis=greedy_hypothesis)

    logger.info("Evaluating beamsearch results ...")
    beam_wordoutput = jiwer.process_words(reference=reference, hypothesis=beam_hypothesis)
    beam_charoutput = jiwer.process_characters(reference=reference, hypothesis=beam_hypothesis)

    outputs = {
        "greedy": {
            "wer": greedy_wordoutput.wer,
            "cer": greedy_charoutput.cer,
            "mer": greedy_wordoutput.mer,
            "wil": greedy_wordoutput.wil,
            "wip": greedy_wordoutput.wip,
        },
        "beam": {
            "wer": beam_wordoutput.wer,
            "cer": beam_charoutput.cer,
            "mer": beam_wordoutput.mer,
            "wil": beam_wordoutput.wil,
            "wip": beam_wordoutput.wip,
        },
    }

    return outputs


def convert_tflite(
    model: BaseModel,
    output: str,
    batch_size: int = 1,
    beam_width: int = 0,
):
    if not math_util.is_power_of_two(model.feature_extraction.nfft):
        logger.error("NFFT must be power of 2 for TFLite conversion")
        overwrite_nfft = input("Do you want to overwrite nfft to the nearest power of 2? (y/n): ")
        if overwrite_nfft.lower() == "y":
            model.feature_extraction.nfft = math_util.next_power_of_two(model.feature_extraction.nfft)
            logger.info(f"Overwritten nfft to {model.feature_extraction.nfft}")
        else:
            raise ValueError("NFFT must be power of 2 for TFLite conversion")

    concrete_func = model.make_tflite_function(batch_size=batch_size, beam_width=beam_width).get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], trackable_obj=model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    output = file_util.preprocess_paths(output)
    with open(output, "wb") as tflite_out:
        tflite_out.write(tflite_model)
