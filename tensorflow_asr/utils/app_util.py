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

import logging

import jiwer
from tensorflow.python.framework import convert_to_constants

from tensorflow_asr import tf
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.utils import file_util, math_util

logger = logging.getLogger(__name__)


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
        {"greedy": {wer, cer, mer, wil, wip}, "beam": {...}, "beam_lm": {...}}
        `beam` is the language-model-free beam and `beam_lm` the fused one; the two are identical
        when no language model was attached. The results are original, NOT multiplied with 100.
    """
    import pandas as pd  # pylint: disable=import-outside-toplevel
    from tqdm import tqdm  # pylint: disable=import-outside-toplevel

    logger.info(f"Reading file {filepath} ...")
    reference, greedy_hypothesis, beam_hypothesis, beam_lm_hypothesis = [], [], [], []
    with file_util.read_file(filepath) as path:
        with tf.io.gfile.GFile(path, "r") as openfile:
            lines = openfile.read().splitlines()
            lines = lines[1:]  # skip header
            for eachline in tqdm(lines, disable=False):
                _, groundtruth, greedy, beamsearch, beamsearch_lm = eachline.split("\t")
                reference.append(groundtruth)
                greedy_hypothesis.append(greedy)
                beam_hypothesis.append(beamsearch)
                beam_lm_hypothesis.append(beamsearch_lm)

    logger.info("Evaluating greedy results ...")
    greedy_wordoutput = jiwer.process_words(reference=reference, hypothesis=greedy_hypothesis)
    greedy_charoutput = jiwer.process_characters(reference=reference, hypothesis=greedy_hypothesis)

    logger.info("Evaluating beamsearch results ...")
    beam_wordoutput = jiwer.process_words(reference=reference, hypothesis=beam_hypothesis)
    beam_charoutput = jiwer.process_characters(reference=reference, hypothesis=beam_hypothesis)

    logger.info("Evaluating beamsearch + language model results ...")
    beam_lm_wordoutput = jiwer.process_words(reference=reference, hypothesis=beam_lm_hypothesis)
    beam_lm_charoutput = jiwer.process_characters(reference=reference, hypothesis=beam_lm_hypothesis)

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
        "beam_lm": {
            "wer": beam_lm_wordoutput.wer,
            "cer": beam_lm_charoutput.cer,
            "mer": beam_lm_wordoutput.mer,
            "wil": beam_lm_wordoutput.wil,
            "wip": beam_lm_wordoutput.wip,
        },
    }
    df = pd.DataFrame.from_dict(outputs, orient="index")
    return df


def validate_lm(model: BaseModel, decoder_config, lm_h5: str = None, internal_lm_h5: str = None):
    """
    Check that the language models attached to `model` make sense for how it is about to decode.

    `BaseModel.make_lm` only builds what the config describes; whether that combination is
    coherent depends on `decoder_config`, which it never sees. This is the other half, called by
    `scripts/test.py` once both are known.

    Raises only for what cannot run at all. Everything else is a warning, because each case is
    legal and occasionally deliberate -- but far more often a config that lost a key or a command
    line that lost a flag, and all of them fail *silently*: the decode completes and the WER is
    just quietly worse than it should be.

    Parameters
    ----------
    model : BaseModel
        After `make_lm`, so `model.lm` and `model.internal_lm` are populated.
    decoder_config : configs.DecoderConfig
        Supplies `lm_type`, `lm_alpha`, `lm_beta` and `beam_width`.
    lm_h5, internal_lm_h5 : str
        The paths that were passed to `make_lm`, used only to tell a trained model from one left
        at its initial weights.

    Raises
    ------
    ValueError
        `lm_type` is "lodr" but no internal language model was built.
    """
    lm_type = str(getattr(decoder_config, "lm_type", "shallow") or "shallow").lower()
    beam_width = int(getattr(decoder_config, "beam_width", 0) or 0)
    lm_alpha = float(getattr(decoder_config, "lm_alpha", 0.0) or 0.0)
    lm_beta = float(getattr(decoder_config, "lm_beta", 0.0) or 0.0)
    has_external, has_internal = model.lm is not None, model.internal_lm is not None

    if lm_type == "lodr" and not has_internal:
        raise ValueError(
            'decoder_config.lm_type is "lodr" but no internal language model was built. '
            "Set lm_config.internal_config to the low-order LM to subtract, or change lm_type."
        )

    if beam_width <= 0 and (has_external or has_internal):
        logger.warning(f"decoder_config.beam_width is {beam_width}, so beam search is off and the language models will not be used at all")
        return

    # An untrained LM is the quiet one: it is built, it is fused, and it contributes noise.
    if has_external and not lm_h5:
        logger.warning("An external language model is configured but no --lm-h5 was given, so it is fused in with its initial (untrained) weights")
    if has_internal and lm_type == "lodr" and not internal_lm_h5:
        logger.warning(
            "An internal language model is configured but no --internal-lm-h5 was given, so it is subtracted with its initial (untrained) weights"
        )

    if lm_type != "shallow" and not has_external:
        # Subtracting the internal LM without fusing anything in strips the model's language
        # knowledge and puts none back. Legal -- it is the density-ratio form with a uniform
        # external LM -- but it is far more often a config that forgot `external_config`.
        logger.warning(f'lm_type is "{lm_type}" but no lm_config.external_config was given, so the internal LM is subtracted with nothing fused in')
    if has_internal and lm_type != "lodr":
        logger.warning(
            f'lm_config.internal_config was built but lm_type is "{lm_type}", which never reads it. Only "lodr" subtracts a separate model.'
        )
    if has_external and lm_alpha == 0.0:
        logger.warning("An external language model is configured but lm_alpha is 0, so it costs a full LM call per step and changes nothing")
    if lm_type != "shallow" and lm_beta == 0.0:
        logger.warning(f'lm_type is "{lm_type}" but lm_beta is 0, so the internal LM correction is computed and then multiplied away')

    logger.info(f"Language model settings validated: lm_type={lm_type}, lm_alpha={lm_alpha}, lm_beta={lm_beta}, beam_width={beam_width}")


def convert_tflite(
    model: BaseModel,
    output: str = None,
    batch_size: int = 1,
    beam_width: int = 0,
) -> bytes:
    """
    Convert a model to TFLite and return the flatbuffer.

    Parameters
    ----------
    output : str, optional
        Where to write the converted model. When None, the model is only returned and
        nothing is written to disk -- useful for callers that just want the bytes.

    Returns
    -------
    bytes
        The converted TFLite model.

    Notes
    -----
    **The exported model needs the TFLite Flex delegate at runtime, and TensorFlow 2.20 dropped
    it from the pip wheel.** Conversion keeps working on 2.20 -- only inference breaks, with::

        RuntimeError: Select TensorFlow op(s), included in the given model, is(are) not
        supported by this interpreter. Make sure you apply/link the Flex delegate before
        inference.

    `SELECT_TF_OPS` below is not optional for these models. A `ctc.Conformer` export contains 12
    ops that have no TFLite builtin equivalent, in three groups that are all structural:

    * control flow -- `FlexTensorListReserve/SetItem/Stack`, `FlexPlaceholder`, `FlexRoll`, from
      the `tf.while_loop` in the greedy and beam decoders
    * text -- `FlexNormalizeUTF8`, `FlexStringLower`, `FlexStringStrip`, `FlexStaticRegexReplace`,
      `FlexReduceJoin`, `FlexLookupTableFindV2`, from `tokenizer.detokenize` producing the
      in-graph transcript
    * decoding -- `FlexCTCGreedyDecoder`

    Dropping to `TFLITE_BUILTINS` alone would mean giving up the in-graph transcript and
    rewriting both decoders without `while_loop`, so it is not a realistic trade.

    Verified availability of the delegate (macOS arm64 unless noted)::

        tensorflow 2.18.0                works
        tensorflow 2.19.x                works
        tensorflow 2.20.0                MISSING
        tensorflow 2.20.0 (Linux x86_64) MISSING
        tf-nightly 2.22.0-dev            MISSING
        ai-edge-litert 2.1.6             MISSING

    The mechanism is a build-configuration change, not a packaging accident:
    `tflite::AcquireFlexDelegate()` is a weak symbol that the real delegate overrides. Up to 2.19
    `libtensorflow_cc` carries ~151 `tflite::flex::` symbols and a strong definition; on 2.20 it
    carries zero and only the weak stub, which returns null and produces the error above. Note
    this contradicts https://ai.google.dev/edge/litert/conversion/tensorflow/ops_select, which
    still says select ops ship with the TensorFlow pip package.

    Consequences for callers:

    * A flatbuffer produced by 2.20 loads fine in a 2.18/2.19 interpreter, so conversion and
      execution can be split across environments if the project ever moves back to 2.20.
    * On Android the delegate is a dependency, not a build flag:
      `org.tensorflow:tensorflow-lite-select-tf-ops`.
    * `tests/test_tflite.py` skips its interpreter tests when the delegate is missing and still
      runs every conversion test, so the coverage loss is limited to executing the flatbuffer.
    * Conversion must run with no GPU visible. Keras selects the fused LSTM kernel whenever one is
      *visible* -- placement does not matter -- and it converts to a `CudnnRNNV3` custom op that no
      interpreter can resolve. `tests/conftest.py` hides accelerators for this reason.
    """
    if not math_util.is_power_of_two(model.feature_extraction.nfft):
        logger.error("NFFT must be power of 2 for TFLite conversion")
        overwrite_nfft = input("Do you want to overwrite nfft to the nearest power of 2? (y/n): ")
        if overwrite_nfft.lower() == "y":
            model.feature_extraction.nfft = math_util.next_power_of_two(model.feature_extraction.nfft)
            logger.info(f"Overwritten nfft to {model.feature_extraction.nfft}")
        else:
            raise ValueError("NFFT must be power of 2 for TFLite conversion")

    concrete_func = model.make_tflite_function(batch_size=batch_size, beam_width=beam_width).get_concrete_function()

    # Freeze the weights to constants before handing the graph over. `from_concrete_functions` does
    # this itself, but its pass gives up on large graphs: beam search over a recurrent encoder left
    # the LSTM kernels as `VAR_HANDLE` inputs to the decoding `WHILE` with no `ASSIGN_VARIABLE`
    # anywhere, so invoking died on `read_variable.cc: variable != nullptr was not true`. Greedy
    # over the same model froze fine, and each beam output added individually was fine -- only the
    # full set tipped it over, which is what a size threshold looks like rather than a bad op.
    # `lower_control_flow=False` keeps the while loops intact instead of unrolling them.
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(concrete_func, lower_control_flow=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_func], trackable_obj=model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    if output is not None:
        output = file_util.preprocess_paths(output)
        with open(output, "wb") as tflite_out:
            tflite_out.write(tflite_model)

    return tflite_model
