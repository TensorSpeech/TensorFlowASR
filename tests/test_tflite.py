"""
End-to-end TFLite conversion tests for every model architecture.

Each test builds a deliberately tiny model and runs the real export path used by
`tensorflow_asr tflite`: `app_util.convert_tflite`, which traces `make_tflite_function` under
`jit_compile=True` and returns a flatbuffer, writing it only when given an output path. Where
the TFLite Flex delegate is available the result is then loaded into an interpreter and
invoked. Nothing is mocked -- a model that cannot be traced, or whose graph the converter
rejects, fails here.

The interpreter tests skip when Flex is missing (it is absent from the macOS arm64 wheel);
the conversion tests always run, so no architecture goes unverified.

The models are shrunk to a few thousand parameters (8-unit layers, one block) because this
exercises graph *structure*, not accuracy. That keeps the whole file at roughly a minute
rather than the tens of minutes the real `small.yml.j2` configs would take.

All eight architectures export. `KNOWN_BROKEN` below is the escape hatch for any that stop
doing so: an entry marks that architecture `xfail(strict=True)` with the root cause. Strict
means a stale entry fails the run, so a marker cannot outlive the bug it documents.
"""

import os

import numpy as np
import pytest

from tensorflow_asr.configs import DecoderConfig
from tensorflow_asr.models.base_model import BaseModel
from tensorflow_asr.tokenizers import CharTokenizer

# Small enough to keep conversion fast; nfft must stay a power of two because
# `app_util.convert_tflite` otherwise stops and prompts on stdin.
SPEECH_CONFIG = dict(
    sample_rate=16000,
    frame_ms=25,
    stride_ms=10,
    nfft=256,
    num_feature_bins=40,
    feature_type="log_mel_spectrogram",
)

SIGNAL_SECONDS = 0.5
PREDICTION_CONFIG = dict(prediction_embed_dim=8, prediction_num_rnns=1, prediction_rnn_units=8, joint_dim=8)


def _conv2d_subsampling_layer():
    """Subsampling spec in `class_name`/`config` form (Conformer models deserialize it)."""
    return {
        "class_name": "tensorflow_asr.models.layers.subsampling>Conv2dSubsampling",
        "config": {
            "filters": [8, 8],
            "kernels": [3, 3],
            "strides": [2, 2],
            "paddings": ["causal", "causal"],
            "norms": ["batch", "batch"],
            "activations": ["swish", "swish"],
        },
    }


def _conv2d_subsampling_typed():
    """
    Subsampling spec in `type` form, which the Transformer encoders expect.

    Returned fresh on every call on purpose: `TransformerEncoder.__init__` does
    `subsampling.pop("type")`, mutating the caller's dict, so a shared instance works
    exactly once. See `test_subsampling_config_is_consumed_destructively`.
    """
    return {
        "type": "conv2d",
        "filters": [8, 8],
        "kernels": [3, 3],
        "strides": [2, 2],
        "paddings": ["causal", "causal"],
        "norms": ["batch", "batch"],
        "activations": ["relu", "relu"],
    }


def _contextnet_blocks():
    # filters must stay >= 16: ContextNet scales them by `encoder_alpha` (0.5) and SEModule
    # then builds a `Dense(filters // 8)`, which rejects units=0.
    return [
        dict(nlayers=1, kernel_size=3, filters=32, strides=1, residual=False, activation="silu", padding="causal"),
        dict(nlayers=1, kernel_size=3, filters=32, strides=2, residual=True, activation="silu", padding="causal"),
    ]


def _build_ctc_conformer(vocab_size):
    from tensorflow_asr.models.ctc.conformer import Conformer

    return Conformer(
        blank=0,
        vocab_size=vocab_size,
        speech_config=SPEECH_CONFIG,
        encoder_subsampling=_conv2d_subsampling_layer(),
        encoder_dmodel=8,
        encoder_num_blocks=1,
        encoder_head_size=4,
        encoder_num_heads=2,
        encoder_kernel_size=3,
    )


def _build_ctc_deepspeech2(vocab_size):
    from tensorflow_asr.models.ctc.deepspeech2 import DeepSpeech2

    return DeepSpeech2(
        blank=0,
        vocab_size=vocab_size,
        speech_config=SPEECH_CONFIG,
        conv_kernels=[[3, 3]],
        conv_strides=[[2, 2]],
        conv_filters=[8],
        rnn_nlayers=1,
        rnn_units=8,
        fc_nlayers=1,
        fc_units=8,
    )


def _build_ctc_jasper(vocab_size):
    from tensorflow_asr.models.ctc.jasper import Jasper

    return Jasper(
        blank=0,
        vocab_size=vocab_size,
        speech_config=SPEECH_CONFIG,
        first_additional_block_channels=8,
        first_additional_block_kernels=3,
        nsubblocks=1,
        block_channels=[8],
        block_kernels=[3],
        block_dropout=[0.1],
        second_additional_block_channels=8,
        second_additional_block_kernels=1,
        third_additional_block_channels=8,
        third_additional_block_kernels=1,
    )


def _build_ctc_transformer(vocab_size):
    from tensorflow_asr.models.ctc.transformer import Transformer

    return Transformer(
        blank=0,
        vocab_size=vocab_size,
        speech_config=SPEECH_CONFIG,
        encoder_subsampling=_conv2d_subsampling_typed(),
        encoder_dmodel=8,
        encoder_dff=16,
        encoder_num_blocks=1,
        encoder_head_size=4,
        encoder_num_heads=2,
    )


def _build_transducer_conformer(vocab_size):
    from tensorflow_asr.models.transducer.conformer import Conformer

    return Conformer(
        blank=0,
        vocab_size=vocab_size,
        speech_config=SPEECH_CONFIG,
        encoder_subsampling=_conv2d_subsampling_layer(),
        encoder_dmodel=8,
        encoder_num_blocks=1,
        encoder_head_size=4,
        encoder_num_heads=2,
        encoder_kernel_size=3,
        **PREDICTION_CONFIG,
    )


def _build_transducer_contextnet(vocab_size):
    from tensorflow_asr.models.transducer.contextnet import ContextNet

    return ContextNet(
        blank=0,
        vocab_size=vocab_size,
        speech_config=SPEECH_CONFIG,
        encoder_blocks=_contextnet_blocks(),
        **PREDICTION_CONFIG,
    )


def _build_transducer_rnnt(vocab_size):
    from tensorflow_asr.models.transducer.rnnt import RnnTransducer

    return RnnTransducer(
        blank=0,
        vocab_size=vocab_size,
        speech_config=SPEECH_CONFIG,
        encoder_reduction_positions=["pre", "pre"],
        encoder_reduction_factors=[3, 0],
        encoder_dmodel=8,
        encoder_nlayers=2,
        encoder_rnn_units=8,
        prediction_projection_units=8,
        **PREDICTION_CONFIG,
    )


def _build_transducer_transformer(vocab_size):
    from tensorflow_asr.models.transducer.transformer import Transformer

    return Transformer(
        blank=0,
        vocab_size=vocab_size,
        speech_config=SPEECH_CONFIG,
        encoder_subsampling=_conv2d_subsampling_typed(),
        encoder_dmodel=8,
        encoder_dff=16,
        encoder_num_blocks=1,
        encoder_head_size=4,
        encoder_num_heads=2,
        **PREDICTION_CONFIG,
    )


BUILDERS = {
    "ctc.Conformer": _build_ctc_conformer,
    "ctc.DeepSpeech2": _build_ctc_deepspeech2,
    "ctc.Jasper": _build_ctc_jasper,
    "ctc.Transformer": _build_ctc_transformer,
    "transducer.Conformer": _build_transducer_conformer,
    "transducer.ContextNet": _build_transducer_contextnet,
    "transducer.RnnTransducer": _build_transducer_rnnt,
    "transducer.Transformer": _build_transducer_transformer,
}

TRANSDUCERS = [name for name in BUILDERS if name.startswith("transducer.")]

# Architectures that cannot currently be exported, mapped to the defect that blocks them.
#
# Empty, and it should stay that way: every architecture converts. Add an entry only with the
# root cause, not the symptom, and delete it the moment the cause is fixed -- `strict=True`
# turns a stale entry into a failure rather than letting it quietly mask working code.
#
# Five entries lived here until the `call_next` arity mismatch (JasperEncoder,
# TransformerEncoder and ContextNetEncoder each returned `call()`'s 2-tuple while every caller
# unpacked 3) and DeepSpeech2's `get_initial_decoder_states` returning None were fixed.
KNOWN_BROKEN = {}


def _maybe_xfail(name):
    """Attach the strict xfail marker for architectures with a recorded defect."""
    reason = KNOWN_BROKEN.get(name)
    return pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=reason)) if reason else name


@pytest.fixture(scope="module")
def tokenizer():
    """Character tokenizer over the built-in English alphabet -- needs no vocabulary file."""
    tok = CharTokenizer(DecoderConfig({"type": "characters", "blank_index": 0, "vocabulary": None}))
    tok.make()
    return tok


@pytest.fixture(scope="module")
def signal():
    """Half a second of deterministic noise, shaped as the tflite signature expects."""
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(int(SPEECH_CONFIG["sample_rate"] * SIGNAL_SECONDS)).astype(np.float32) * 0.1
    return samples.reshape(1, -1)


def build_model(name, tokenizer, batch_size=1) -> BaseModel:
    """
    Build and materialise a model.

    `batch_size` must match the one later handed to `make_tflite_function`: `Transducer.recognize`
    branches on `self._batch_size` to pick `recognize_single` vs `recognize_batch`, so a model
    made at 1 and exported at 3 traces the single-utterance path against a batched signature
    and fails on a shape mismatch.
    """
    model = BUILDERS[name](tokenizer.num_classes)
    model.tokenizer = tokenizer
    model.make(batch_size=batch_size)
    return model


def convert(model: BaseModel, batch_size=1, beam_width=0, output=None) -> bytes:
    """
    Export through the real `app_util.convert_tflite`.

    Going through the helper rather than driving `TFLiteConverter` directly is deliberate: it
    is what `tensorflow_asr tflite` actually calls, so its nfft guard and converter flags are
    covered here too. `output=None` skips the write and just returns the flatbuffer.
    """
    from tensorflow_asr.utils import app_util

    return app_util.convert_tflite(model=model, output=output, batch_size=batch_size, beam_width=beam_width)


def _make_interpreter(tflite_model: bytes):
    import tensorflow_text as tft
    from tensorflow.lite.python import interpreter as tflite_interpreter

    return tflite_interpreter.InterpreterWithCustomOps(
        model_content=tflite_model,
        custom_op_registerers=tft.tflite_registrar.SELECT_TFTEXT_OPS,
    )


@pytest.fixture(scope="module")
def flex_delegate(tokenizer):
    """
    Skip the interpreter tests where the Flex delegate is missing.

    Conversion emits `SELECT_TF_OPS`, so every model here needs Flex to run. The delegate ships
    only in some TensorFlow builds -- notably not the macOS arm64 wheel -- and its absence is an
    environment limitation, not a defect in the model. Conversion itself is covered separately
    and always runs, so nothing goes unverified when this skips.
    """
    model = build_model("ctc.Conformer", tokenizer)
    try:
        _make_interpreter(convert(model)).allocate_tensors()
    except RuntimeError as error:
        if "Flex delegate" not in str(error) and "Select TensorFlow op" not in str(error):
            raise
        pytest.skip(f"TFLite Flex delegate unavailable in this build: {str(error).splitlines()[0][:120]}")
    return True


def invoke(tflite_model: bytes, signal: np.ndarray, blank=0):
    """
    Run a converted model exactly the way `examples/inferences/tflite.py` does.

    Returns the interpreter's output tensors in signature order.
    """
    runner = _make_interpreter(tflite_model)
    input_details = runner.get_input_details()
    output_details = runner.get_output_details()

    runner.resize_tensor_input(input_details[0]["index"], signal.shape, strict=True)
    runner.allocate_tensors()
    runner.set_tensor(input_details[0]["index"], signal)
    runner.set_tensor(input_details[1]["index"], np.array([signal.shape[1]], dtype=np.int32))
    for detail in input_details[2:]:
        fill = blank if detail["index"] == input_details[2]["index"] else 0
        runner.set_tensor(detail["index"], np.full(detail["shape"], fill, dtype=detail["dtype"]))

    runner.invoke()
    return [runner.get_tensor(detail["index"]) for detail in output_details]


@pytest.mark.parametrize("name", [_maybe_xfail(name) for name in BUILDERS])
def test_greedy_conversion(name, tokenizer):
    """Every architecture must export a non-empty flatbuffer with beam_width=0."""
    tflite_model = convert(build_model(name, tokenizer), batch_size=1, beam_width=0)
    assert len(tflite_model) > 0


@pytest.mark.parametrize("name", [_maybe_xfail(name) for name in TRANSDUCERS])
def test_beam_search_conversion(name, tokenizer):
    """
    `make_tflite_function` routes to `recognize_beam` when beam_width > 0.

    The ALSD++ beam search is a `tf.while_loop` with `top_k` and rolling-hash recombination,
    so it is much more likely than greedy to hit an op the converter rejects.
    """
    tflite_model = convert(build_model(name, tokenizer), batch_size=1, beam_width=2)
    assert len(tflite_model) > 0


def graph_ops(model: BaseModel, beam_width) -> set:
    """The set of op types in the exported graph, including `tf.while_loop` function bodies."""
    graph_def = model.make_tflite_function(batch_size=1, beam_width=beam_width).get_concrete_function().graph.as_graph_def()
    ops = {node.op for node in graph_def.node}
    for function in graph_def.library.function:
        ops |= {node.op for node in function.node_def}
    return ops


@pytest.mark.parametrize("name", [_maybe_xfail(name) for name in TRANSDUCERS])
def test_beam_export_traces_the_beam_search(name, tokenizer):
    """
    A beam_width>0 export must embed the beam search, not the greedy decoder.

    Guards against `make_tflite_function` silently ignoring `beam_width`, which would ship a
    "beam search" tflite file that is really a greedy one. Comparing the converted flatbuffers
    byte-for-byte does not work -- conversion is not byte-deterministic, tensor names carry a
    counter that advances between calls -- so this compares op *types*, which are stable.

    `TopKV2` is the ALSD++ prune to the W best candidates and `OneHot` builds the forced-blank
    mask; neither exists in the greedy path.
    """
    model = build_model(name, tokenizer)
    greedy, beam = graph_ops(model, 0), graph_ops(model, 2)

    assert graph_ops(model, 0) == greedy, "op sets are not stable across tracing; the comparison below is meaningless"
    assert "TopKV2" not in greedy, "greedy decoding should not need a top_k"
    assert "TopKV2" in beam, "beam_width>0 did not trace the beam search"
    assert "OneHot" in beam, "the forced-blank mask is missing from the beam graph"


@pytest.mark.parametrize("name", [_maybe_xfail(name) for name in ["ctc.Conformer", "transducer.RnnTransducer"]])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_conversion_honours_batch_size(name, tokenizer, batch_size):
    """The signature's leading dimension must follow `batch_size`, since it is baked in."""
    model = build_model(name, tokenizer, batch_size=batch_size)
    concrete_func = model.make_tflite_function(batch_size=batch_size).get_concrete_function()
    inputs_spec = concrete_func.structured_input_signature[0][0]
    assert inputs_spec.inputs.shape[0] == batch_size
    assert inputs_spec.inputs.shape[1] is None, "the time axis must stay dynamic"
    assert inputs_spec.inputs_length.shape[0] == batch_size


@pytest.mark.usefixtures("flex_delegate")
@pytest.mark.parametrize("name", [_maybe_xfail(name) for name in BUILDERS])
def test_converted_model_runs_in_interpreter(name, tokenizer, signal):
    """Load the flatbuffer into an interpreter, invoke it, and check the output contract."""
    tflite_model = convert(build_model(name, tokenizer), batch_size=1, beam_width=0)
    outputs = invoke(tflite_model, signal)

    # transcript, tokens, next_tokens, then optional encoder/decoder states
    assert len(outputs) >= 3, f"expected at least transcript/tokens/next_tokens, got {len(outputs)}"
    transcript, tokens, next_tokens = outputs[0], outputs[1], outputs[2]
    assert transcript.dtype == object or transcript.dtype.type is np.bytes_, f"transcript dtype {transcript.dtype}"
    assert tokens.dtype == np.int32 and next_tokens.dtype == np.int32
    assert tokens.ndim == 2 and tokens.shape[0] == 1, f"tokens shape {tokens.shape}"
    assert np.all(tokens >= 0) and np.all(tokens < tokenizer.num_classes), "decoded a token outside the vocabulary"


@pytest.mark.usefixtures("flex_delegate")
@pytest.mark.parametrize("name", [_maybe_xfail(name) for name in TRANSDUCERS])
def test_converted_beam_model_runs_in_interpreter(name, tokenizer, signal):
    """The beam-search export must also survive an actual interpreter run."""
    tflite_model = convert(build_model(name, tokenizer), batch_size=1, beam_width=2)
    tokens = invoke(tflite_model, signal)[1]
    assert tokens.dtype == np.int32
    assert np.all(tokens >= 0) and np.all(tokens < tokenizer.num_classes)


def test_conversion_writes_a_file_when_output_is_given(tokenizer, tmp_path):
    """
    `output=None` returns the flatbuffer without touching disk; a path writes it.

    The rest of this file uses the None form, so the write branch that `tensorflow_asr tflite`
    actually depends on would otherwise never be exercised.
    """
    model = build_model("ctc.Conformer", tokenizer)
    output = tmp_path / "model.tflite"

    returned = convert(model, output=str(output))

    assert output.exists(), "convert_tflite was given a path but wrote nothing"
    assert output.read_bytes() == returned, "the file on disk differs from the returned model"
    assert len(returned) > 0


def test_conversion_without_output_writes_nothing(tokenizer, tmp_path):
    """The default must not create stray files next to the caller."""
    before = set(os.listdir(tmp_path))
    returned = convert(build_model("ctc.Conformer", tokenizer))

    assert len(returned) > 0
    assert set(os.listdir(tmp_path)) == before


def test_nfft_must_be_power_of_two_for_conversion():
    """
    `app_util.convert_tflite` refuses a non-power-of-two nfft.

    It asks on stdin first, so this pins the guard itself rather than calling the helper --
    a test that reached the `input()` call would hang the suite rather than fail.
    """
    from tensorflow_asr.utils import math_util

    assert math_util.is_power_of_two(SPEECH_CONFIG["nfft"]), "the fixture config would trigger the prompt"
    assert not math_util.is_power_of_two(400)
    assert math_util.next_power_of_two(400) == 512


def test_subsampling_config_is_consumed_destructively():
    """
    `TransformerEncoder.__init__` does `subsampling.pop("type")`, mutating its argument.

    Building two models from one config dict therefore fails on the second. This pins the
    current behaviour so the test helpers' use of fresh dicts is not mistaken for noise.
    """
    from tensorflow_asr.models.encoders.transformer import TransformerEncoder

    shared = _conv2d_subsampling_typed()
    TransformerEncoder(subsampling=shared, num_blocks=1, dmodel=8, dff=16, num_heads=2, head_size=4)

    assert "type" not in shared, "pop() no longer mutates the caller's dict -- this test can go"
    with pytest.raises(ValueError, match="subsampling must be either"):
        TransformerEncoder(subsampling=shared, num_blocks=1, dmodel=8, dff=16, num_heads=2, head_size=4)
