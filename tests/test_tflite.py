"""
End-to-end TFLite conversion tests for every model architecture.

Each test builds a deliberately tiny model and runs the real export path used by
`tensorflow_asr tflite`: `app_util.convert_tflite`, which traces `make_tflite_function` under
`jit_compile=True` and returns a flatbuffer, writing it only when given an output path. Where
the TFLite Flex delegate is available the result is then loaded into an interpreter and
invoked. Nothing is mocked -- a model that cannot be traced, or whose graph the converter
rejects, fails here.

The interpreter tests skip only where the TFLite Flex delegate is unavailable -- TensorFlow 2.20
dropped it from the pip wheel, see the note on `app_util.convert_tflite`. The conversion tests
always run, so no architecture goes unverified even then.

The models are shrunk to a few thousand parameters (8-unit layers, one block) because this
exercises graph *structure*, not accuracy. That keeps the whole file at roughly a minute
rather than the tens of minutes the real `small.yml.j2` configs would take.

All eight architectures export. `KNOWN_BROKEN` below is the escape hatch for any that stop
doing so: an entry marks that architecture `xfail(strict=True)` with the root cause. Strict
means a stale entry fails the run, so a marker cannot outlive the bug it documents.
"""

import os
import re

import numpy as np
import pytest

from tensorflow_asr.configs import DecoderConfig, LanguageModelConfig
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

    # The converter does not preserve signature order, so inputs are located by their descriptors
    # rather than position -- index 0 is a state tensor for some architectures. The audio is the
    # only input with a dynamic dimension (its length is unknown at export), and `inputs_length`
    # the only rank-1 integer. Everything else is a token tensor (integral, seeded with `blank`) or
    # a carried state (floating point, seeded with zeros). Note rank and dtype alone are not enough:
    # a beam export also has a rank-2 float input, `previous_beam_scores` of shape [B, W].
    def only(details, predicate, description):
        matches = [d for d in details if predicate(d)]
        assert len(matches) == 1, f"expected exactly one {description} input, got {len(matches)}: {[d['name'] for d in matches]}"
        return matches[0]

    def is_signal(detail):
        return np.issubdtype(detail["dtype"], np.floating) and -1 in list(detail["shape_signature"])

    def is_length(detail):
        return np.issubdtype(detail["dtype"], np.integer) and len(detail["shape"]) == 1

    signal_input = only(runner.get_input_details(), is_signal, "dynamically shaped float (audio)")
    runner.resize_tensor_input(signal_input["index"], signal.shape, strict=True)
    runner.allocate_tensors()

    details = runner.get_input_details()  # descriptors are rebuilt by the resize
    signal_input = only(details, is_signal, "dynamically shaped float (audio)")
    length_input = only(details, is_length, "rank-1 integer (inputs_length)")

    runner.set_tensor(signal_input["index"], signal)
    runner.set_tensor(length_input["index"], np.array([signal.shape[1]], dtype=length_input["dtype"]))
    for detail in details:
        if detail["index"] in (signal_input["index"], length_input["index"]):
            continue
        fill = blank if np.issubdtype(detail["dtype"], np.integer) else 0
        runner.set_tensor(detail["index"], np.full(detail["shape"], fill, dtype=detail["dtype"]))

    runner.invoke()

    # `get_output_details()` is not in signature order either -- freezing variables to constants
    # renames the outputs to `StatefulPartitionedCall:N` and returns them shuffled, where N is the
    # true position. Unfrozen graphs use `Identity` / `Identity_N`. Sort on whichever suffix is
    # present so callers can rely on position; the transcript assertion downstream is the guard
    # that catches it if a future naming scheme defeats this.
    def signature_position(detail):
        name = detail["name"]
        head, _, tail = name.rpartition(":")
        if head and tail.isdigit():
            return int(tail)
        match = re.fullmatch(r"Identity(?:_(\d+))?", name)
        if match:
            return int(match.group(1) or 0)
        return len(name)  # unrecognised: keep a stable order rather than crashing

    ordered = sorted(runner.get_output_details(), key=signature_position)
    return [runner.get_tensor(detail["index"]) for detail in ordered]


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


def _lm_tokenizer(beam_width=2, lm_alpha=0.4):
    """
    A tokenizer of its own, so mutating `decoder_config` cannot leak into the module fixture.

    `beam_width` here is deliberately *not* what the export uses -- `make_tflite_function` takes
    its own -- but the rest of the language model settings do come off this object.
    """
    tok = CharTokenizer(DecoderConfig({"type": "characters", "blank_index": 0, "vocabulary": None, "beam_width": beam_width, "lm_alpha": lm_alpha}))
    tok.make()
    return tok


def _attach_bigram_lm(model, tokenizer, internal=False):
    """
    Give `model` a `BigramLanguageModel` through the real `make_lm` path.

    A bigram is the cheapest LM that is a real one: its forward pass is an `Embedding` gather, so
    it traces and converts like any other layer, and it is what LODR subtracts in practice.
    """
    import keras

    from tensorflow_asr.models.lm.bigram_language_model import BigramLanguageModel

    blob = keras.saving.serialize_keras_object(BigramLanguageModel(vocab_size=tokenizer.num_classes, blank=0))
    # LODR needs both: one to fuse in and one to subtract. The same architecture stands in for
    # both here -- what is being tested is that they reach the graph, not what they score.
    config = {"external_config": blob, "internal_config": blob} if internal else {"external_config": blob}
    model.make_lm(LanguageModelConfig(config))
    return model


@pytest.mark.parametrize("name", [_maybe_xfail(name) for name in ["transducer.RnnTransducer"]])
def test_beam_export_passes_the_language_model_settings(name):
    """
    A beam export must decode with the LM, not without it.

    `make_tflite_function` used to call `recognize_beam(inputs, beam_width=beam_width)` and pass
    nothing else, so every fusion setting silently reverted to its default and an exported
    `.tflite` was a plain ALSD++ beam however the config was written. It now builds its arguments
    with `get_beam_decoding_kwargs`, the same call `predict_step` uses.
    """
    tokenizer = _lm_tokenizer(beam_width=0, lm_alpha=0.4)  # 0 in the config: the export's own width must still win
    model = build_model(name, tokenizer)
    _attach_bigram_lm(model, tokenizer)

    captured = {}
    original = model.recognize_beam
    model.recognize_beam = lambda inputs, **kwargs: (captured.update(kwargs), original(inputs, **kwargs))[1]

    model.make_tflite_function(batch_size=1, beam_width=3).get_concrete_function()

    assert captured["lm"] is model.lm, "the external language model never reached recognize_beam"
    assert captured["lm_alpha"] == 0.4, "lm_alpha did not come from decoder_config"
    assert captured["beam_width"] == 3, "the export's own beam width must win over decoder_config"


@pytest.mark.parametrize("name", [_maybe_xfail(name) for name in ["transducer.RnnTransducer"]])
def test_greedy_export_ignores_the_language_model(name):
    """A greedy export has no beam to fuse into, so an attached LM must not change what it traces."""
    tokenizer = _lm_tokenizer()
    plain = graph_ops(build_model(name, tokenizer), 0)
    fused = graph_ops(_attach_bigram_lm(build_model(name, tokenizer), tokenizer), 0)
    assert plain == fused, "an attached language model leaked into the greedy export"


def loop_node_names(model: BaseModel, beam_width) -> set:
    """Names of the nodes inside the traced `tf.while_loop` bodies, where decoding happens."""
    graph_def = model.make_tflite_function(batch_size=1, beam_width=beam_width).get_concrete_function().graph.as_graph_def()
    return {node.name for function in graph_def.library.function for node in function.node_def}


@pytest.mark.parametrize("name", [_maybe_xfail(name) for name in ["transducer.RnnTransducer"]])
def test_beam_export_with_a_language_model_converts(name):
    """
    The fused beam must survive the converter, and the LM must land inside the decoding loop.

    The LM is called once per step from inside the beam's `tf.while_loop` under
    `jit_compile=True`, which is where this was most likely to break, so it is worth converting
    rather than assuming. `BigramLanguageModel`'s forward pass is a gather over its `table` layer,
    so a `table` node in a loop body is the LM being evaluated per step rather than merely being
    reachable from the graph.
    """
    tokenizer = _lm_tokenizer()
    model = _attach_bigram_lm(build_model(name, tokenizer), tokenizer)

    fused = loop_node_names(model, 2)
    plain = loop_node_names(build_model(name, tokenizer), 2)
    assert not any("table" in n for n in plain), "the plain beam already has a table; this test cannot tell the LM apart"
    assert any("table" in n for n in fused), "the language model is not evaluated inside the beam loop"

    assert len(convert(model, batch_size=1, beam_width=2)) > 0


@pytest.mark.parametrize("name", [_maybe_xfail(name) for name in ["transducer.RnnTransducer"]])
def test_lodr_beam_export_converts(name):
    """Same, for the `"lodr"` correction, which sends a second language model into the loop."""
    tokenizer = _lm_tokenizer()
    tokenizer.decoder_config.lm_type = "lodr"
    tokenizer.decoder_config.lm_beta = 0.2
    model = _attach_bigram_lm(build_model(name, tokenizer), tokenizer, internal=True)

    kwargs = model.get_beam_decoding_kwargs(beam_width=2)
    assert kwargs["internal_lm"] is model.internal_lm and kwargs["lm_beta"] == 0.2
    assert len(convert(model, batch_size=1, beam_width=2)) > 0


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
    """
    Load the flatbuffer into an interpreter, invoke it, and check the output contract.

    Only `transcript` and `tokens` are positionally guaranteed. Fields the model returns as None
    are dropped from the flatbuffer entirely -- CTC has no `next_tokens`, being non-autoregressive
    -- so anything past index 1 differs per architecture and is checked by dtype instead.
    """
    tflite_model = convert(build_model(name, tokenizer), batch_size=1, beam_width=0)
    outputs = invoke(tflite_model, signal)

    assert len(outputs) >= 2, f"expected at least transcript and tokens, got {len(outputs)}"
    transcript, tokens = outputs[0], outputs[1]

    assert transcript.dtype == object or transcript.dtype.type is np.bytes_, f"transcript dtype {transcript.dtype}"
    assert transcript.size == 1, f"one transcript per utterance at batch size 1, got {transcript.shape}"
    assert isinstance(transcript.reshape(-1)[0].decode(), str), "transcript did not decode as text"

    assert tokens.dtype == np.int32, f"tokens dtype {tokens.dtype}"
    assert tokens.ndim == 2 and tokens.shape[0] == 1, f"tokens shape {tokens.shape}"
    assert np.all(tokens >= 0) and np.all(tokens < tokenizer.num_classes), "decoded a token outside the vocabulary"

    # every remaining output is either a token tensor or a carried state, never something else
    for index, output in enumerate(outputs[2:], start=2):
        assert np.issubdtype(output.dtype, np.integer) or np.issubdtype(output.dtype, np.floating), (
            f"output {index} has unexpected dtype {output.dtype}"
        )
        assert np.all(np.isfinite(output)) if np.issubdtype(output.dtype, np.floating) else True


# Architectures whose *beam* export converts but cannot be invoked. Unlike KNOWN_BROKEN these
# convert cleanly, so only the interpreter test is affected.
BEAM_INTERPRETER_BROKEN = {}
"""
Architectures whose *beam* export converts but cannot be invoked.

Empty. It held `transducer.RnnTransducer` until `convert_tflite` began freezing variables to
constants explicitly -- the converter's own pass was giving up on the beam graph and leaving the
LSTM kernels as unassigned resource variables inside the decoding `WHILE`.
"""


def _maybe_xfail_beam_interpreter(name):
    reason = BEAM_INTERPRETER_BROKEN.get(name) or KNOWN_BROKEN.get(name)
    return pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=reason)) if reason else name


@pytest.mark.usefixtures("flex_delegate")
@pytest.mark.parametrize("name", [_maybe_xfail_beam_interpreter(name) for name in TRANSDUCERS])
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
