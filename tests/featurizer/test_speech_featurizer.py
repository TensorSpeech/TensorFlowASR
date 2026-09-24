"""
Feature extraction, SpecAugment and chunked (streaming) extraction.

This was an exploratory script, fully commented out, written against an API that no longer exists:
`SpeechConfig`, `tensorflow_asr.features.speech_featurizers.SpeechFeaturizer`, `sf.extract()`,
`sf.speech_config.normalize_per_frame`. All of that is now the `FeatureExtraction` keras layer.

It is uncommented and turned into real tests here. Its most interesting question -- does extracting
features chunk by chunk give the same answer as extracting them from the whole signal? -- is now
asserted rather than eyeballed on a plot. The figures it used to `plt.show()` are written through
`plot_util`, which puts them in `$TFASR_PLOT_DIR` -- `tests/figs/`, set by `tests/conftest.py`.
"""

import os

import numpy as np
import pytest

from tensorflow_asr import tf
from tensorflow_asr.augmentations.methods import specaugment
from tensorflow_asr.models.layers.feature_extraction import FeatureExtraction
from tensorflow_asr.utils import data_util, plot_util

AUDIO_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test.flac")


@pytest.fixture(scope="module")
def signal():
    raw = data_util.load_and_convert_to_wav(AUDIO_FILE_PATH)
    return data_util.read_raw_audio(raw)


def extract(layer, samples):
    """Run the layer over a 1-D signal and drop the batch axis."""
    features, features_length = layer((tf.expand_dims(samples, 0), tf.expand_dims(tf.shape(samples)[0], 0)))
    return tf.squeeze(features, 0), features_length[0]


def as_image(features):
    """[T, num_feature_bins, 1] -> [num_feature_bins, T], the orientation the plots expect."""
    return np.squeeze(features.numpy() if isinstance(features, tf.Tensor) else features).T


def test_extracts_log_mel_spectrogram(signal):
    layer = FeatureExtraction(feature_type="log_mel_spectrogram", preemphasis=0.0)
    features, length = extract(layer, signal)

    assert features.shape[1] == layer.num_feature_bins
    assert features.shape[2] == 1
    assert int(length) == features.shape[0]
    assert bool(tf.reduce_all(tf.math.is_finite(features)))

    path = plot_util.plotmesh(as_image(features), title="log_mel_spectrogram")
    assert os.path.getsize(path) > 0


@pytest.mark.parametrize("mask_value", ["zero", "min"])
def test_specaugment_masks_features(signal, mask_value):
    layer = FeatureExtraction(feature_type="log_mel_spectrogram", preemphasis=0.0)
    features, length = extract(layer, signal)

    masked, _ = specaugment.FreqMasking(prob=1.0, num_masks=2, mask_factor=27, mask_value=mask_value).augment((features, length))
    masked, _ = specaugment.TimeMasking(prob=1.0, num_masks=2, p_upperbound=0.05, mask_value=mask_value).augment((masked, length))

    assert masked.shape == features.shape
    assert not np.allclose(masked.numpy(), features.numpy()), "masking at prob=1.0 changed nothing"
    if mask_value == "zero":
        assert (masked.numpy() == 0).any()
    else:
        # masking with "min" can never introduce a value below the original minimum
        assert masked.numpy().min() >= features.numpy().min() - 1e-5

    path = plot_util.plotmesh(as_image(masked), title=f"specaugment_{mask_value}")
    assert os.path.getsize(path) > 0


def test_normalization_options(signal):
    """`normalize_per_frame` is gone; z-score and min-max are what the layer offers now."""
    common = dict(feature_type="log_mel_spectrogram", preemphasis=0.0)
    plain, _ = extract(FeatureExtraction(**common), signal)
    zscore, _ = extract(FeatureExtraction(**common, normalize_zscore=True), signal)
    min_max, _ = extract(FeatureExtraction(**common, normalize_min_max=True), signal)

    assert zscore.shape == plain.shape == min_max.shape
    assert np.isclose(zscore.numpy().mean(), 0.0, atol=1e-4)
    assert np.isclose(zscore.numpy().std(), 1.0, atol=1e-4)
    assert np.isclose(min_max.numpy().max(), 1.0, atol=1e-4)
    assert min_max.numpy().min() >= -1.0


@pytest.mark.parametrize("nframes", [1, 5, 10])
def test_chunked_extraction_matches_whole_signal(signal, nframes):
    """
    Streaming inference feeds the layer one chunk at a time, so chunked features must agree with
    the features of the whole signal.

    `get_signal_chunk_size_and_step` sizes the chunks so each yields exactly `nframes` frames, but
    that only holds with `pad_end=False` -- otherwise every chunk is zero-padded to a whole frame
    and produces extra ones. Preemphasis is off for the same reason: it reads one sample back, so
    it would differ across a chunk boundary.
    """
    layer = FeatureExtraction(feature_type="log_mel_spectrogram", preemphasis=0.0, pad_end=False)
    whole, _ = extract(layer, signal)
    whole = whole.numpy()

    chunk_size, chunk_step = layer.get_signal_chunk_size_and_step(nframes)
    assert chunk_size == (nframes - 1) * layer.frame_step + layer.frame_length

    chunks = []
    start = 0
    while start + chunk_size <= int(signal.shape[0]):
        chunk_features, _ = extract(layer, signal[start : start + chunk_size])
        assert chunk_features.shape[0] == nframes, "chunk did not yield exactly nframes frames"
        chunks.append(chunk_features.numpy())
        start += chunk_step

    chunked = np.concatenate(chunks, axis=0)
    compared = min(len(chunked), len(whole))
    assert compared > 0
    # only the tail shorter than one chunk is dropped
    assert 0 <= len(whole) - compared <= nframes

    difference = chunked[:compared] - whole[:compared]
    assert np.abs(difference).max() < 1e-3, f"chunked extraction drifted by {np.abs(difference).max()}"

    plot_util.plotmesh(as_image(difference), title=f"chunked_minus_whole_{nframes}")
    rmse = np.sqrt(np.mean(np.squeeze(difference) ** 2, axis=-1))
    path = plot_util.plotline(rmse, title=f"chunked_rmse_{nframes}")
    assert os.path.getsize(path) > 0


def test_chunked_extraction_with_pad_end_produces_extra_frames(signal):
    """The counterpart of the note above: with `pad_end=True` the chunk sizing no longer holds."""
    nframes = 5
    layer = FeatureExtraction(feature_type="log_mel_spectrogram", preemphasis=0.0, pad_end=True)
    chunk_size, _ = layer.get_signal_chunk_size_and_step(nframes)
    chunk_features, _ = extract(layer, signal[:chunk_size])
    assert chunk_features.shape[0] > nframes
