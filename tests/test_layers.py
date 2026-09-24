"""
Tests for the `FeatureExtraction` layer.

The previous version of this file computed features and rendered each one with
`matplotlib.pyplot.show()`, which blocks on an interactive backend -- running the suite hung here
indefinitely rather than failing. It also asserted nothing.

The plotting is kept commented out below: it is how the figures in `docs/features.md` were
produced, so it is worth keeping around as a developer aid, but it must never run as part of the
suite.
"""

import os

import numpy as np
import pytest

from tensorflow_asr import tf
from tensorflow_asr.augmentations.augmentation import Augmentation
from tensorflow_asr.models.layers.feature_extraction import FeatureExtraction
from tensorflow_asr.utils import data_util

AUDIO_FILE_PATH = os.path.join(os.path.dirname(__file__), "test.flac")
FEATURE_TYPES = ("spectrogram", "log_mel_spectrogram", "log_gammatone_spectrogram", "mfcc")
PADDING = 16000


def plot_specs(features, title):  # pylint: disable=unused-argument
    """
    Spectrogram plot, disabled. Uncomment the body (and the imports it needs) to regenerate the
    figures in `docs/features.md`. It saves rather than shows: `plt.show()` blocks on a GUI
    backend and would hang the suite.
    """
    # import librosa.display
    # import matplotlib.pyplot as plt
    #
    # from tensorflow_asr.utils import plot_util
    #
    # features = features.numpy() if isinstance(features, tf.Tensor) else features
    # features = np.squeeze(features).T
    # figure = plt.figure(figsize=(24, 5))
    # axes = plt.gca()
    # axes.set_title(title, fontweight="bold")
    # librosa.display.specshow(features, cmap="viridis")
    # ticks = np.linspace(features.min(), features.max(), 8, endpoint=True)
    # plt.colorbar(pad=0.01, fraction=0.02, ax=axes, format="%.2f", ticks=ticks)
    # plt.tight_layout()
    # plt.savefig(plot_util.get_plot_path(title))
    # plt.close(figure)


@pytest.fixture(scope="module")
def signal():
    """The fixture audio, padded so the unpadded length is distinguishable from the total."""
    raw = data_util.load_and_convert_to_wav(AUDIO_FILE_PATH)
    raw = tf.expand_dims(data_util.read_raw_audio(raw), axis=0)
    length = tf.expand_dims(tf.shape(raw)[1], axis=0)
    return tf.pad(raw, paddings=[[0, 0], [0, PADDING]], mode="CONSTANT", constant_values=0.0), length


@pytest.mark.parametrize("feature_type", FEATURE_TYPES)
def test_feature_extraction_shapes_and_values(signal, feature_type):
    inputs, inputs_length = signal
    layer = FeatureExtraction()
    layer.feature_type = feature_type
    features, features_length = layer((inputs, inputs_length))

    assert features.shape[0] == 1
    assert features.shape[2] == layer.num_feature_bins
    assert features.shape[3] == 1, "only the first channel is implemented"
    assert bool(tf.reduce_all(tf.math.is_finite(features))), f"{feature_type} produced non-finite values"
    # the padding must not be counted as signal
    assert 0 < int(features_length[0]) < features.shape[1]
    # plot_specs(features, feature_type)


def test_feature_types_are_actually_different(signal):
    """Guards against the feature_type switch silently falling through to a single branch."""
    inputs, inputs_length = signal
    layer = FeatureExtraction()
    outputs = {}
    for feature_type in FEATURE_TYPES:
        layer.feature_type = feature_type
        features, _ = layer((inputs, inputs_length))
        outputs[feature_type] = features.numpy()

    for a, b in zip(FEATURE_TYPES, FEATURE_TYPES[1:]):
        assert not np.allclose(outputs[a], outputs[b]), f"{a} and {b} produced identical features"


def test_mask_covers_exactly_the_unpadded_frames(signal):
    inputs, inputs_length = signal
    layer = FeatureExtraction()
    mask, _ = layer.compute_mask((inputs, inputs_length))
    _, features_length = layer((inputs, inputs_length))
    nframes = int(features_length[0])

    assert mask.dtype == tf.bool
    assert mask.shape[0] == 1
    assert int(tf.reduce_sum(tf.cast(mask, tf.int32))) == nframes
    # the mask is a prefix: every true frame precedes every false one
    assert bool(tf.reduce_all(mask[0, :nframes]))
    assert not bool(tf.reduce_any(mask[0, nframes:]))


def test_preemphasis_changes_the_features(signal):
    inputs, inputs_length = signal
    layer = FeatureExtraction()
    layer.feature_type = "log_mel_spectrogram"

    layer.preemphasis = 0.0
    without, _ = layer((inputs, inputs_length))
    layer.preemphasis = 0.97
    with_preemphasis, _ = layer((inputs, inputs_length))

    assert without.shape == with_preemphasis.shape
    assert not np.allclose(without.numpy(), with_preemphasis.numpy()), "preemphasis had no effect"
    # plot_specs(without - with_preemphasis, "preemphasis_difference")


def test_augmentation_at_zero_probability_is_a_no_op(signal):
    """prob=0.0 must leave the features untouched even when training=True."""
    inputs, inputs_length = signal
    layer = FeatureExtraction()
    layer.feature_type = "log_mel_spectrogram"
    layer.preemphasis = 0.0
    baseline, _ = layer((inputs, inputs_length), training=False)

    layer.augmentations = Augmentation(
        {
            "feature_augment": {
                "freq_masking": {"num_masks": 2, "mask_factor": 27, "prob": 0.0, "mask_value": 0},
                "time_masking": {"num_masks": 2, "mask_factor": -1, "prob": 0.0, "mask_value": 0, "p_upperbound": 0.05},
            }
        }
    )
    augmented, _ = layer((inputs, inputs_length), training=True)

    assert augmented.shape == baseline.shape
    assert np.allclose(augmented.numpy(), baseline.numpy()), "augmentation ran despite prob=0.0"


def test_augmentation_at_full_probability_masks_features(signal):
    inputs, inputs_length = signal
    layer = FeatureExtraction()
    layer.feature_type = "log_mel_spectrogram"
    layer.preemphasis = 0.0
    baseline, _ = layer((inputs, inputs_length), training=False)

    layer.augmentations = Augmentation(
        {
            "feature_augment": {
                "freq_masking": {"num_masks": 2, "mask_factor": 27, "prob": 1.0, "mask_value": 0},
            }
        }
    )
    augmented, _ = layer((inputs, inputs_length), training=True)

    assert augmented.shape == baseline.shape
    assert not np.allclose(augmented.numpy(), baseline.numpy()), "freq masking at prob=1.0 changed nothing"
    # plot_specs(augmented, "freq_masked")
