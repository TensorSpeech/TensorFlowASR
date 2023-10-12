# pylint: disable=line-too-long
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np

from tensorflow_asr import tf
from tensorflow_asr.models.layers.feature_extraction import FeatureExtraction
from tensorflow_asr.utils import data_util, file_util

# config_path = os.path.join(os.path.dirname(__file__), "..", "examples", "configs", "log_mel_spectrogram.yml.j2")
# config = file_util.load_yaml(config_path)

audio_file_path = os.path.join(os.path.dirname(__file__), "test.flac")


def plot_specs(ft, title):
    ft = ft.numpy() if isinstance(ft, tf.Tensor) else ft
    ft = np.squeeze(ft)
    ft = ft.T
    plt.figure(figsize=(24, 5))
    ax = plt.gca()
    ax.set_title(title, fontweight="bold")
    librosa.display.specshow(ft, cmap="viridis")
    v1 = np.linspace(ft.min(), ft.max(), 8, endpoint=True)
    plt.colorbar(pad=0.01, fraction=0.02, ax=ax, format="%.2f", ticks=v1)
    plt.tight_layout()
    plt.show()


def test_feature_extraction():
    signal = data_util.load_and_convert_to_wav(audio_file_path)
    signal = tf.expand_dims(data_util.read_raw_audio(signal), axis=0)
    signal_length = tf.expand_dims(tf.shape(signal)[1], axis=0)

    feature_extraction_layer = FeatureExtraction(normalize_feature=False)

    for ftype in ("spectrogram", "log_mel_spectrogram", "log_gammatone_spectrogram", "mfcc"):
        feature_extraction_layer.feature_type = ftype
        ft, _ = feature_extraction_layer((signal, signal_length))
        plot_specs(ft, feature_extraction_layer.feature_type)

    mask, _ = feature_extraction_layer.compute_mask((signal, signal_length))
    print(mask)

    feature_extraction_layer.feature_type = "log_mel_spectrogram"
    feature_extraction_layer.preemphasis = 0.0
    ft1, _ = feature_extraction_layer((signal, signal_length))
    feature_extraction_layer.preemphasis = 0.97
    ft2, _ = feature_extraction_layer((signal, signal_length))
    ft = ft1 - ft2
    plot_specs(ft, feature_extraction_layer.feature_type)
