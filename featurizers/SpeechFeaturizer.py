from __future__ import absolute_import

import soundfile
import numpy as np
import librosa
import tensorflow as tf

WINDOW_FN = {"hanning": np.hanning, "hamming": np.hamming}


def normalize_audio_feature(audio_feature):
    """ Mean and variance normalization """
    mean = np.mean(audio_feature, axis=0)
    std_dev = np.std(audio_feature, axis=0)
    normalized = (audio_feature - mean) / std_dev
    return normalized


def normalize_signal(signal):
    """Normailize signal to [-1, 1] range"""
    gain = 1.0 / (np.max(np.abs(signal)) + 1e-5)
    return signal * gain


class SpeechFeaturizer:
    """A class for extraction speech features"""

    def __init__(self, sample_rate, frame_ms, stride_ms, num_feature_bins, window_fn=WINDOW_FN.get("hanning"),
                 pre_augmentation=None, post_augmentation=None):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.stride_ms = stride_ms
        self.num_feature_bins = num_feature_bins
        self.window_fn = window_fn
        self.pre_augmentation = pre_augmentation
        self.post_augmentation = post_augmentation

    def __augment_audio_signal(self, samples):
        return self.pre_augmentation(signal=samples)

    def __augment_features(self, features):
        return self.post_augmentation(features=features)

    def __compute_spectrogram_feature(self, signal):
        """Function to convert raw audio signal to spectrogram using librosa backend
        Args:
            signal (np.array): np.array containing raw audio signal
        Returns:
            features (np.array): spectrogram of shape=[num_timesteps, num_feature_bins]
            audio_duration (float): duration of the signal in seconds
        """
        n_window_size = int(self.sample_rate * (self.frame_ms / 1000))
        n_window_stride = int(self.sample_rate * (self.stride_ms / 1000))

        powspec = np.square(
            np.abs(
                librosa.core.stft(
                    signal, n_fft=n_window_size,
                    hop_length=n_window_stride, win_length=n_window_size,
                    center=True, window=self.window_fn
                )))

        # remove small bins
        powspec[powspec <= 1e-30] = 1e-30
        features = 10 * np.log10(powspec.T)

        assert self.num_feature_bins <= n_window_size // 2 + 1, \
            "num_features for spectrogram should be <= (sample_rate * window_size // 2 + 1)"

        # cut high frequency part, keep num_feature_bins features
        features = features[:, :self.num_feature_bins]

        return features

    def compute_speech_features(self, audio_file_path):
        """Load audio file, preprocessing, compute features, postprocessing
        Args:
            audio_file_path (string): the path to audio file
        Returns:
            features (np.array): spectrogram of shape=[num_timesteps, num_feature_bins, 1]
            audio_duration (float): duration of the signal in seconds
        """
        if isinstance(audio_file_path, str):
            data, _ = soundfile.read(audio_file_path)
        else:
            data = audio_file_path

        data = normalize_signal(data.astype(np.float32))
        data = self.__compute_spectrogram_feature(data)
        data = normalize_audio_feature(data)

        # Adding Channel dimmension for conv2D input
        data = np.expand_dims(data, axis=2)
        return tf.convert_to_tensor(data)
