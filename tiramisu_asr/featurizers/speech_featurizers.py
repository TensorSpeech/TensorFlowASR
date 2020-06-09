# Copyright 2020 Huy Le Nguyen (@usimarit)
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
import os
import io
import numpy as np
import librosa
import soundfile as sf


# import math
# import tensorflow as tf


def read_raw_audio(audio, sample_rate=16000):
    if isinstance(audio, str):
        wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate)
    elif isinstance(audio, bytes):
        wave, sr = sf.read(io.BytesIO(audio))
        if sr != sample_rate:
            wave = librosa.resample(wave, sr, sample_rate)
    else:
        raise ValueError("input audio must be either a path or bytes")
    return wave


def normalize_audio_feature(audio_feature: np.ndarray, per_feature=False):
    """ Mean and variance normalization """
    axis = 0 if per_feature else None
    mean = np.mean(audio_feature, axis=axis)
    std_dev = np.std(audio_feature, axis=axis) + 1e-6
    normalized = (audio_feature - mean) / std_dev
    return normalized


def normalize_signal(signal: np.ndarray):
    """Normailize signal to [-1, 1] range"""
    gain = 1.0 / (np.max(np.abs(signal)) + 1e-5)
    return signal * gain


def preemphasis(signal: np.ndarray, coeff=0.97):
    if not coeff or coeff == 0.0:
        return signal
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def deemphasis(signal: np.ndarray, coeff=0.97):
    if coeff <= 0:
        return signal
    x = np.zeros(signal.shape[0], dtype=np.float32)
    x[0] = signal[0]
    for n in range(1, signal.shape[0], 1):
        x[n] = coeff * x[n - 1] + signal[n]
    return x


class SpeechFeaturizer:
    def __init__(self, speech_config: dict):
        """
        speech_config = {
            "sample_rate": int,
            "frame_ms": int,
            "stride_ms": int,
            "num_feature_bins": int,
            "feature_type": "spectrogram", "logfbank" or "mfcc",
            "delta": bool,
            "delta_delta": bool,
            "pitch": bool,
            "normalize_signal": bool,
            "normalize_feature": bool,
            "normalize_per_feature": bool
        }
        """
        # Samples
        self.sample_rate = speech_config["sample_rate"]
        self.frame_length = int(self.sample_rate * (speech_config["frame_ms"] / 1000))
        self.frame_step = int(self.sample_rate * (speech_config["stride_ms"] / 1000))
        # Features
        self.num_feature_bins = speech_config["num_feature_bins"]
        self.feature_type = speech_config["feature_type"]
        self.delta = speech_config["delta"]
        self.delta_delta = speech_config["delta_delta"]
        self.pitch = speech_config["pitch"]
        self.preemphasis = speech_config["preemphasis"]
        # Normalization
        self.normalize_signal = speech_config["normalize_signal"]
        self.normalize_feature = speech_config["normalize_feature"]
        self.normalize_per_feature = speech_config["normalize_per_feature"]

    def compute_time_dim(self, seconds):
        # implementation using pad "reflect" with n_fft // 2
        return int(seconds * (1 + ((self.sample_rate + 2 * (self.frame_length // 2)) - self.frame_length) // self.frame_step))

    def compute_feature_dim(self):
        channel_dim = 1

        if self.delta:
            channel_dim += 1

        if self.delta_delta:
            channel_dim += 1

        if self.pitch:
            channel_dim += 1

        return self.num_feature_bins, channel_dim

    def extract(self, signal: np.ndarray):
        if self.normalize_signal:
            signal = normalize_signal(signal)
        signal = preemphasis(signal, self.preemphasis)

        if self.feature_type == "mfcc":
            features = self._compute_mfcc_feature(signal)
        elif self.feature_type == "logfbank":
            features = self._compute_logfbank_feature(signal)
        elif self.feature_type == "spectrogram":
            features = self._compute_spectrogram_feature(signal)
        else:
            raise ValueError("feature_type must be either 'mfcc', 'logfbank' or 'spectrogram'")

        original_features = np.copy(features)

        if self.normalize_feature:
            features = normalize_audio_feature(features, per_feature=self.normalize_per_feature)

        features = np.expand_dims(features, axis=-1)

        if self.delta:
            delta = librosa.feature.delta(original_features.T).T
            if self.normalize_feature:
                delta = normalize_audio_feature(delta, per_feature=self.normalize_per_feature)
            features = np.concatenate([features, np.expand_dims(delta, axis=-1)], axis=-1)

        if self.delta_delta:
            delta_delta = librosa.feature.delta(original_features.T, order=2).T
            if self.normalize_feature:
                delta_delta = normalize_audio_feature(delta_delta, per_feature=self.normalize_per_feature)
            features = np.concatenate([features, np.expand_dims(delta_delta, axis=-1)], axis=-1)

        if self.pitch:
            pitches = self._compute_pitch_feature(signal)
            if self.normalize_feature:
                pitches = normalize_audio_feature(pitches, per_feature=self.normalize_per_feature)
            features = np.concatenate([features, np.expand_dims(pitches, axis=-1)], axis=-1)

        return features

    def _compute_pitch_feature(self, signal: np.ndarray):
        pitches, _ = librosa.core.piptrack(y=signal, sr=self.sample_rate,
                                           n_fft=self.frame_length, hop_length=self.frame_step,
                                           fmin=0, fmax=int(self.sample_rate / 2), win_length=self.frame_length, center=True)

        pitches = pitches.T

        assert self.num_feature_bins <= self.frame_length // 2 + 1, \
            "num_features for spectrogram should \
        be <= (sample_rate * window_size // 2 + 1)"

        return pitches[:, :self.num_feature_bins]

    def _compute_spectrogram_feature(self, signal: np.ndarray):
        powspec = np.abs(librosa.core.stft(signal, n_fft=self.frame_length, hop_length=self.frame_step,
                                           win_length=self.frame_length, center=True))

        # remove small bins
        features = 20 * np.log10(powspec.T)

        assert self.num_feature_bins <= self.frame_length // 2 + 1, \
            "num_features for spectrogram should \
        be <= (sample_rate * window_size // 2 + 1)"

        # cut high frequency part, keep num_feature_bins features
        features = features[:, :self.num_feature_bins]

        return features

    def _compute_mfcc_feature(self, signal: np.ndarray):
        S = np.square(
            np.abs(
                librosa.core.stft(
                    signal, n_fft=self.frame_length, hop_length=self.frame_step,
                    win_length=self.frame_length, center=True
                )))

        mel_basis = librosa.filters.mel(self.sample_rate, self.frame_length,
                                        n_mels=128,
                                        fmin=0, fmax=int(self.sample_rate / 2))

        mfcc = librosa.feature.mfcc(sr=self.sample_rate, S=librosa.core.power_to_db(np.dot(mel_basis, S) + 1e-20),
                                    n_mfcc=self.num_feature_bins)

        return mfcc.T

    def _compute_logfbank_feature(self, signal: np.ndarray):
        S = np.square(np.abs(librosa.core.stft(signal, n_fft=self.frame_length, hop_length=self.frame_step,
                                               win_length=self.frame_length, center=True)))

        mel_basis = librosa.filters.mel(self.sample_rate, self.frame_length,
                                        n_mels=self.num_feature_bins,
                                        fmin=0, fmax=int(self.sample_rate / 2))

        return np.log(np.dot(mel_basis, S) + 1e-20).T

# class SpeechFeaturizerKeras(tf.keras.layers.Layer):
#     """A class for extraction speech features"""
#
#     def __init__(self, sample_rate, frame_ms, stride_ms, num_feature_bins,
#                  feature_type="mfcc", name="speech_featurizer", **kwargs):
#         self.sample_rate = sample_rate
#         self.num_feature_bins = num_feature_bins
#         self.feature_type = feature_type
#         self.frame_length = int(self.sample_rate * (frame_ms / 1000))
#         self.frame_step = int(self.sample_rate * (stride_ms / 1000))
#         self.num_fft = 2 ** math.ceil(math.log2(self.frame_length))
#         super(SpeechFeaturizerKeras, self).__init__(name=name, trainable=False, **kwargs)
#
#     def __compute_spectrogram(self, signal):
#         return tf.abs(tf.signal.stft(signal, frame_length=self.frame_length,
#                                      frame_step=self.frame_step, fft_length=self.num_fft))
#
#     def __compute_logfbank(self, signal):
#         spectrogram = self.__compute_spectrogram(signal)
#         linear_to_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
#             num_mel_bins=self.num_feature_bins,
#             num_spectrogram_bins=spectrogram.shape[-1],
#             sample_rate=self.sample_rate,
#             lower_edge_hertz=80.0, upper_edge_hertz=7600.0
#         )
#         mel_spectrogram = tf.tensordot(spectrogram, linear_to_weight_matrix, 1)
#         mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_weight_matrix.shape[-1:]))
#         return tf.math.log(mel_spectrogram + 1e-6)
#
#     def compute_tf_spectrogram_features(self, signal):
#         spectrogram = self.__compute_spectrogram(signal)
#         spectrogram = spectrogram[:, :self.num_feature_bins]
#         return tf.expand_dims(spectrogram, -1)
#
#     def compute_tf_logfbank_features(self, signal):
#         log_mel_spectrogram = self.__compute_logfbank(signal)
#         log_mel_spectrogram = log_mel_spectrogram[:, :self.num_feature_bins]
#         return tf.expand_dims(log_mel_spectrogram, -1)
#
#     def compute_tf_mfcc_features(self, signal):
#         log_mel_spectrogram = self.__compute_logfbank(signal)
#         mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
#         return tf.expand_dims(mfcc, -1)
#
#     def call(self, signal, **kwargs):
#         if self.feature_type == "mfcc":
#             return self.compute_tf_mfcc_features(signal)
#         elif self.feature_type == "spectrogram":
#             return self.compute_tf_spectrogram_features(signal)
#         elif self.feature_type == "logfbank":
#             return self.compute_tf_logfbank_features(signal)
#
#     def get_config(self):
#         config = super(SpeechFeaturizerKeras, self).get_config()
#         config.update({'sample_rate':      self.sample_rate,
#                        'num_feature_bins': self.num_feature_bins,
#                        'frame_length':     self.frame_length,
#                        'frame_step':       self.frame_step,
#                        'num_fft':          self.num_fft})
#         return config
#
#     def from_config(self, config):
#         return self(**config)
