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
import tensorflow as tf


def read_raw_audio(audio, sample_rate=16000):
    if isinstance(audio, str):
        wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate)
    elif isinstance(audio, bytes):
        wave, sr = sf.read(io.BytesIO(audio))
        wave = np.asfortranarray(wave)
        if sr != sample_rate:
            wave = librosa.resample(wave, sr, sample_rate)
    elif isinstance(audio, np.ndarray):
        return audio
    else:
        raise ValueError("input audio must be either a path or bytes")
    return wave


def slice_signal(signal, window_size, stride=0.5) -> np.ndarray:
    """ Return windows of the given signal by sweeping in stride fractions of window """
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    offset = int(window_size * stride)
    slices = []
    for beg_i, end_i in zip(range(0, n_samples, offset),
                            range(window_size, n_samples + offset,
                                  offset)):
        slice_ = signal[beg_i:end_i]
        if slice_.shape[0] < window_size:
            slice_ = np.pad(
                slice_, (0, window_size - slice_.shape[0]), 'constant', constant_values=0.0)
        if slice_.shape[0] == window_size:
            slices.append(slice_)
    return np.array(slices, dtype=np.float32)


def tf_merge_slices(slices: tf.Tensor) -> tf.Tensor:
    # slices shape = [batch, window_size]
    return tf.keras.backend.flatten(slices)  # return shape = [-1, ]


def merge_slices(slices: np.ndarray) -> np.ndarray:
    # slices shape = [batch, window_size]
    return np.reshape(slices, [-1])


def normalize_audio_feature(audio_feature: np.ndarray, per_feature=False):
    """ Mean and variance normalization """
    axis = 0 if per_feature else None
    mean = np.mean(audio_feature, axis=axis)
    std_dev = np.std(audio_feature, axis=axis) + 1e-9
    normalized = (audio_feature - mean) / std_dev
    return normalized


def tf_normalize_audio_features(audio_feature: tf.Tensor, per_feature=False):
    """
    TF Mean and variance features normalization
    Args:
        audio_feature: tf.Tensor with shape [T, F]

    Returns:
        normalized audio features with shape [T, F]
    """
    axis = 0 if per_feature else None
    mean = tf.reduce_mean(audio_feature, axis=axis)
    std_dev = tf.math.reduce_std(audio_feature, axis=axis) + 1e-9
    return (audio_feature - mean) / std_dev


def normalize_signal(signal: np.ndarray):
    """ Normailize signal to [-1, 1] range """
    gain = 1.0 / (np.max(np.abs(signal)) + 1e-9)
    return signal * gain


def tf_normalize_signal(signal: tf.Tensor):
    """
    TF Normailize signal to [-1, 1] range
    Args:
        signal: tf.Tensor with shape [None]

    Returns:
        normalized signal with shape [None]
    """
    gain = 1.0 / (tf.reduce_max(tf.abs(signal), axis=-1) + 1e-9)
    return signal * gain


def preemphasis(signal: np.ndarray, coeff=0.97):
    if not coeff or coeff <= 0.0:
        return signal
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def tf_preemphasis(signal: tf.Tensor, coeff=0.97):
    """
    TF Pre-emphasis
    Args:
        signal: tf.Tensor with shape [None]
        coeff: Float that indicates the preemphasis coefficient

    Returns:
        pre-emphasized signal with shape [None]
    """
    if not coeff or coeff <= 0.0: return signal
    s0 = tf.expand_dims(signal[0], axis=-1)
    s1 = signal[1:] - coeff * signal[:-1]
    return tf.concat([s0, s1], axis=-1)


def deemphasis(signal: np.ndarray, coeff=0.97):
    if not coeff or coeff <= 0.0: return signal
    x = np.zeros(signal.shape[0], dtype=np.float32)
    x[0] = signal[0]
    for n in range(1, signal.shape[0], 1):
        x[n] = coeff * x[n - 1] + signal[n]
    return x


def tf_depreemphasis(signal: tf.Tensor, coeff=0.97):
    """
    TF Depreemphasis
    Args:
        signal: tf.Tensor with shape [B, None]
        coeff: Float that indicates the preemphasis coefficient

    Returns:
        depre-emphasized signal with shape [B, None]
    """
    if not coeff or coeff <= 0.0: return signal

    def map_fn(elem):
        x = tf.expand_dims(elem[0], axis=-1)
        for n in range(1, elem.shape[0], 1):
            current = coeff * x[n - 1] + elem[n]
            x = tf.concat([x, [current]], axis=0)
        return x

    return tf.map_fn(map_fn, signal)


class SpeechFeaturizer:
    def __init__(self, speech_config: dict):
        """
        We should use TFSpeechFeaturizer for training to avoid differences
        between tf and librosa when converting to tflite in post-training stage
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

    def compute_time_dim(self, seconds: float) -> int:
        # implementation using pad "reflect" with n_fft // 2
        total_frames = seconds * self.sample_rate + 2 * (self.frame_length // 2)
        return int(1 + (total_frames - self.frame_length) // self.frame_step)

    def compute_feature_shape(self) -> list:
        # None for time dimension
        channel_dim = 1

        if self.delta:
            channel_dim += 1

        if self.delta_delta:
            channel_dim += 1

        if self.pitch:
            channel_dim += 1

        return [None, self.num_feature_bins, channel_dim]

    def extract(self, signal: np.ndarray) -> np.ndarray:
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
                delta_delta = normalize_audio_feature(
                    delta_delta, per_feature=self.normalize_per_feature)
            features = np.concatenate([features, np.expand_dims(delta_delta, axis=-1)], axis=-1)

        if self.pitch:
            pitches = self._compute_pitch_feature(signal)
            if self.normalize_feature:
                pitches = normalize_audio_feature(
                    pitches, per_feature=self.normalize_per_feature)
            features = np.concatenate([features, np.expand_dims(pitches, axis=-1)], axis=-1)

        return features

    def _compute_pitch_feature(self, signal: np.ndarray) -> np.ndarray:
        pitches, _ = librosa.core.piptrack(
            y=signal, sr=self.sample_rate,
            n_fft=self.frame_length, hop_length=self.frame_step,
            fmin=0, fmax=int(self.sample_rate / 2), win_length=self.frame_length, center=True
        )

        pitches = pitches.T

        assert self.num_feature_bins <= self.frame_length // 2 + 1, \
            "num_features for spectrogram should \
        be <= (sample_rate * window_size // 2 + 1)"

        return pitches[:, :self.num_feature_bins]

    def _compute_spectrogram_feature(self, signal: np.ndarray) -> np.ndarray:
        powspec = np.abs(librosa.core.stft(signal, n_fft=self.frame_length,
                                           hop_length=self.frame_step,
                                           win_length=self.frame_length, center=True))

        # remove small bins
        features = 20 * np.log10(powspec.T)

        assert self.num_feature_bins <= self.frame_length // 2 + 1, \
            "num_features for spectrogram should \
        be <= (sample_rate * window_size // 2 + 1)"

        # cut high frequency part, keep num_feature_bins features
        features = features[:, :self.num_feature_bins]

        return features

    def _compute_mfcc_feature(self, signal: np.ndarray) -> np.ndarray:
        S = np.square(
            np.abs(
                librosa.core.stft(
                    signal, n_fft=self.frame_length, hop_length=self.frame_step,
                    win_length=self.frame_length, center=True
                )))

        mel_basis = librosa.filters.mel(self.sample_rate, self.frame_length,
                                        n_mels=128,
                                        fmin=0, fmax=int(self.sample_rate / 2))

        mfcc = librosa.feature.mfcc(sr=self.sample_rate,
                                    S=librosa.core.power_to_db(np.dot(mel_basis, S) + 1e-20),
                                    n_mfcc=self.num_feature_bins)

        return mfcc.T

    def _compute_logfbank_feature(self, signal: np.ndarray) -> np.ndarray:
        S = np.square(np.abs(librosa.core.stft(signal, n_fft=self.frame_length,
                                               hop_length=self.frame_step,
                                               win_length=self.frame_length, center=True)))

        mel_basis = librosa.filters.mel(self.sample_rate, self.frame_length,
                                        n_mels=self.num_feature_bins,
                                        fmin=0, fmax=int(self.sample_rate / 2))

        return np.log(np.dot(mel_basis, S) + 1e-20).T


class TFSpeechFeaturizer:
    def __init__(self, speech_config: dict):
        """
        TF Speech Featurizer does not support delta, delta's deltas and pitch features yet
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
        This class is mainly for tflite
        (because tflite doesnt support py_func so we cant use librosa)
        """
        # Samples
        self.sample_rate = speech_config["sample_rate"]
        self.frame_length = int(self.sample_rate * (speech_config["frame_ms"] / 1000))
        self.frame_step = int(self.sample_rate * (speech_config["stride_ms"] / 1000))
        # Features
        self.num_feature_bins = speech_config["num_feature_bins"]
        self.feature_type = speech_config["feature_type"]
        for k in ["delta", "delta_delta", "pitch"]:
            if speech_config.get(k, None) is not None:
                raise ValueError("delta, delta_delta, pitch not supported yet")
        # self.delta = speech_config["delta"]
        # self.delta_delta = speech_config["delta_delta"]
        # self.pitch = speech_config["pitch"]
        self.preemphasis = speech_config["preemphasis"]
        # Normalization
        self.normalize_signal = speech_config["normalize_signal"]
        self.normalize_feature = speech_config["normalize_feature"]
        self.normalize_per_feature = speech_config["normalize_per_feature"]
        self.num_fft = self.frame_length

    def __compute_spectrogram(self, signal):
        return tf.abs(tf.signal.stft(signal, frame_length=self.frame_length,
                                     frame_step=self.frame_step, fft_length=self.num_fft))

    def __compute_logfbank(self, signal):
        spectrogram = self.__compute_spectrogram(signal)
        linear_to_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_feature_bins,
            num_spectrogram_bins=spectrogram.shape[-1],
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0, upper_edge_hertz=(self.sample_rate / 2)
        )
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_weight_matrix, 1)
        mel_spectrogram.set_shape(
            spectrogram.shape[:-1].concatenate(linear_to_weight_matrix.shape[-1:]))
        return tf.math.log(mel_spectrogram + 1e-6)

    def compute_feature_shape(self) -> list:
        # None for time dimension
        return [None, self.num_feature_bins, 1]

    def compute_tf_spectrogram_features(self, signal):
        spectrogram = self.__compute_spectrogram(signal)
        spectrogram = spectrogram[:, :self.num_feature_bins]
        return tf.expand_dims(spectrogram, axis=-1)

    def compute_tf_logfbank_features(self, signal):
        log_mel_spectrogram = self.__compute_logfbank(signal)
        return tf.expand_dims(log_mel_spectrogram, axis=-1)

    def compute_tf_mfcc_features(self, signal):
        log_mel_spectrogram = self.__compute_logfbank(signal)
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        return tf.expand_dims(mfcc, axis=-1)

    def tf_extract(self, signal: tf.Tensor) -> tf.Tensor:
        """
        Extract speech features from signals (for using in tflite)
        Args:
            signal: tf.Tensor with shape [None]

        Returns:
            features: tf.Tensor with shape [T, F, 1]
        """
        if self.normalize_signal:
            signal = tf_normalize_signal(signal)
        signal = tf_preemphasis(signal, self.preemphasis)

        if self.feature_type == "mfcc":
            features = self.compute_tf_mfcc_features(signal)
        elif self.feature_type == "spectrogram":
            features = self.compute_tf_spectrogram_features(signal)
        elif self.feature_type == "logfbank":
            features = self.compute_tf_logfbank_features(signal)

        if self.normalize_feature:
            features = tf_normalize_audio_features(
                features, per_feature=self.normalize_per_feature)

        return features

    def extract(self, signal: np.ndarray) -> np.ndarray:
        with tf.device("/CPU:0"):  # Use in tf.data => avoid error copying
            features = self.tf_extract(tf.convert_to_tensor(signal, dtype=tf.float32))
        return features.numpy()
