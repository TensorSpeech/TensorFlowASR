# Copyright 2020 Huy Le Nguyen (@usimarit) and Huy Phan (@pquochuy)
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
import abc
import six
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf

from ..utils.utils import log10
from .gammatone import fft_weights


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


class SpeechFeaturizer(metaclass=abc.ABCMeta):
    def __init__(self, speech_config: dict):
        """
        We should use TFSpeechFeaturizer for training to avoid differences
        between tf and librosa when converting to tflite in post-training stage
        speech_config = {
            "sample_rate": int,
            "frame_ms": int,
            "stride_ms": int,
            "num_feature_bins": int,
            "feature_type": str,
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
        self.preemphasis = speech_config["preemphasis"]
        # Normalization
        self.normalize_signal = speech_config["normalize_signal"]
        self.normalize_feature = speech_config["normalize_feature"]
        self.normalize_per_feature = speech_config["normalize_per_feature"]

    @property
    def nfft(self) -> int:
        """ Number of FFT """
        return 2 ** (self.frame_length - 1).bit_length()

    @property
    def shape(self) -> list:
        """ The shape of extracted features """
        raise NotImplementedError()

    @abc.abstractclassmethod
    def stft(self, signal):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def power_to_db(self, S, ref=1.0, amin=1e-10, top_db=80.0):
        raise NotImplementedError()

    @abc.abstractmethod
    def extract(self, signal):
        """ Function to perform feature extraction """
        raise NotImplementedError()


class NumpySpeechFeaturizer(SpeechFeaturizer):
    def __init__(self, speech_config: dict):
        super(NumpySpeechFeaturizer, self).__init__(speech_config)
        self.delta = speech_config.get("delta", False)
        self.delta_delta = speech_config.get("delta_delta", False)
        self.pitch = speech_config.get("pitch", False)

    @property
    def shape(self) -> list:
        # None for time dimension
        channel_dim = 1

        if self.delta:
            channel_dim += 1

        if self.delta_delta:
            channel_dim += 1

        if self.pitch:
            channel_dim += 1

        return [None, self.num_feature_bins, channel_dim]

    def stft(self, signal):
        return np.square(
            np.abs(librosa.core.stft(signal, n_fft=self.nfft, hop_length=self.frame_step,
                                     win_length=self.frame_length, center=True, window="hann")))

    def power_to_db(self, S, ref=1.0, amin=1e-10, top_db=80.0):
        return librosa.power_to_db(S, ref=ref, amin=amin, top_db=top_db)

    def extract(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asfortranarray(signal)
        if self.normalize_signal:
            signal = normalize_signal(signal)
        signal = preemphasis(signal, self.preemphasis)

        if self.feature_type == "mfcc":
            features = self.compute_mfcc(signal)
        elif self.feature_type == "log_mel_spectrogram":
            features = self.compute_log_mel_spectrogram(signal)
        elif self.feature_type == "spectrogram":
            features = self.compute_spectrogram(signal)
        elif self.feature_type == "log_gammatone_spectrogram":
            features = self.compute_log_gammatone_spectrogram(signal)
        else:
            raise ValueError("feature_type must be either 'mfcc', "
                             "'log_mel_spectrogram', 'log_gammatone_spectrogram' "
                             "or 'spectrogram'")

        original_features = features.copy()

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
            pitches = self.compute_pitch(signal)
            if self.normalize_feature:
                pitches = normalize_audio_feature(
                    pitches, per_feature=self.normalize_per_feature)
            features = np.concatenate([features, np.expand_dims(pitches, axis=-1)], axis=-1)

        return features

    def compute_pitch(self, signal: np.ndarray) -> np.ndarray:
        pitches, _ = librosa.core.piptrack(
            y=signal, sr=self.sample_rate,
            n_fft=self.nfft, hop_length=self.frame_step,
            fmin=0.0, fmax=int(self.sample_rate / 2), win_length=self.frame_length, center=True
        )

        pitches = pitches.T

        assert self.num_feature_bins <= self.frame_length // 2 + 1, \
            "num_features for spectrogram should \
        be <= (sample_rate * window_size // 2 + 1)"

        return pitches[:, :self.num_feature_bins]

    def compute_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        powspec = self.stft(signal)
        features = self.power_to_db(powspec.T)

        assert self.num_feature_bins <= self.frame_length // 2 + 1, \
            "num_features for spectrogram should \
        be <= (sample_rate * window_size // 2 + 1)"

        # cut high frequency part, keep num_feature_bins features
        features = features[:, :self.num_feature_bins]

        return features

    def compute_mfcc(self, signal: np.ndarray) -> np.ndarray:
        S = self.stft(signal)

        mel = librosa.filters.mel(self.sample_rate, self.nfft,
                                  n_mels=self.num_feature_bins,
                                  fmin=0.0, fmax=int(self.sample_rate / 2))

        mel_spectrogram = np.dot(S.T, mel.T)

        mfcc = librosa.feature.mfcc(sr=self.sample_rate,
                                    S=self.power_to_db(mel_spectrogram).T,
                                    n_mfcc=self.num_feature_bins)

        return mfcc.T

    def compute_log_mel_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        S = self.stft(signal)

        mel = librosa.filters.mel(self.sample_rate, self.nfft,
                                  n_mels=self.num_feature_bins,
                                  fmin=0.0, fmax=int(self.sample_rate / 2))

        mel_spectrogram = np.dot(S.T, mel.T)

        return self.power_to_db(mel_spectrogram)

    def compute_log_gammatone_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        S = self.stft(signal)

        gammatone = fft_weights(self.nfft, self.sample_rate,
                                self.num_feature_bins, width=1.0,
                                fmin=0, fmax=int(self.sample_rate / 2),
                                maxlen=(self.nfft / 2 + 1))

        gammatone = gammatone.numpy().astype(np.float32)

        gammatone_spectrogram = np.dot(S.T, gammatone)

        return self.power_to_db(gammatone_spectrogram)


class TFSpeechFeaturizer(SpeechFeaturizer):
    @property
    def shape(self) -> list:
        # None for time dimension
        return [None, self.num_feature_bins, 1]

    def stft(self, signal):
        return tf.square(
            tf.abs(tf.signal.stft(signal, frame_length=self.frame_length,
                                  frame_step=self.frame_step, fft_length=self.nfft)))

    def power_to_db(self, S, ref=1.0, amin=1e-10, top_db=80.0):
        if amin <= 0:
            raise ValueError('amin must be strictly positive')

        magnitude = S

        if six.callable(ref):
            # User supplied a function to calculate reference power
            ref_value = ref(magnitude)
        else:
            ref_value = np.abs(ref)

        log_spec = 10.0 * log10(tf.maximum(amin, magnitude))
        log_spec -= 10.0 * log10(tf.maximum(amin, ref_value))

        if top_db is not None:
            if top_db < 0:
                raise ValueError('top_db must be non-negative')
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

        return log_spec

    def extract(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asfortranarray(signal)
        features = self.tf_extract(tf.convert_to_tensor(signal, dtype=tf.float32))
        return features.numpy()

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

        if self.feature_type == "spectrogram":
            features = self.compute_spectrogram(signal)
        elif self.feature_type == "log_mel_spectrogram":
            features = self.compute_log_mel_spectrogram(signal)
        elif self.feature_type == "mfcc":
            features = self.compute_mfcc(signal)
        elif self.feature_type == "log_gammatone_spectrogram":
            features = self.compute_log_gammatone_spectrogram(signal)
        else:
            raise ValueError("feature_type must be either 'mfcc',"
                             "'log_mel_spectrogram' or 'spectrogram'")

        features = tf.expand_dims(features, axis=-1)

        if self.normalize_feature:
            features = tf_normalize_audio_features(
                features, per_feature=self.normalize_per_feature)

        return features

    def compute_log_mel_spectrogram(self, signal):
        spectrogram = self.stft(signal)
        linear_to_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_feature_bins,
            num_spectrogram_bins=spectrogram.shape[-1],
            sample_rate=self.sample_rate,
            lower_edge_hertz=0.0, upper_edge_hertz=(self.sample_rate / 2)
        )
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_weight_matrix, 1)
        return self.power_to_db(mel_spectrogram)

    def compute_spectrogram(self, signal):
        S = self.stft(signal)
        spectrogram = self.power_to_db(S)
        return spectrogram[:, :self.num_feature_bins]

    def compute_mfcc(self, signal):
        log_mel_spectrogram = self.compute_log_mel_spectrogram(signal)
        return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

    def compute_log_gammatone_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        S = self.stft(signal)

        gammatone = fft_weights(self.nfft, self.sample_rate,
                                self.num_feature_bins, width=1.0,
                                fmin=0, fmax=int(self.sample_rate / 2),
                                maxlen=(self.nfft / 2 + 1))

        gammatone_spectrogram = tf.tensordot(S, gammatone, 1)

        return self.power_to_db(gammatone_spectrogram)
