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

import io
import math
import os
from typing import Union

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_io as tfio

from tensorflow_asr.configs.config import SpeechConfig
from tensorflow_asr.featurizers.methods import gammatone
from tensorflow_asr.utils import env_util, math_util


def load_and_convert_to_wav(
    path: str,
) -> tf.Tensor:
    wave, rate = librosa.load(os.path.expanduser(path), sr=None, mono=True)
    return tf.audio.encode_wav(tf.expand_dims(wave, axis=-1), sample_rate=rate)


def read_raw_audio(
    audio: Union[str, bytes, np.ndarray],
    sample_rate=16000,
) -> np.ndarray:
    if isinstance(audio, str):
        wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate, mono=True)
    elif isinstance(audio, bytes):
        wave, sr = sf.read(io.BytesIO(audio))
        if wave.ndim > 1:
            wave = np.mean(wave, axis=-1)
        wave = np.asfortranarray(wave)
        if sr != sample_rate:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=sample_rate)
    elif isinstance(audio, np.ndarray):
        if audio.ndim > 1:
            ValueError("input audio must be single channel")
        return audio
    else:
        raise ValueError("input audio must be either a path or bytes")
    return wave


def tf_read_raw_audio(
    audio: tf.Tensor,
    sample_rate=16000,
) -> tf.Tensor:
    wave, rate = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=-1)
    if not env_util.has_devices("TPU"):
        resampled = tfio.audio.resample(wave, rate_in=tf.cast(rate, dtype=tf.int64), rate_out=sample_rate)
        return tf.reshape(resampled, shape=[-1])  # reshape for using tf.signal
    return tf.reshape(wave, shape=[-1])  # reshape for using tf.signal


def slice_signal(
    signal,
    window_size,
    stride=0.5,
) -> np.ndarray:
    """Return windows of the given signal by sweeping in stride fractions of window"""
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    offset = int(window_size * stride)
    slices = []
    for beg_i, end_i in zip(range(0, n_samples, offset), range(window_size, n_samples + offset, offset)):
        slice_ = signal[beg_i:end_i]
        if slice_.shape[0] < window_size:
            slice_ = np.pad(slice_, (0, window_size - slice_.shape[0]), "constant", constant_values=0.0)
        if slice_.shape[0] == window_size:
            slices.append(slice_)
    return np.array(slices, dtype=np.float32)


def tf_merge_slices(
    slices: tf.Tensor,
) -> tf.Tensor:
    # slices shape = [batch, window_size]
    return tf.keras.backend.flatten(slices)  # return shape = [-1, ]


def tf_normalize_audio_features(
    audio_feature: tf.Tensor,
    per_frame=False,
) -> tf.Tensor:
    """
    TF z-score features normalization
    Args:
        audio_feature: tf.Tensor with shape [T, F]
        per_frame:

    Returns:
        normalized audio features with shape [T, F]
    """
    axis = -1 if per_frame else 0
    mean = tf.reduce_mean(audio_feature, axis=axis, keepdims=True)
    stddev = tf.sqrt(tf.math.reduce_variance(audio_feature, axis=axis, keepdims=True) + 1e-9)
    return tf.divide(tf.subtract(audio_feature, mean), stddev)


def tf_normalize_signal(
    signal: tf.Tensor,
) -> tf.Tensor:
    """
    TF Normailize signal to [-1, 1] range
    Args:
        signal: tf.Tensor with shape [None]

    Returns:
        normalized signal with shape [None]
    """
    gain = 1.0 / (tf.reduce_max(tf.abs(signal), axis=-1) + 1e-9)
    return signal * gain


def tf_preemphasis(
    signal: tf.Tensor,
    coeff=0.97,
):
    """
    TF Pre-emphasis
    Args:
        signal: tf.Tensor with shape [None]
        coeff: Float that indicates the preemphasis coefficient

    Returns:
        pre-emphasized signal with shape [None]
    """
    if not coeff or coeff <= 0.0:
        return signal
    s0 = tf.expand_dims(signal[0], axis=-1)
    s1 = signal[1:] - coeff * signal[:-1]
    return tf.concat([s0, s1], -1)


def tf_depreemphasis(
    signal: tf.Tensor,
    coeff=0.97,
) -> tf.Tensor:
    """
    TF Depreemphasis
    Args:
        signal: tf.Tensor with shape [B, None]
        coeff: Float that indicates the preemphasis coefficient

    Returns:
        depre-emphasized signal with shape [B, None]
    """
    if not coeff or coeff <= 0.0:
        return signal

    def map_fn(elem):
        x = tf.expand_dims(elem[0], axis=-1)
        for n in range(1, elem.shape[0], 1):
            current = coeff * x[n - 1] + elem[n]
            x = tf.concat([x, [current]], 0)
        return x

    return tf.map_fn(map_fn, signal)


class SpeechFeaturizer:
    def __init__(self, speech_config: SpeechConfig):
        self.speech_config = speech_config
        self.max_length = 0

    @property
    def nfft(self) -> int:
        """Number of FFT"""
        fft_length = int(max(512, math.pow(2, math.ceil(math.log(self.speech_config.frame_length, 2)))))
        if self.speech_config.fft_overdrive:
            fft_length *= 2
        return fft_length

    @property
    def shape(self) -> list:
        length = self.max_length if self.max_length > 0 else None
        return [length, self.speech_config.num_feature_bins, 1]

    def get_length_from_duration(self, duration):
        nsamples = math.ceil(float(duration) * self.speech_config.sample_rate)
        # https://www.tensorflow.org/api_docs/python/tf/signal/frame
        if self.speech_config.use_librosa_like_stft:
            return 1 + (nsamples - self.nfft) // self.speech_config.frame_step
        if self.speech_config.pad_end:
            return -(-nsamples // self.speech_config.frame_step)
        return 1 + (nsamples - self.speech_config.frame_length) // self.speech_config.frame_step

    def update_length(self, length: int):
        self.max_length = max(self.max_length, length)

    def reset_length(self):
        self.max_length = 0

    def stft(self, signal):
        if self.speech_config.use_librosa_like_stft:
            # signal = tf.pad(signal, [[self.nfft // 2, self.nfft // 2]], mode="REFLECT")
            window = tf.signal.hann_window(self.speech_config.frame_length, periodic=True)
            left_pad = (self.nfft - self.speech_config.frame_length) // 2
            right_pad = self.nfft - self.speech_config.frame_length - left_pad
            window = tf.pad(window, [[left_pad, right_pad]])
            framed_signals = tf.signal.frame(signal, frame_length=self.nfft, frame_step=self.speech_config.frame_step)
            framed_signals *= window
            fft_features = tf.abs(tf.signal.rfft(framed_signals, [self.nfft]))
        else:
            fft_features = tf.abs(
                tf.signal.stft(
                    signal,
                    frame_length=self.speech_config.frame_length,
                    frame_step=self.speech_config.frame_step,
                    fft_length=self.nfft,
                    pad_end=self.speech_config.pad_end,
                )
            )
        if self.speech_config.compute_energy:
            fft_features = tf.square(fft_features)
        return fft_features

    def logarithm(self, S):
        if self.speech_config.use_natural_log:
            return tf.math.log(tf.maximum(float(self.speech_config.output_floor), S))
        log_spec = 10.0 * math_util.log10(tf.maximum(self.speech_config.output_floor, S))
        log_spec -= 10.0 * math_util.log10(tf.maximum(self.speech_config.output_floor, 1.0))
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
        if self.speech_config.normalize_signal:
            signal = tf_normalize_signal(signal)
        signal = tf_preemphasis(signal, self.speech_config.preemphasis)

        if self.speech_config.feature_type == "spectrogram":
            features = self.compute_spectrogram(signal)
        elif self.speech_config.feature_type == "log_mel_spectrogram":
            features = self.compute_log_mel_spectrogram(signal)
        elif self.speech_config.feature_type == "mfcc":
            features = self.compute_mfcc(signal)
        elif self.speech_config.feature_type == "log_gammatone_spectrogram":
            features = self.compute_log_gammatone_spectrogram(signal)
        else:
            raise ValueError("feature_type must be either 'mfcc', 'log_mel_spectrogram' or 'spectrogram'")

        if self.speech_config.normalize_feature:
            features = tf_normalize_audio_features(features, per_frame=self.speech_config.normalize_per_frame)

        features = tf.expand_dims(features, axis=-1)
        return features

    def compute_log_mel_spectrogram(self, signal):
        spectrogram = self.stft(signal)
        linear_to_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.speech_config.num_feature_bins,
            num_spectrogram_bins=spectrogram.shape[-1],
            sample_rate=self.speech_config.sample_rate,
            lower_edge_hertz=self.speech_config.lower_edge_hertz,
            upper_edge_hertz=self.speech_config.upper_edge_hertz,
        )
        mel_spectrogram = tf.matmul(spectrogram, linear_to_weight_matrix)
        return self.logarithm(mel_spectrogram)

    def compute_spectrogram(self, signal):
        S = self.stft(signal)
        spectrogram = self.logarithm(S)
        return spectrogram[:, : self.speech_config.num_feature_bins]

    def compute_mfcc(self, signal):
        log_mel_spectrogram = self.compute_log_mel_spectrogram(signal)
        return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

    def compute_log_gammatone_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        S = self.stft(signal)
        gtone = gammatone.fft_weights(
            self.nfft,
            self.speech_config.sample_rate,
            self.speech_config.num_feature_bins,
            width=1.0,
            fmin=int(self.speech_config.lower_edge_hertz),
            fmax=int(self.speech_config.upper_edge_hertz),
            maxlen=(self.nfft / 2 + 1),
        )
        gtone_spectrogram = tf.matmul(S, gtone)
        return self.logarithm(gtone_spectrogram)
