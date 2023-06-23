# Copyright 2023 Huy Le Nguyen (@nglehuy)
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

import math

import tensorflow as tf

from tensorflow_asr.featurizers.methods import gammatone
from tensorflow_asr.models.base_layer import Layer
from tensorflow_asr.utils import math_util


class FeatureExtraction(Layer):
    def __init__(
        self,
        sample_rate=16000,
        frame_ms=25,
        stride_ms=10,
        num_feature_bins=80,
        feature_type="log_mel_spectrogram",
        preemphasis=0.0,
        pad_end=True,
        use_librosa_like_stft=False,
        fft_overdrive=True,
        output_floor=1e-10,
        lower_edge_hertz=125.0,
        upper_edge_hertz=8000.0,
        normalize_signal=False,
        normalize_feature=False,
        normalize_per_frame=False,
        name="feature_extraction",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        # Sample rate in Hz
        self.sample_rate = sample_rate
        # Amount of data grabbed for each frame during analysis
        self.frame_ms = frame_ms
        self.frame_length = int(round(self.sample_rate * self.frame_ms / 1000.0))
        # Number of ms to jump between frames
        self.stride_ms = stride_ms
        self.frame_step = int(round(self.sample_rate * self.stride_ms / 1000.0))
        # Number of bins in the feature output
        self.num_feature_bins = num_feature_bins
        # Type of feature extraction
        self.feature_type = feature_type
        # The first-order filter coefficient used for preemphasis. When it is 0.0, preemphasis is turned off.
        self.preemphasis = preemphasis
        # Whether to pad the end of `signals` with zeros when framing produces a frame that lies partially past its end.
        self.pad_end = pad_end
        # Use librosa like stft
        self.use_librosa_like_stft = use_librosa_like_stft
        # Whether to use twice the minimum fft resolution.
        self.fft_overdrive = fft_overdrive
        # Minimum output of filterbank output prior to taking logarithm.
        self.output_floor = output_floor
        # The lowest frequency of the feature analysis
        self.lower_edge_hertz = lower_edge_hertz
        # The highest frequency of the feature analysis
        self.upper_edge_hertz = upper_edge_hertz
        # Normalization
        self._normalize_signal = normalize_signal
        self._normalize_feature = normalize_feature
        self._normalize_per_frame = normalize_per_frame
        # NFFT
        self.nfft = int(max(512, math.pow(2, math.ceil(math.log(self.frame_length, 2)))))
        if self.fft_overdrive:
            self.nfft *= 2

    # ---------------------------------- signals --------------------------------- #

    def normalize_signal(self, signal):
        """
        TF Normailize signal to [-1, 1] range
        Args:
            signal: tf.Tensor with shape [B, None]

        Returns:
            normalized signal with shape [B, None]
        """
        if not self._normalize_signal:
            return signal
        gain = 1.0 / (tf.reduce_max(tf.abs(signal), axis=-1) + 1e-9)
        return signal * gain

    def preemphasis_signal(self, signal):
        """
        TF Pre-emphasis
        Args:
            signal: tf.Tensor with shape [B, None]
            coeff: Float that indicates the preemphasis coefficient

        Returns:
            pre-emphasized signal with shape [B, None]
        """
        if not self.preemphasis or self.preemphasis <= 0.0:
            return signal
        s0 = tf.expand_dims(signal[:, 0], axis=-1)
        s1 = signal[:, 1:] - self.preemphasis * signal[:, :-1]
        return tf.concat([s0, s1], -1)

    # --------------------------------- features --------------------------------- #

    def normalize_audio_features(self, audio_feature):
        """
        TF z-score features normalization
        Args:
            audio_feature: tf.Tensor with shape [B, T, F]
            per_frame:

        Returns:
            normalized audio features with shape [B, T, F]
        """
        if not self._normalize_feature:
            return audio_feature
        axis = -1 if self._normalize_per_frame else 1
        mean = tf.reduce_mean(audio_feature, axis=axis, keepdims=True)
        stddev = tf.sqrt(tf.math.reduce_variance(audio_feature, axis=axis, keepdims=True) + 1e-9)
        return tf.divide(tf.subtract(audio_feature, mean), stddev)

    def stft(self, signal):
        if self.use_librosa_like_stft:
            # signal = tf.pad(signal, [[self.nfft // 2, self.nfft // 2]], mode="REFLECT")
            window = tf.signal.hann_window(self.frame_length, periodic=True)
            left_pad = (self.nfft - self.frame_length) // 2
            right_pad = self.nfft - self.frame_length - left_pad
            window = tf.pad(window, [[left_pad, right_pad]])
            framed_signals = tf.signal.frame(signal, frame_length=self.nfft, frame_step=self.frame_step)
            framed_signals *= window
            fft_features = tf.abs(tf.signal.rfft(framed_signals, [self.nfft]))
        else:
            fft_features = tf.abs(
                tf.signal.stft(signal, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.nfft, pad_end=self.pad_end)
            )
        fft_features = tf.square(fft_features)
        return fft_features

    def logarithm(self, S):
        log_spec = 10.0 * math_util.log10(tf.maximum(self.output_floor, S))
        log_spec -= 10.0 * math_util.log10(tf.maximum(self.output_floor, 1.0))
        return log_spec

    def log_mel_spectrogram(self, signal):
        S = self.stft(signal)
        linear_to_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_feature_bins,
            num_spectrogram_bins=S.shape[-1],
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.lower_edge_hertz,
            upper_edge_hertz=self.upper_edge_hertz,
        )
        mel_spectrogram = tf.matmul(S, linear_to_weight_matrix)
        return self.logarithm(mel_spectrogram)

    def spectrogram(self, signal):
        spectrogram = self.logarithm(self.stft(signal))
        return spectrogram[:, : self.num_feature_bins]

    def mfcc(self, signal):
        log_mel_spectrogram = self.log_mel_spectrogram(signal)
        return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

    def log_gammatone_spectrogram(self, signal):
        S = self.stft(signal)
        gtone = gammatone.fft_weights(
            self.nfft,
            self.sample_rate,
            self.num_feature_bins,
            width=1.0,
            fmin=int(self.lower_edge_hertz),
            fmax=int(self.upper_edge_hertz),
            maxlen=(self.nfft / 2 + 1),
        )
        gtone_spectrogram = tf.matmul(S, gtone)
        return self.logarithm(gtone_spectrogram)

    def call(self, inputs, training=False):
        signal = self.normalize_signal(inputs)
        signal = self.preemphasis_signal(signal)

        if self.feature_type == "spectrogram":
            features = self.spectrogram(signal)
        elif self.feature_type == "log_mel_spectrogram":
            features = self.log_mel_spectrogram(signal)
        elif self.feature_type == "mfcc":
            features = self.mfcc(signal)
        elif self.feature_type == "log_gammatone_spectrogram":
            features = self.log_gammatone_spectrogram(signal)
        else:
            raise ValueError("feature_type must be either 'mfcc', 'log_mel_spectrogram' or 'spectrogram'")

        features = self.normalize_audio_features(features)
        features = tf.expand_dims(features, axis=-1)
        return features

    def compute_output_shape(self, input_shape):
        return input_shape
