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

from dataclasses import asdict, dataclass

import tensorflow as tf

from tensorflow_asr.augmentations.augmentation import Augmentation
from tensorflow_asr.features import gammatone
from tensorflow_asr.models.base_layer import Layer
from tensorflow_asr.utils import math_util


@dataclass
class FEATURE_TYPES:
    SPECTROGRAM: str = "spectrogram"
    LOG_MEL_SPECTROGRAM: str = "log_mel_spectrogram"
    MFCC: str = "mfcc"
    LOG_GAMMATONE_SPECTROGRAM: str = "log_gammatone_spectrogram"


class FeatureExtraction(Layer):
    def __init__(
        self,
        sample_rate=16000,
        frame_ms=25,
        stride_ms=10,
        num_feature_bins=80,
        feature_type="log_mel_spectrogram",
        preemphasis=0.97,
        pad_end=True,
        use_librosa_like_stft=False,
        output_floor=1e-10,
        lower_edge_hertz=0.0,
        upper_edge_hertz=8000.0,
        log_base="e",  # "10", "e"
        nfft=None,
        normalize_signal=False,
        normalize_zscore=False,
        normalize_min_max=False,
        padding=0,
        augmentation_config=None,
        **kwargs,
    ):
        """
        Audio Features Extraction Keras Layer

        Parameters
        ----------
        sample_rate : int, optional
            Sample rate of audio signals in Hz, by default 16000
        frame_ms : int, optional
            Amount of data grabbed for each frame during analysis in ms, by default 25
        stride_ms : int, optional
            Number of ms to jump between frames, by default 10
        num_feature_bins : int, optional
            Number of bins in the feature output, by default 80
        feature_type : str, optional
            Type of feature extraction, by default "log_mel_spectrogram"
        preemphasis : float, optional
            The first-order filter coefficient used for preemphasis, when it is 0.0, preemphasis is turned off, by default 0.0
        pad_end : bool, optional
            Whether to pad the end of `signals` with zeros when framing produces a frame that lies partially past its end, by default True
        use_librosa_like_stft : bool, optional
            Use librosa like stft, by default False
        output_floor : _type_, optional
            Minimum output value, by default 1e-10
        lower_edge_hertz : float, optional
            The lowest frequency of the feature analysis, by default 125.0
        upper_edge_hertz : float, optional
            The highest frequency of the feature analysis, by default 8000.0
        log_base : str, optional
            The base of logarithm, by default 'e'
        nfft : int, optional
            NFFT, if None, equals frame_length derived from frame_ms, by default None
        normalize_signal : bool, optional
            Normalize signals to [-1,1] range, by default False
        normalize_zscore : bool, optional
            Normalize features using z-score, by default False
        normalize_min_max : bool, optional
            Normalize features as (value - min) / (max - min), by default True
        padding : int, optional
            Number of samples to pad with 0 before feature extraction, by default 0
        augmentation_config : dict, optional
            Dictionary of augmentation config for training
        """
        assert feature_type in asdict(FEATURE_TYPES()).values(), f"feature_type must be in {asdict(FEATURE_TYPES()).values()}"

        super().__init__(name=feature_type, trainable=False, **kwargs)
        self.sample_rate = sample_rate

        self.frame_ms = frame_ms
        self.frame_length = int(round(self.sample_rate * self.frame_ms / 1000.0))

        self.stride_ms = stride_ms
        self.frame_step = int(round(self.sample_rate * self.stride_ms / 1000.0))

        self.num_feature_bins = num_feature_bins

        self.feature_type = feature_type

        self.preemphasis = preemphasis

        self.pad_end = pad_end

        self.use_librosa_like_stft = use_librosa_like_stft

        self.output_floor = output_floor

        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz

        self.log_base = log_base
        assert self.log_base in ("10", "e"), "log_base must be '10' or 'e'"

        self._normalize_signal = normalize_signal
        self._normalize_zscore = normalize_zscore
        self._normalize_min_max = normalize_min_max

        self.padding = padding
        self.nfft = self.frame_length if nfft is None else nfft

        self.augmentations = Augmentation(augmentation_config)

    # ---------------------------------- signals --------------------------------- #

    def normalize_signal(self, signal):
        if not self._normalize_signal:
            return signal
        gain = 1.0 / (tf.reduce_max(tf.abs(signal), axis=-1) + 1e-9)
        return signal * gain

    def preemphasis_signal(self, signal):
        if not self.preemphasis or self.preemphasis <= 0.0:
            return signal
        s0 = tf.expand_dims(signal[:, 0], axis=-1)
        s1 = signal[:, 1:] - self.preemphasis * signal[:, :-1]
        return tf.concat([s0, s1], -1)

    # --------------------------------- features --------------------------------- #

    def normalize_audio_features(self, audio_feature):
        if self._normalize_zscore:
            mean = tf.reduce_mean(audio_feature, axis=1, keepdims=True)
            stddev = tf.sqrt(tf.math.reduce_variance(audio_feature, axis=1, keepdims=True) + 1e-9)
            return tf.divide(tf.subtract(audio_feature, mean), stddev)
        if self._normalize_min_max:
            if self.feature_type.startswith("log_") or self.feature_type == FEATURE_TYPES.SPECTROGRAM:
                min_value = self.logarithm(self.output_floor)
            else:
                min_value = tf.reduce_min(audio_feature, axis=1, keepdims=True)
            return (audio_feature - min_value) / (tf.reduce_max(audio_feature, axis=1, keepdims=True) - min_value)
        return audio_feature

    def stft(self, signal):
        signal = tf.cast(signal, tf.float32)
        if self.use_librosa_like_stft:
            # signal = tf.pad(signal, [[self.nfft // 2, self.nfft // 2]], mode="REFLECT")
            window = tf.signal.hann_window(self.frame_length, periodic=True)
            left_pad = (self.nfft - self.frame_length) // 2
            right_pad = self.nfft - self.frame_length - left_pad
            window = tf.pad(window, [[left_pad, right_pad]])
            framed_signals = tf.signal.frame(signal, frame_length=self.nfft, frame_step=self.frame_step, pad_end=self.pad_end)
            framed_signals *= window
            fft_features = tf.abs(tf.signal.rfft(framed_signals, [self.nfft]))
        else:
            fft_features = tf.abs(
                tf.signal.stft(signal, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.nfft, pad_end=self.pad_end)
            )
        fft_features = tf.square(fft_features)
        fft_features = tf.cast(fft_features, self.dtype)
        return fft_features

    def logarithm(self, S):
        if self.log_base == "10":
            return math_util.log10(tf.maximum(S, self.output_floor))
        return tf.math.log(tf.maximum(S, self.output_floor))

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
        return spectrogram[:, :, : self.num_feature_bins]

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
        """
        Compute features of audio signals

        Parameters
        ----------
        inputs : tf.Tensor, shape [B, None]
            Audio signals that were resampled to sample_rate

        training : bool, optional
            Training mode, by default False

        Returns
        -------
        tf.Tensor, shape = [B, n_frames, num_feature_bins, 1] if has_channel_dim else [B, n_frames, num_feature_bins]
            Features extracted from audio signals
        """
        signals, signals_length = inputs

        if training:
            signals, signals_length = self.augmentations.signal_augment(signals, signals_length)

        if self.padding > 0:
            signals = tf.pad(signals, [[0, 0], [0, self.padding]], mode="CONSTANT", constant_values=0)

        signals = self.normalize_signal(signals)
        signals = self.preemphasis_signal(signals)

        if self.feature_type == FEATURE_TYPES.SPECTROGRAM:
            features = self.spectrogram(signals)
        elif self.feature_type == FEATURE_TYPES.MFCC:
            features = self.mfcc(signals)  # TODO: add option to compute delta features for mfccs
        elif self.feature_type == FEATURE_TYPES.LOG_GAMMATONE_SPECTROGRAM:
            features = self.log_gammatone_spectrogram(signals)
        else:  # default as log_mel_spectrogram
            features = self.log_mel_spectrogram(signals)

        features = self.normalize_audio_features(features)
        features = tf.expand_dims(features, axis=-1)
        features_length = tf.map_fn(fn=self.get_nframes, elems=signals_length, fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.int32))

        if training:
            features, features_length = self.augmentations.feature_augment(features, features_length)

        return features, features_length

    def get_nframes(self, nsamples):
        # https://www.tensorflow.org/api_docs/python/tf/signal/frame
        if self.use_librosa_like_stft:
            if self.pad_end:
                return -(-nsamples // self.frame_step)
            return 1 + (nsamples - self.nfft) // self.frame_step
        if self.pad_end:
            return -(-nsamples // self.frame_step)
        return 1 + (nsamples - self.frame_length) // self.frame_step

    def compute_mask(self, inputs, mask=None):
        signals, signals_length = inputs
        mask = tf.sequence_mask(signals_length, maxlen=(tf.shape(signals)[1] + self.padding), dtype=tf.bool)
        nsamples = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        nframes = tf.map_fn(fn=self.get_nframes, elems=nsamples, fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.int32))
        padded_nframes = self.get_nframes(tf.shape(signals, tf.int32)[1])
        return tf.sequence_mask(nframes, maxlen=padded_nframes, dtype=tf.bool), None

    def compute_output_shape(self, input_shape):
        signal_shape, signal_length_shape = input_shape
        B, nsamples = signal_shape
        if nsamples is None:
            output_shape = [B, None, self.num_feature_bins, 1]
        else:
            output_shape = [B, self.get_nframes(nsamples + self.padding), self.num_feature_bins, 1]
        return tf.TensorShape(output_shape), tf.TensorShape(signal_length_shape)
