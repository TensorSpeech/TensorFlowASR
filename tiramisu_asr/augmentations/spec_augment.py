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
""" Augmentation on spectrogram: http://arxiv.org/abs/1904.08779 """
import numpy as np

from nlpaug.flow import Sequential
from nlpaug.util import Action
from nlpaug.model.spectrogram import Spectrogram
from nlpaug.augmenter.spectrogram import SpectrogramAugmenter

# ---------------------------- FREQ MASKING ----------------------------


class FreqMaskingModel(Spectrogram):
    def __init__(self, mask_factor: int = 27):
        """
        Args:
            freq_mask_param: parameter F of frequency masking
        """
        super(FreqMaskingModel, self).__init__()
        self.mask_factor = mask_factor

    def mask(self, data: np.ndarray) -> np.ndarray:
        """
        Masking the frequency channels (make features on some channel 0)
        Args:
            spectrogram: shape (T, num_feature_bins, V)
        Returns:
            frequency masked spectrogram
        """
        spectrogram = data.copy()
        freq = np.random.randint(0, self.mask_factor + 1)
        freq = min(freq, spectrogram.shape[1])
        freq0 = np.random.randint(0, spectrogram.shape[1] - freq + 1)
        spectrogram[:, freq0:freq0 + freq, :] = 0  # masking
        return spectrogram


class FreqMaskingAugmenter(SpectrogramAugmenter):
    def __init__(self,
                 mask_factor=27,
                 name="FreqMaskingAugmenter",
                 verbose=0):
        super(FreqMaskingAugmenter, self).__init__(
            action=Action.SUBSTITUTE, zone=(0.2, 0.8), name=name, device="cpu", verbose=verbose,
            coverage=1., factor=(40, 80), silence=False, stateless=True)
        self.model = FreqMaskingModel(mask_factor)

    def substitute(self, data):
        return self.model.mask(data)


class FreqMasking(SpectrogramAugmenter):
    def __init__(self,
                 num_masks=1,
                 mask_factor=27,
                 name="FreqMasking",
                 verbose=0):
        super(FreqMasking, self).__init__(
            action=Action.SUBSTITUTE, zone=(0.2, 0.8), name=name, device="cpu", verbose=verbose,
            coverage=1., factor=(40, 80), silence=False, stateless=True)
        self.flow = Sequential([FreqMaskingAugmenter(mask_factor) for _ in range(num_masks)])

    def substitute(self, data):
        return self.flow.augment(data)

# ---------------------------- TIME MASKING ----------------------------


class TimeMaskingModel(Spectrogram):
    def __init__(self, mask_factor: int = 100, p_upperbound: float = 1.0):
        """
        Args:
            time_mask_param: parameter W of time masking
            p_upperbound: an upperbound so that the number of masked time
                steps must not exceed p_upperbound * total_time_steps
        """
        super(TimeMaskingModel, self).__init__()
        self.mask_factor = mask_factor
        self.p_upperbound = p_upperbound
        assert 0.0 <= self.p_upperbound <= 1.0, "0.0 <= p_upperbound <= 1.0"

    def mask(self, data: np.ndarray) -> np.ndarray:
        """
        Masking the time steps (make features on some time steps 0)
        Args:
            spectrogram: shape (T, num_feature_bins, V)
        Returns:
            a tensor that's applied time masking
        """
        spectrogram = data.copy()
        time = np.random.randint(0, self.mask_factor + 1)
        time = min(time, spectrogram.shape[0])
        time0 = np.random.randint(0, spectrogram.shape[0] - time + 1)
        time = min(time, int(self.p_upperbound * spectrogram.shape[0]))
        spectrogram[time0:time0 + time, :, :] = 0
        return spectrogram


class TimeMaskingAugmenter(SpectrogramAugmenter):
    def __init__(self,
                 mask_factor=100,
                 p_upperbound=1,
                 name="TimeMaskingAugmenter",
                 verbose=0):
        super(TimeMaskingAugmenter, self).__init__(
            action=Action.SUBSTITUTE, zone=(0.2, 0.8), name=name, device="cpu", verbose=verbose,
            coverage=1., silence=False, stateless=True)
        self.model = TimeMaskingModel(mask_factor, p_upperbound)

    def substitute(self, data):
        return self.model.mask(data)


class TimeMasking(SpectrogramAugmenter):
    def __init__(self,
                 num_masks=1,
                 mask_factor=100,
                 p_upperbound=1,
                 name="TimeMasking",
                 verbose=0):
        super(TimeMasking, self).__init__(
            action=Action.SUBSTITUTE, zone=(0.2, 0.8), name=name, device="cpu", verbose=verbose,
            coverage=1., silence=False, stateless=True)
        self.flow = Sequential([
            TimeMaskingAugmenter(mask_factor, p_upperbound) for _ in range(num_masks)
        ])

    def substitute(self, data):
        return self.flow.augment(data)
