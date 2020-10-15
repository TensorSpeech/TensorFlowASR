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
import glob
import librosa
import nlpaug.augmenter.audio as naa


class SignalCropping(naa.CropAug):
    def __init__(self,
                 zone=(0.2, 0.8),
                 coverage=0.1,
                 crop_range=(0.2, 0.8),
                 crop_factor=2):
        super(SignalCropping, self).__init__(sampling_rate=None, zone=zone, coverage=coverage,
                                             crop_range=crop_range, crop_factor=crop_factor,
                                             duration=None)


class SignalLoudness(naa.LoudnessAug):
    def __init__(self,
                 zone=(0.2, 0.8),
                 coverage=1.,
                 factor=(0.5, 2)):
        super(SignalLoudness, self).__init__(zone=zone, coverage=coverage, factor=factor)


class SignalMask(naa.MaskAug):
    def __init__(self,
                 zone=(0.2, 0.8),
                 coverage=1.,
                 mask_range=(0.2, 0.8),
                 mask_factor=2,
                 mask_with_noise=True):
        super(SignalMask, self).__init__(sampling_rate=None, zone=zone, coverage=coverage,
                                         duration=None, mask_range=mask_range,
                                         mask_factor=mask_factor,
                                         mask_with_noise=mask_with_noise)


class SignalNoise(naa.NoiseAug):
    def __init__(self,
                 sample_rate=16000,
                 zone=(0.2, 0.8),
                 coverage=1.,
                 color="random",
                 noises: str = None):
        if noises is not None:
            noises = glob.glob(os.path.join(noises, "**", "*.wav"), recursive=True)
            noises = [librosa.load(n, sr=sample_rate)[0] for n in noises]
        super(SignalNoise, self).__init__(zone=zone, coverage=coverage,
                                          color=color, noises=noises)


class SignalPitch(naa.PitchAug):
    def __init__(self,
                 zone=(0.2, 0.8),
                 coverage=1.,
                 factor=(-10, 10)):
        super(SignalPitch, self).__init__(None, zone=zone, coverage=coverage,
                                          duration=None, factor=factor)


class SignalShift(naa.ShiftAug):
    def __init__(self,
                 sample_rate=16000,
                 duration=3,
                 direction="random"):
        super(SignalShift, self).__init__(sample_rate, duration=duration, direction=direction)


class SignalSpeed(naa.SpeedAug):
    def __init__(self,
                 zone=(0.2, 0.8),
                 coverage=1.,
                 factor=(0.5, 2)):
        super(SignalSpeed, self).__init__(zone=zone, coverage=coverage,
                                          duration=None, factor=factor)


class SignalVtlp(naa.VtlpAug):
    def __init__(self,
                 sample_rate=16000,
                 zone=(0.2, 0.8),
                 coverage=0.1,
                 fhi=4800,
                 factor=(0.9, 1.1)):
        super(SignalVtlp, self).__init__(sample_rate, zone=zone, coverage=coverage,
                                         duration=None, fhi=fhi, factor=factor)
