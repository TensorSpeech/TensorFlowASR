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

# from __future__ import absolute_import, print_function
#
# import os.path as o
import sys
from tiramisu_asr.utils import setup_environment
setup_environment()
import librosa
# sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
#
import matplotlib.pyplot as plt
from tiramisu_asr.featurizers.speech_featurizers import read_raw_audio, \
    TFSpeechFeaturizer, NumpySpeechFeaturizer
from tiramisu_asr.augmentations.augments import UserAugmentation


def main(argv):
    speech_file = argv[1]
    feature_type = argv[2]
    augments = {
        # "after": {
        #     "time_masking": {
        #         "num_masks": 10,
        #         "mask_factor": 100,
        #         "p_upperbound": 0.05
        #     },
        #     "freq_masking": {
        #         "mask_factor": 27
        #     }
        # },
    }
    au = UserAugmentation(augments)
    speech_conf = {
        "sample_rate": 16000,
        "frame_ms": 25,
        "stride_ms": 10,
        "feature_type": feature_type,
        "preemphasis": 0.97,
        "normalize_signal": True,
        "normalize_feature": True,
        "normalize_per_feature": False,
        "num_feature_bins": 80,
    }
    signal = read_raw_audio(speech_file, speech_conf["sample_rate"])

    sf = NumpySpeechFeaturizer(speech_conf)
    ft = sf.extract(signal)
    ft = au["after"].augment(ft)[:, :, 0]

    plt.figure(figsize=(20, 3))
    librosa.display.specshow(ft.T, cmap="magma")
    plt.tight_layout()
    plt.colorbar()
    plt.show()
    # plt.figure(figsize=(15, 5))
    # for i in range(4):
    #     plt.subplot(2, 2, i + 1)
    #     plt.imshow(ft[:, :, i].T, origin="lower")
    #     plt.title(ftypes[i])
    #     plt.colorbar()
    #     plt.tight_layout()
    # plt.savefig(argv[3])
    # plt.show()


if __name__ == "__main__":
    main(sys.argv)
