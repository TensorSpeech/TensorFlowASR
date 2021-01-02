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
import sys
from tensorflow_asr.utils import setup_environment
setup_environment()
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio, TFSpeechFeaturizer, NumpySpeechFeaturizer


def main(argv):
    speech_file = argv[1]
    feature_type = argv[2]
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

    nsf = NumpySpeechFeaturizer(speech_conf)
    sf = TFSpeechFeaturizer(speech_conf)
    ft = nsf.stft(signal)
    print(ft.shape, np.mean(ft))
    ft = sf.stft(signal).numpy()
    print(ft.shape, np.mean(ft))
    ft = sf.extract(signal)

    plt.figure(figsize=(16, 2.5))
    ax = plt.gca()
    ax.set_title(f"{feature_type}", fontweight="bold")
    librosa.display.specshow(ft.T, cmap="magma")
    v1 = np.linspace(ft.min(), ft.max(), 8, endpoint=True)
    plt.colorbar(pad=0.01, fraction=0.02, ax=ax, format="%.2f", ticks=v1)
    plt.tight_layout()
    # plt.savefig(argv[3])
    # plt.show()
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
