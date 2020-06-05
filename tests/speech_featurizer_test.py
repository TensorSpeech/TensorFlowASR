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
# import sys
# sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
#
# import matplotlib.pyplot as plt
# from augmentations.augments import FreqMasking, TimeWarping, WhiteNoise, RealWorldNoise
# from featurizers.speech_featurizers import read_raw_audio, speech_feature_extraction, compute_time_dim
#
#
# def main(argv):
#     speech_file = argv[1]
#     feature_type = argv[2]
#     speech_conf = {
#         "sample_rate": 22500,
#         "frame_ms": 25,
#         "stride_ms": 10,
#         "feature_type": feature_type,
#         "pre_emph": 0.97,
#         "normalize_signal": True,
#         "normalize_feature": True,
#         "norm_per_feature": False,
#         "num_feature_bins": 128,
#         "delta": True,
#         "delta_delta": True,
#         "pitch": True
#     }
#     signal = read_raw_audio(speech_file, speech_conf["sample_rate"])
#     # signal = signal[:speech_conf["sample_rate"]]
#     print(len(signal))
#     au = RealWorldNoise(snr_list=[0], max_noises=3, noise_dir="/mnt/Data/ML/ASR/Preprocessed/Noises")
#     signal = au(signal=signal, sample_rate=16000)
#     ft = speech_feature_extraction(signal, speech_conf)
#     print(compute_time_dim(speech_conf, 1))
#     print(ft.shape[0])
#
#     ftypes = [feature_type, "delta", "delta_delta", "pitch"]
#
#     plt.figure(figsize=(15, 5))
#     for i in range(4):
#         plt.subplot(2, 2, i + 1)
#         plt.imshow(ft[:, :, i].T, origin="lower")
#         plt.title(ftypes[i])
#         plt.colorbar()
#         plt.tight_layout()
#     plt.savefig(argv[3])
#     plt.show()
#
#
# if __name__ == "__main__":
#     main(sys.argv)
