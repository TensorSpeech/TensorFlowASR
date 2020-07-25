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

# from __future__ import absolute_import
# import matplotlib.pyplot as plt
# from augmentations.augments import TimeWarping, TimeMasking, FreqMasking
# from featurizers.speech_featurizers import SpeechFeaturizer
# import tensorflow as tf
#
# import sys
# import os.path as o
# sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
#
#
# def main(argv):
#     fm = FreqMasking(num_freq_mask=2)
#     tm = TimeMasking()
#     tw = TimeWarping()
#
#     speech_file = argv[1]
#     sf = SpeechFeaturizer(sample_rate=16000, frame_ms=20, stride_ms=10, num_feature_bins=128)
#     ft = sf.compute_speech_features(speech_file)
#
#     plt.figure(figsize=(15, 5))
#
#     plt.subplot(2, 1, 1)
#     plt.imshow(tf.transpose(tf.squeeze(ft)))
#
#     ft = fm(ft)
#
#     print(ft)
#
#     ft = tf.squeeze(ft)
#     ft = tf.transpose(ft)
#
#     plt.subplot(2, 1, 2)
#     plt.imshow(ft)
#     plt.show()
#
#
# if __name__ == "__main__":
#     main(sys.argv)
