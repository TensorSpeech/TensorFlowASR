from __future__ import absolute_import

from featurizers.SpeechFeaturizer import SpeechFeaturizer

import sys
import librosa


def main(argv):
    speech_file = argv[1]
    sf = SpeechFeaturizer(sample_rate=16000, frame_ms=20, stride_ms=10, num_feature_bins=128)
    # ft = sf.compute_speech_features(speech_file)
    y, sr = librosa.load(speech_file)
    print(sr)
    print(len(y))
    y = librosa.resample(y, sr, 16000, scale=True)
    print(len(y))
    # print(ft)
    # print(len(ft))
    # print(len(ft[0]))


if __name__ == "__main__":
    main(sys.argv)
