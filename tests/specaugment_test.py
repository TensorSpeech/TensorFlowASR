from __future__ import absolute_import

import sys

import tensorflow as tf
from featurizers.SpeechFeaturizer import SpeechFeaturizer
from augmentations.Augments import TimeWarping, TimeMasking, FreqMasking
import matplotlib.pyplot as plt


def main(argv):
    fm = FreqMasking(num_freq_mask=2)
    tm = TimeMasking()
    tw = TimeWarping()

    speech_file = argv[1]
    sf = SpeechFeaturizer(sample_rate=16000, frame_ms=20, stride_ms=10, num_feature_bins=128)
    ft = sf.compute_speech_features(speech_file)

    plt.figure(figsize=(15, 5))

    plt.subplot(2, 1, 1)
    plt.imshow(tf.transpose(tf.squeeze(ft)))

    ft = fm(ft)

    print(ft)

    ft = tf.squeeze(ft)
    ft = tf.transpose(ft)

    plt.subplot(2, 1, 2)
    plt.imshow(ft)
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
