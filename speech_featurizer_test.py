from __future__ import absolute_import

from featurizers.SpeechFeaturizer import read_raw_audio, speech_feature_extraction

import sys
import matplotlib.pyplot as plt


def main(argv):
  speech_file = argv[1]
  feature_type = argv[2]
  speech_conf = {
    "sample_rate":       16000,
    "frame_ms":          25,
    "stride_ms":         10,
    "feature_type":      feature_type,
    "pre_emph":          0.97,
    "normalize_signal":  True,
    "normalize_feature": True,
    "num_feature_bins":  12,
    "is_delta":          True,
    "pitch":             40
  }
  signal = read_raw_audio(speech_file, speech_conf["sample_rate"])
  ft = speech_feature_extraction(signal, speech_conf).T

  plt.figure(figsize=(15, 5))
  plt.plot(1, 1, 1)
  plt.imshow(ft, origin="lower")
  plt.title(feature_type)
  plt.colorbar()
  plt.tight_layout()
  plt.savefig(argv[3])
  plt.show()


if __name__ == "__main__":
  main(sys.argv)
