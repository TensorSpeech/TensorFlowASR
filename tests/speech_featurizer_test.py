from __future__ import absolute_import, print_function

import os.path as o
import sys
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from featurizers.SpeechFeaturizer import read_raw_audio, speech_feature_extraction


def main(argv):
  speech_file = argv[1]
  feature_type = argv[2]
  speech_conf = {
    "sample_rate": 16384,
    "frame_ms": 20,
    "stride_ms": 10,
    "feature_type": feature_type,
    "pre_emph": 0.97,
    "normalize_signal": True,
    "normalize_feature": True,
    "norm_per_feature": False,
    "num_feature_bins": 12,
    "delta": False,
    "delta_delta": False,
    "pitch": False
  }
  signal = read_raw_audio(speech_file, speech_conf["sample_rate"])
  ft = speech_feature_extraction(signal, speech_conf)
  f, c = np.shape(ft)[1:]
  ft = np.reshape(ft, [-1, f*c])

  plt.figure(figsize=(15, 5))
  plt.plot(1, 1, 1)
  plt.imshow(ft.T, origin="lower")
  plt.title(feature_type)
  plt.colorbar()
  plt.tight_layout()
  plt.savefig(argv[3])
  plt.show()


if __name__ == "__main__":
  main(sys.argv)
