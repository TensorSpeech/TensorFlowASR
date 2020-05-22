from __future__ import absolute_import, print_function

import os.path as o
import sys
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))
from featurizers.SpeechFeaturizer import read_raw_audio, speech_feature_extraction, compute_time_dim
from augmentations.Augments import FreqMasking
import matplotlib.pyplot as plt


def main(argv):
  speech_file = argv[1]
  feature_type = argv[2]
  speech_conf = {
    "sample_rate": 22500,
    "frame_ms": 25,
    "stride_ms": 10,
    "feature_type": feature_type,
    "pre_emph": 0.97,
    "normalize_signal": True,
    "normalize_feature": True,
    "norm_per_feature": False,
    "num_feature_bins": 128,
    "delta": True,
    "delta_delta": True,
    "pitch": True
  }
  signal = read_raw_audio(speech_file, speech_conf["sample_rate"])
  # signal = signal[:speech_conf["sample_rate"]]
  print(len(signal))
  ft = speech_feature_extraction(signal, speech_conf)
  # au = FreqMasking()
  # ft[:, :, 0] = au(ft[:, :, 0])
  print(compute_time_dim(speech_conf, 1))
  print(ft.shape[0])

  ftypes = [feature_type, "delta", "delta_delta", "pitch"]

  plt.figure(figsize=(15, 5))
  for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(ft[:, :, i].T, origin="lower")
    plt.title(ftypes[i])
    plt.colorbar()
    plt.tight_layout()
  plt.savefig(argv[3])
  plt.show()


if __name__ == "__main__":
  main(sys.argv)
