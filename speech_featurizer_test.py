from __future__ import absolute_import

from featurizers.SpeechFeaturizer import SpeechFeaturizer

import sys
import io
import librosa


def main(argv):
    speech_file = argv[1]
    sf = SpeechFeaturizer(sample_rate=16000, frame_ms=20, stride_ms=10, num_feature_bins=128)
    ft = sf.tf_read_raw_audio(speech_file)
    print(ft, len(ft))
    with open(speech_file, "rb") as f:
        b = f.read()
    ft1 = sf.tf_read_raw_audio(b)
    print(ft1)
    ft2 = sf.convert_bytesarray_to_float(b, channels=1)
    print(ft2, len(ft2))
    print(librosa.load(io.BytesIO(b), sr=16000))


if __name__ == "__main__":
    main(sys.argv)
