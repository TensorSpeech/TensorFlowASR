# pylint: disable=line-too-long
import os

from tensorflow_asr import tf
from tensorflow_asr.configs import SpeechConfig
from tensorflow_asr.features import speech_featurizers
from tensorflow_asr.utils import file_util

file_util.ENABLE_PATH_PREPROCESS = False
DIRNAME = os.path.dirname(__file__)

config_path = os.path.join(DIRNAME, "..", "examples", "configs", "log_mel_spectrogram.yml.j2")
print(config_path)
config = file_util.load_yaml(config_path)

speech_config = SpeechConfig(config["speech_config"])


def test():
    featurizer = speech_featurizers.SpeechFeaturizer(speech_config)
    audios = tf.convert_to_tensor(
        [speech_featurizers.load_and_convert_to_wav(os.path.join(DIRNAME, "featurizer", "test.flac")) for _ in range(4)], dtype=tf.string
    )
    signals = speech_featurizers.tf_read_raw_audio_batch(audios, sample_rate=16000)
    print(signals.numpy())
