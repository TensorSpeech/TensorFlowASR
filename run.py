from __future__ import absolute_import

import os
import argparse
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from asr.SpeechToText import SpeechToText
from asr.SEGAN import SEGAN
from featurizers.SpeechFeaturizer import read_raw_audio

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,",
          len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

tf.keras.backend.clear_session()

modes = ["test", "infer_single"]

parser = argparse.ArgumentParser(description="Script to run both SEGAN and ASR")

parser.add_argument("--mode", "-m", type=str, default="test",
                    help=f"Mode in {modes}")

parser.add_argument("--segan_config", "-s", type=str, default=None,
                    help="The file path of segan configuration file")

parser.add_argument("--asr_config", "-a", type=str, default=None,
                    help="The file path of asr configuration file")

parser.add_argument("--input_file_path", "-i", type=str, default=None,
                    help="Path to input file")

parser.add_argument("--output_file_path", "-o", type=str, default=None,
                    help="Path to output file")

parser.add_argument("--saved_segan", type=str, default=None,
                    help="Path to the saved segan weights")

parser.add_argument("--saved_asr", type=str, default=None,
                    help="Path to the saved asr model")

args = parser.parse_args()


def main():
  assert args.mode in modes, f"Mode must in {modes}"

  tf.random.set_seed(0)

  segan = SEGAN(config_path=args.segan_config, training=False)
  segan.load_model(args.saved_segan)

  asr = SpeechToText(configs_path=args.asr_config, noise_filter=segan)

  if args.mode == "test":
    assert args.output_file_path
    asr.test_with_noise_filter(model_file=args.saved_asr,
                               output_file_path=args.output_file_path)

  elif args.mode == "infer_single":
    assert args.input_file_path
    asr.load_model(args.saved_asr)
    signal = read_raw_audio(args.input_file_path, 16000)
    pred = asr.infer_single(signal)
    print(pred)


if __name__ == "__main__":
  main()
