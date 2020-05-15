from __future__ import absolute_import

import os
import pathlib
import librosa
import argparse
from logging import ERROR
import tensorflow as tf
from asr.SEGAN import SEGAN
from featurizers.SpeechFeaturizer import read_raw_audio

tf.get_logger().setLevel(ERROR)
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

parser = argparse.ArgumentParser(description="SEGAN Commands")

parser.add_argument("--mode", "-m", type=str, default="train",
                    help="Mode for training, testing or infering")

parser.add_argument("--config", "-c", type=str, default=None,
                    help="The file path of model configuration file")

parser.add_argument("--input_file_path", "-i", type=str, default=None,
                    help="Path to input file")

parser.add_argument("--export_file", "-e", type=str, default=None,
                    help="Path to the model file to be exported")

parser.add_argument("--output_file_path", "-o", type=str, default=None,
                    help="Path to output file")

args = parser.parse_args()

def main():
  assert args.mode in ["train", "test", "infer", "save",
                       "save_from_checkpoint", "convert_to_tflite"]

  if args.mode == "train":
    tf.random.set_seed(2020)
  else:
    tf.random.set_seed(0)

  segan = SEGAN(config_path=args.config, training=True)
  if args.mode == "train":
    assert args.export_file
    segan.train(export_dir=args.export_file)
  elif args.mode == "test":
    segan.test()
  elif args.mode == "save_from_checkpoint":
    assert args.export_file
    segan.save_from_checkpoint(args.export_file)
  elif args.mode == "save":
    assert args.export_file
    segan.save(args.export_file)
  elif args.mode == "infer":
    assert args.export_file and args.input_file_path and args.output_file_path
    msg = segan.load_model(args.export_file)
    assert msg is None
    signal = read_raw_audio(args.input_file_path, 16000)
    clean_signal = segan.generate(signal)
    librosa.output.write_wav(args.output_file_path, clean_signal, 16000)
  elif args.mode == "convert_to_tflite":
    assert args.export_file and args.output_file_path
    msg = segan.load_model(args.export_file)
    assert msg is None
    converter = tf.lite.TFLiteConverter.from_keras_model(segan.generator)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    supported_ops += [tf.lite.OpsSet.SELECT_TF_OPS]
    converter.target_spec.supported_ops = supported_ops
    tflite_model = converter.convert()

    tflite_model_dir = pathlib.Path(os.path.dirname(args.output_file_path))
    tflite_model_dir.mkdir(exist_ok=True, parents=True)

    tflite_model_file = tflite_model_dir/f"{os.path.basename(args.output_file_path)}"
    tflite_model_file.write_bytes(tflite_model)


if __name__ == "__main__":
  main()
