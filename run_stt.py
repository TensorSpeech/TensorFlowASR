from __future__ import absolute_import

import os
import argparse
import pathlib
from logging import ERROR
import tensorflow as tf
from asr.SpeechToText import SpeechToText
from featurizers.SpeechFeaturizer import read_raw_audio
from data.Dataset import Dataset

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

parser = argparse.ArgumentParser(description="ASR Commands")

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

parser.add_argument("--ckpt_index", type=int, default=-1,
                    help="Checkpoint index")

args = parser.parse_args()


def main():
  assert args.mode in ["train", "train_builtin", "test", "infer", "infer_single",
                       "create_tfrecords", "convert_to_tflite",
                       "save", "save_from_checkpoint", "save_from_checkpoint_builtin"]

  if args.mode in ["train", "train_keras", "convert_to_tflite"]:
    tf.random.set_seed(2020)
  else:
    tf.random.set_seed(0)

  asr = SpeechToText(configs_path=args.config)

  if args.mode == "train":
    asr.train_and_eval(model_file=args.export_file)

  elif args.mode == "train_builtin":
    asr.train_and_eval_builtin(model_file=args.export_file)

  elif args.mode == "test":
    assert args.output_file_path and args.export_file
    asr.test(model_file=args.export_file, output_file_path=args.output_file_path)

  elif args.mode == "infer":
    assert args.input_file_path and args.output_file_path and args.export_file
    asr.infer(model_file=args.export_file, input_file_path=args.input_file_path,
              output_file_path=args.output_file_path)

  elif args.mode == "infer_single":
    assert args.input_file_path and args.export_file
    asr.load_model(args.export_file)
    signal = read_raw_audio(args.input_file_path)
    text = asr.infer_single(signal)
    print(text)

  elif args.mode == "save":
    assert args.export_file
    asr.save_model(args.export_file)

  elif args.mode == "save_from_checkpoint":
    assert args.export_file
    asr.save_from_checkpoint(args.export_file, args.ckpt_index)

  elif args.mode == "save_from_checkpoint_keras":
    assert args.export_file
    asr.save_from_checkpoint(args.export_file, args.ckpt_index, is_builtin=True)

  elif args.mode == "create_tfrecords":
    config = asr.configs
    tfrecords_dir = config["tfrecords_dir"]
    eval_data = config["eval_data_transcript_paths"]
    augmentations = config["augmentations"]

    train_dataset = Dataset(config["train_data_transcript_paths"], tfrecords_dir, mode="train")
    train_dataset.create_tfrecords(augmentations=augmentations)
    test_dataset = Dataset(config["test_data_transcript_paths"], tfrecords_dir, mode="test")
    test_dataset.create_tfrecords(augmentations=[None])
    if eval_data:
      eval_dataset = Dataset(eval_data, tfrecords_dir, mode="eval")
      eval_dataset.create_tfrecords(augmentations=[None])

  elif args.mode == "convert_to_tflite":
    assert args.export_file and args.output_file_path
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=args.export_file)
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

    tflite_model_dir = pathlib.Path(os.path.dirname(args.output_file_path))
    tflite_model_dir.mkdir(exist_ok=True, parents=True)

    tflite_model_file = tflite_model_dir / f"{os.path.basename(args.output_file_path)}"
    tflite_model_file.write_bytes(tflite_model)


if __name__ == "__main__":
  main()
