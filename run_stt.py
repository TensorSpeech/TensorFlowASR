from __future__ import absolute_import

import argparse
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from asr.SpeechCTC import SpeechCTC
from featurizers.SpeechFeaturizer import read_raw_audio

tf.keras.backend.clear_session()

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

    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    tf.keras.mixed_precision.experimental.set_policy(policy)

modes = ["train", "train_builtin", "test", "infer", "infer_single",
         "convert_to_tflite", "infer_interpreter",
         "save", "save_from_checkpoint", "save_from_checkpoint_builtin"]

parser = argparse.ArgumentParser(description="ASR Commands")

parser.add_argument("--mode", "-m", type=str, default="train",
                    help=f"Mode in {modes}")

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
    assert args.mode in modes, f"Mode must in {modes}"

    if args.mode in ["train", "train_keras"]:
        tf.random.set_seed(2020)
    else:
        tf.random.set_seed(0)

    asr = SpeechCTC(configs_path=args.config)

    if args.mode == "train":
        asr.train_and_eval(model_file=args.export_file, gpu=len(gpus))

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

    elif args.mode == "convert_to_tflite":
        assert args.export_file and args.output_file_path
        asr.convert_to_tflite(args.export_file, 4, args.output_file_path)

    elif args.mode == "infer_interpreter":
        assert args.export_file and args.input_file_path
        msg = asr.load_interpreter(args.export_file)
        assert msg is None
        signal = read_raw_audio(args.input_file_path, asr.configs["speech_conf"]["sample_rate"])
        pred = asr.infer_single_interpreter(signal, 4)
        print(pred)


if __name__ == "__main__":
    main()
