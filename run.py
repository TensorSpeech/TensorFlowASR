from __future__ import absolute_import

import librosa
from logging import ERROR
import tensorflow as tf
from utils.Flags import args_parser
from asr.SpeechToText import SpeechToText
from asr.SEGAN import SEGAN
from featurizers.SpeechFeaturizer import read_raw_audio, preemphasis

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

if args_parser.mode in ["train", "train_keras"]:
  tf.compat.v1.set_random_seed(2020)
else:
  tf.compat.v1.set_random_seed(0)

if args_parser.model == "asr":
  asr = SpeechToText(configs_path=args_parser.config)
  if args_parser.mode == "train":
    asr.train_and_eval(model_file=args_parser.export_file)
  elif args_parser.mode == "train_keras":
    asr.keras_train_and_eval(model_file=args_parser.export_file)
  elif args_parser.mode == "test":
    if args_parser.output_file_path is None:
      raise ValueError("Flag 'output_file_path' must be set")
    asr.test(model_file=args_parser.export_file,
             output_file_path=args_parser.output_file_path)
  elif args_parser.mode == "infer":
    if args_parser.output_file_path is None:
      raise ValueError("Flag 'output_file_path' must be set")
    if args_parser.input_file_path is None:
      raise ValueError("Flag 'input_file_path' must be set")
    asr.infer(model_file=args_parser.export_file,
              input_file_path=args_parser.input_file_path,
              output_file_path=args_parser.output_file_path)
  elif args_parser.mode == "infer_single":
    if args_parser.input_file_path is None:
      raise ValueError("Flag 'input_file_path' must be set")
    signal = read_raw_audio(args_parser.input_file_path)
    signal = preemphasis(signal, 0.95)
    text = asr.infer_single(signal)
    print(text)
  elif args_parser.mode == "save":
    asr.save_model(args_parser.export_file)
  elif args_parser.mode == "load":
    asr.load_model(args_parser.export_file)
  else:
    raise ValueError("Flag 'mode' must be either 'train', 'test' or 'infer'")
elif args_parser.model == "segan":
  segan = SEGAN(config_path=args_parser.config, training=True)
  if args_parser.mode == "train":
    segan.train(export_dir=args_parser.export_file)
  elif args_parser.mode == "test":
    segan.test()
  elif args_parser.mode == "save_from_ckpt":
    segan.save_from_checkpoint(args_parser.export_file)
  elif args_parser.mode == "save":
    segan.save(args_parser.export_file)
  elif args_parser.mode == "infer":
    segan.load_model(args_parser.export_file)
    if args_parser.input_file_path and args_parser.output_file_path:
      signal = read_raw_audio(args_parser.input_file_path, 16000)
      signal = preemphasis(signal, 0.95)
      clean_signal = segan.generate(signal)
      clean_signal = clean_signal.numpy()
      librosa.output.write_wav(args_parser.output_file_path, clean_signal, 16000)
  else:
    raise ValueError("Flag 'mode' must be either 'train' or 'test'")
