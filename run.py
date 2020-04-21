from __future__ import absolute_import

from logging import ERROR
import tensorflow as tf
from utils.Flags import args_parser
from asr.SpeechToText import SpeechToText
from asr.SEGAN import SEGAN

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

if args_parser.model == "asr":
  asr = SpeechToText(configs_path=args_parser.config)
  if args_parser.mode == "train":
    asr.train_and_eval(model_file=args_parser.export_file)
  elif args_parser.mode == "test":
    if args_parser.output_file_path is None:
      raise ValueError("Flag 'output_file_path must be set")
    asr.test(model_file=args_parser.export_file,
             output_file_path=args_parser.output_file_path)
  elif args_parser.mode == "infer":
    if args_parser.output_file_path is None:
      raise ValueError("Flag 'output_file_path must be set")
    if args_parser.input_file_path is None:
      raise ValueError("Flag 'input_file_path must be set")
    asr.infer(model_file=args_parser.export_file,
              input_file_path=args_parser.input_file_path,
              output_file_path=args_parser.output_file_path)
  elif args_parser.mode == "infer_single":
    if args_parser.input_file_path is None:
      raise ValueError("Flag 'input_file_path must be set")
    text = asr.infer_single(audio=args_parser.input_file_path)
    print(text)
  else:
    raise ValueError("Flag 'mode' must be either 'train', 'test' or 'infer'")
elif args_parser.model == "segan":
  segan = SEGAN(config_path=args_parser.config, mode="training")
  if args_parser.mode == "train":
    segan.train(export_dir=args_parser.export_file)
  elif args_parser.mode == "test":
    segan.test()
  else:
    raise ValueError("Flag 'mode' must be either 'train' or 'test'")
