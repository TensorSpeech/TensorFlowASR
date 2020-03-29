from __future__ import absolute_import

from logging import ERROR
import tensorflow as tf
from utils.Flags import args_parser
from asr.SpeechToText import SpeechToText

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

if args_parser.export_file is None:
  raise ValueError("Flag 'export_file' must be set")
if args_parser.mode == "train":
  tf.compat.v1.set_random_seed(1)
  asr = SpeechToText(configs_path=args_parser.config, mode="train")
  asr(model_file=args_parser.export_file)
elif args_parser.mode == "test":
  asr = SpeechToText(configs_path=args_parser.config, mode="test")
  if args_parser.output_file_path is None:
    raise ValueError("Flag 'output_file_path must be set")
  asr(model_file=args_parser.export_file,
      output_file_path=args_parser.output_file_path)
elif args_parser.mode == "infer":
  if args_parser.output_file_path is None:
    raise ValueError("Flag 'output_file_path must be set")
  if args_parser.input_file_path is None:
    raise ValueError("Flag 'input_file_path must be set")
  asr = SpeechToText(configs_path=args_parser.config, mode="infer")
  asr(model_file=args_parser.export_file,
      input_file_path=args_parser.input_file_path,
      output_file_path=args_parser.output_file_path)
elif args_parser.mode == "save":
  asr = SpeechToText(configs_path=args_parser.config, mode="infer")
  asr.save_infer_model(args_parser.export_file)
elif args_parser.mode == "save_from_weights":
  if args_parser.input_file_path is None:
    raise ValueError("Flag 'input_file_path must be set")
  asr = SpeechToText(configs_path=args_parser.config, mode="infer")
  asr.load_infer_model(args_parser.input_file_path)
  asr.save_infer_model_from_weights(args_parser.export_file)
else:
  raise ValueError("Flag 'mode' must be either 'train', 'test', \
    'infer', 'save' or 'save_from_weights'")
