from __future__ import absolute_import

from logging import ERROR
import tensorflow as tf
from absl import app
from utils.Flags import flags_obj
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


def main(argv):
  if flags_obj.export_file is None:
    raise ValueError("Flag 'export_file' must be set")
  if flags_obj.mode == "train":
    asr = SpeechToText(configs_path=flags_obj.config, mode="train")
    asr(model_file=flags_obj.export_file)
  elif flags_obj.mode == "save":
    asr = SpeechToText(configs_path=flags_obj.config, mode="infer")
    asr.save_model(flags_obj.export_file)
  elif flags_obj.mode == "test":
    tf.compat.v1.set_random_seed(0)
    asr = SpeechToText(configs_path=flags_obj.config, mode="test")
    if flags_obj.output_file_path is None:
      raise ValueError("Flag 'output_file_path must be set")
    asr(model_file=flags_obj.export_file,
        output_file_path=flags_obj.output_file_path)
  elif flags_obj.mode == "infer":
    tf.compat.v1.set_random_seed(0)
    if flags_obj.output_file_path is None:
      raise ValueError("Flag 'output_file_path must be set")
    if flags_obj.speech_file_path is None:
      raise ValueError("Flag 'speech_file_path must be set")
    asr = SpeechToText(configs_path=flags_obj.config, mode="infer")
    asr(model_file=flags_obj.export_file,
        speech_file_path=flags_obj.speech_file_path,
        output_file_path=flags_obj.output_file_path)
  else:
    raise ValueError("Flag 'mode' must be either 'save', 'train', \
                         'test' or 'infer'")


if __name__ == '__main__':
  app.run(main)
