from __future__ import absolute_import

import os
from absl import flags

current_path = os.path.dirname(os.path.abspath(__file__))
_CONF_FILE = "/".join(
  [current_path, "..", "configs", "DefaultConfig.py"])

flags_obj = flags.FLAGS

flags.DEFINE_string(
  name="config",
  default=_CONF_FILE,
  help="The file path of model configuration file")

flags.DEFINE_string(
  name="mode",
  default="",
  help="Mode for training, testing or infering")

flags.DEFINE_string(
  name="speech_file_path",
  default=None,
  help="Path to the file containing speech file paths for inference")

flags.DEFINE_string(
  name="export_file",
  default=None,
  help="Path to the model file to be exported")

flags.DEFINE_string(
  name="output_file_path",
  default=None,
  help="Path to the file contains results")
