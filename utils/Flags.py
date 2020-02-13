from __future__ import absolute_import

from absl import flags, app

import os

current_path = os.path.dirname(os.path.abspath(__file__))
_CONF_FILE = "/".join([current_path, "..", "configs", "DefaultConfig.py"])

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
    name="infer_file_path",
    default="",
    help="Path to the file containing speech file paths for inference")
