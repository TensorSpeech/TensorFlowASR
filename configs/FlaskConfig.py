from __future__ import absolute_import

import os
from dotenv import load_dotenv

load_dotenv()


class FlaskConfig:
  """
  Set flask config vars from .env file
  """
  # General
  FLASK_APP = os.getenv("FLASK_APP", "api.py")

  # ASR
  MODEL_FILE = os.path.expanduser(os.getenv("MODEL_FILE", ""))
  STATIC_WAV_FILE = os.path.expanduser(
    os.getenv("STATIC_WAV_FILE", "/tmp/temp.wav"))

  CONFIG_PATH = os.path.expanduser(
    os.getenv("CONFIG_PATH", "/tmp/config.py"))
