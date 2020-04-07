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
  SECRET_KEY = "this is a secret key muahahah"

  # ASR
  MODEL_FILE = os.path.expanduser(os.getenv("MODEL_FILE", ""))
  STATIC_WAV_FILE = os.path.expanduser(
    os.getenv("STATIC_WAV_FILE", "/tmp/temp.wav"))

  UNI_CONFIG_PATH = os.path.expanduser(
    os.getenv("UNI_CONFIG_PATH", "/app/configs/UniConfig.py"))

  BI_CONFIG_PATH = os.path.expanduser(
    os.getenv("BI_CONFIG_PATH", "/app/configs/BiConfig.py"))
