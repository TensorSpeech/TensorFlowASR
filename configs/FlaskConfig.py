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
    SECRET_KEY = "this is a secret key muahahahahahaha"

    # ASR
    STATIC_WAV_FILE = os.path.abspath(os.getenv("STATIC_WAV_FILE", "/tmp/temp.wav"))

    SAVED_MODEL = os.path.abspath(os.getenv("SAVED_MODEL", "/app/trained/deepspeech2/model"))
    STREAMING_SAVED_MODEL = os.path.abspath(os.getenv("STREAMING_SAVED_MODEL", "/app/trained/deepspeech2/streaming"))

    TFLITE = os.path.abspath(os.getenv("TFLITE", "/app/trained/deepspeech2/model.tflite"))
    STREAMING_TFLITE = os.path.abspath(os.getenv("STREAMING_TFLITE", "/app/trained/deepspeech2/streaming.tflite"))
    TFLITE_LENGTH = int(os.getenv("ASR_TFLITE_LENGTH", 4))

    CONFIG_PATH = os.path.abspath(os.getenv("CONFIG_PATH", ""))
    STREAMING_CONFIG_PATH = os.path.abspath(os.getenv("STREAMING_CONFIG_PATH", ""))

    # SEGAN
    SEGAN_SAVED_WEIGHTS = os.path.abspath(os.getenv("SEGAN_SAVED_WEIGHTS", "/app/trained/segan/trained"))
    SEGAN_TFLITE = os.path.abspath(os.getenv("SEGAN_TFLITE", "/app/trained/segan/model.tflite"))
    SEGAN_CONFIG_PATH = os.path.abspath(os.getenv("SEGAN_CONFIG_PATH", "/app/configs/SeganConfig.py"))
