from __future__ import absolute_import

import functools
from logging import ERROR
import tensorflow as tf
from flask import Flask, Blueprint, request, make_response
from flask_socketio import SocketIO
from flask_cors import CORS
from configs.FlaskConfig import FlaskConfig
from asr.SpeechToText import SpeechToText
from asr.SEGAN import SEGAN
from utils.Utils import check_key_in_dict
from featurizers.SpeechFeaturizer import preemphasis, read_raw_audio

tf.get_logger().setLevel(ERROR)
tf.compat.v1.set_random_seed(0)

socketio = SocketIO()

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config.from_object(FlaskConfig)

socketio.init_app(app, cors_allowed_origins="*")

asr_blueprint = Blueprint("asr", __name__)

"""
  Json request format:
  {
    "payload": "some data as bytes",
  }
"""


def check_form_request(func):
  @functools.wraps(func)
  def decorated_func(*args, **kwargs):
    try:
      check_key_in_dict(dictionary=request.form,
                        keys=["sampleRate", "channels"])
    except ValueError as e:
      return make_response(({"payload": str(e)}, 400))
    return func(*args, **kwargs)

  return decorated_func


segan = SEGAN(config_path=app.config["SEGAN_CONFIG_PATH"], training=False)
segan_error = segan.load_model(app.config["SEGAN_FILE"])

if segan_error:
  segan = None

asr = SpeechToText(configs_path=app.config["BI_CONFIG_PATH"], noise_filter=segan)
asr_error = asr.load_model(app.config["MODEL_FILE"])


# asr_streaming = SpeechToText(configs_path=app.config["UNI_CONFIG_PATH"])
# asr_streaming_error = asr_streaming.load_model(app.config["MODEL_FILE"])


def predict(signal, streaming=False):
  signal = read_raw_audio(signal, asr.configs["sample_rate"])
  if not streaming and not asr_error:
    return asr.infer_single(signal)
  # if streaming and not asr_streaming_error:
  #   return asr_streaming.infer_single(signal)
  return "Model is not trained"


@asr_blueprint.route("/", methods=["GET"])
def hello():
  return "Hello, this is Huy Le Nguyen's ASR APIs"


@asr_blueprint.route("/asr", methods=["POST"])
@check_form_request
def inference():
  if "payload" not in request.files.keys():
    return make_response((
      {"error": "Missing audio binary file/blob"}, 400))

  payload = request.files["payload"].read()
  transcript = predict(payload)
  return make_response(({"payload": transcript}, 200))


@asr_blueprint.route("/asrfile", methods=["POST"])
def file():
  if "payload" not in request.files.keys():
    return make_response(({"error": "Missing audio binary file/blob"}, 400))

  request.files["payload"].save(app.config["STATIC_WAV_FILE"])
  transcript = predict(app.config["STATIC_WAV_FILE"])
  return make_response(({"payload": transcript}, 200))


# @socketio.on("connect", namespace="/asr_streaming")
# def connect():
#   if asr_streaming_error:
#     return asr_streaming_error, False
#   return "Connected", True
#
#
# @socketio.on("asr_streaming", namespace="/asr_streaming")
# def asr_streaming(content):
#   return predict(content, streaming=True)


app.register_blueprint(asr_blueprint)
