from __future__ import absolute_import

import functools
from logging import ERROR
import tensorflow as tf
from flask import Flask, Blueprint, jsonify, request, make_response
from flask_socketio import SocketIO
from flask_cors import CORS
from configs.FlaskConfig import FlaskConfig
from asr.SpeechToText import SpeechToText
from utils.Utils import check_key_in_dict

tf.get_logger().setLevel(ERROR)

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


asr = SpeechToText(configs_path=app.config["UNI_CONFIG_PATH"],
                   mode="infer_single")
is_asr_loaded = asr.load_model(app.config["MODEL_FILE"])
print(is_asr_loaded)
asr_streaming = SpeechToText(
  configs_path=app.config["UNI_CONFIG_PATH"],
  mode="infer_streaming")
is_asr_streaming_loaded = asr.load_model(app.config["MODEL_FILE"])
print(is_asr_streaming_loaded)


@asr_blueprint.route("/", methods=["GET"])
def hello():
  return "Hello world"


@asr_blueprint.route("/asr", methods=["POST"])
@check_form_request
def inference():
  if is_asr_loaded:
    return make_response(({"payload": is_asr_loaded}, 200))
  if "payload" not in request.files.keys():
    return make_response((
      {"error": "Missing audio binary file/blob"}, 400))

  payload = request.files["payload"].read()
  sampleRate = int(request.form["sampleRate"])
  channels = int(request.form["channels"])
  transcript = asr(audio=payload, sample_rate=sampleRate, channels=channels)
  return make_response(({"payload": transcript}, 200))


@asr_blueprint.route("/asrfile", methods=["POST"])
def file():
  if is_asr_loaded:
    return make_response(({"payload": is_asr_loaded}, 200))
  if "payload" not in request.files.keys():
    return make_response((
      {"error": "Missing audio binary file/blob"}, 400))

  request.files["payload"].save(app.config["STATIC_WAV_FILE"])
  transcript = asr(audio=app.config["STATIC_WAV_FILE"])
  return make_response(({"payload": transcript}, 200))


@socketio.on("connect", namespace="/asr_streaming")
def connect():
  if is_asr_streaming_loaded:
    return is_asr_streaming_loaded, False
  return "Connected", True


@socketio.on("asr_streaming", namespace="/asr_streaming")
def streaming(content, sample_rate, channels):
  return asr_streaming(audio=content, sample_rate=int(sample_rate),
                       channels=int(channels))


app.register_blueprint(asr_blueprint)
