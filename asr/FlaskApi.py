from __future__ import absolute_import

import functools
from logging import ERROR
import tensorflow as tf
from flask import Flask, Blueprint, jsonify, request
from flask_socketio import SocketIO, emit
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


def check_request(func):
  @functools.wraps(func)
  def decorated_func(*args, **kwargs):
    try:
      check_key_in_dict(dictionary=request.json,
                        keys=["payload", "sampleRate", "channels"])
    except ValueError as e:
      return jsonify({"payload": str(e)})
    return func(*args, **kwargs)

  return decorated_func


asr = SpeechToText(configs_path=app.config["UNI_CONFIG_PATH"],
                   mode="infer_single")
is_asr_loaded = asr.load_model(app.config["MODEL_FILE"])

asr_streaming = SpeechToText(
  configs_path=app.config["UNI_CONFIG_PATH"],
  mode="infer_streaming")
is_asr_streaming_loaded = asr.load_model(app.config["MODEL_FILE"])


@asr_blueprint.route("/", methods=["GET"])
def hello():
  return "Hello world"


@asr_blueprint.route("/asr", methods=["POST"])
@check_request
def asr_inference():
  payload = request.json["payload"]
  payload = bytes(payload, "utf-8")
  transcript = asr(audio=payload, sample_rate=48000)
  return jsonify({"payload": transcript})


@socketio.on("asr_streaming", namespace="/asr_streaming")
def asr_stream(json):
  payload = json["payload"]
  # payload = bytes(payload)
  print(payload)
  sampleRate = int(json["sampleRate"])
  # transcript = asr_streaming(audio=payload,
  #                            sample_rate=sampleRate)
  return {"payload": payload}


app.register_blueprint(asr_blueprint)
