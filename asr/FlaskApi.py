from __future__ import absolute_import

import functools
from logging import ERROR
import tensorflow as tf
from flask import Flask, Blueprint, jsonify, request, Response
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
@check_form_request
def asr_inference():
  if "payload" not in request.files.keys():
    return Response(jsonify({"error": "Missing audio binary file/blob"}),
                    status=400, mimetype="application/json")

  payload = request.files["payload"].read()
  sampleRate = request.form["sampleRate"]
  channels = request.form["channels"]
  transcript = asr(payload, sample_rate=sampleRate, channels=channels)
  return Response(jsonify({"payload": transcript}),
                  status=200, mimetype="application/json")


@asr_blueprint.route("/asrfile", methods=["POST"])
def asr_file():
  if "payload" not in request.files.keys():
    return Response(jsonify({"error": "Missing audio binary file/blob"}),
                    status=400, mimetype="application/json")

  request.files["payload"].save(app.config["STATIC_WAV_FILE"])
  transcript = asr(app.config["STATIC_WAV_FILE"])
  return Response(jsonify({"payload": transcript}),
                  status=200, mimetype="application/json")


@socketio.on("asr_streaming", namespace="/asr_streaming")
def asr_stream(content, sample_rate, channels):
  return asr_stream(content, sample_rate=sample_rate, channels=channels)


app.register_blueprint(asr_blueprint)
