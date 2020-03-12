from __future__ import absolute_import

import functools
from logging import ERROR
import tensorflow as tf
from flask import Flask, Blueprint, jsonify, request
from flask_cors import CORS
from configs.FlaskConfig import FlaskConfig
from asr.SpeechToText import SpeechToText
from utils.Utils import check_key_in_dict

tf.get_logger().setLevel(ERROR)

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config.from_object(FlaskConfig)

asr_blueprint = Blueprint("asr", __name__, url_prefix="/api")

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
asr_streaming = SpeechToText(
  configs_path=app.config["UNI_CONFIG_PATH"],
  mode="infer_streaming")


@asr_blueprint.route("/", methods=["GET"])
def hello():
  return "Hello world"


@asr_blueprint.route("/asr", methods=["POST"])
@check_request
def asr_inference():
  payload = request.json["payload"]
  payload = bytes(payload, "utf-8")
  transcript = asr(audio=payload, model_file=app.config["MODEL_FILE"])
  return jsonify({"payload": transcript})


@asr_blueprint.route("/asr_streaming", methods=["POST"])
def streaming_inference():
  """
  Transcribes the audio bytes sent in realtime and
  immediately returns the text at that time-step
  :return: Json that contains the text
  """
  payload = request.json["payload"]
  payload = bytes(payload, "utf-8")
  features = asr_streaming.speech_featurizer.compute_speech_features(
    payload)
  print(features)
  return jsonify({"payload": "haha"})


app.register_blueprint(asr_blueprint)
