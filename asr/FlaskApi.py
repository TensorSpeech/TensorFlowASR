from __future__ import absolute_import

import soundfile as sf
import librosa
from flask import Flask, Blueprint, jsonify, request
from flask_cors import CORS
from configs.FlaskConfig import FlaskConfig
from asr.SpeechToText import SpeechToText

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


@asr_blueprint.route("/static", methods=["POST"])
def static_inference():
  """
  Saves audio bytes from requests into a file, then transcribes that
  file into text and return the text
  :return: Json that contains the text
  """
  request.files["payload"].save(app.config["STATIC_WAV_FILE"])
  asr = SpeechToText(configs_path=app.config["CONFIG_PATH"],
                     mode="infer_single")
  features = asr.speech_featurizer.compute_speech_features(
    app.config["STATIC_WAV_FILE"])

  transcript = asr(features=features,
                   model_file=app.config["MODEL_FILE"])

  print(transcript)

  return jsonify({"payload": transcript})


@asr_blueprint.route("/streaming", methods=["POST"])
def streaming_inference():
  """
  Transcribes the audio bytes sent in realtime and
  immediately returns the text at that time-step
  :return: Json that contains the text
  """


app.register_blueprint(asr_blueprint)
