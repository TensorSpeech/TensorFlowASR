from __future__ import absolute_import

from flask import Flask, Blueprint, jsonify, request
from flask_cors import CORS
from asr.SpeechToText import SpeechToText
from configs.FlaskConfig import FlaskConfig

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config.from_object(FlaskConfig)

asr_blueprint = Blueprint("asr", __name__, url_prefix="/api")


@asr_blueprint.route("/static")
def static_inference():
  pass


@asr_blueprint.route("/streaming")
def streaming_inference():
  pass


app.register_blueprint(asr_blueprint)
