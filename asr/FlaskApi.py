from __future__ import absolute_import
from featurizers.SpeechFeaturizer import read_raw_audio
from utils.Utils import check_key_in_dict
from asr.SEGAN import SEGAN
from asr.SpeechToText import SpeechToText
from configs.FlaskConfig import FlaskConfig
from flask_cors import CORS
from flask_socketio import SocketIO
from flask import Flask, Blueprint, request, make_response
import tensorflow as tf

import os
import functools
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

tf.get_logger().setLevel('ERROR')


tf.random.set_seed(0)

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
            check_key_in_dict(dictionary=request.files, keys=["payload"])
        except ValueError as e:
            return make_response(({"payload": str(e)}, 400))
        return func(*args, **kwargs)

    return decorated_func


segan = None

if app.config["SEGAN_SAVED_WEIGHTS"]:
    segan = SEGAN(config_path=app.config["SEGAN_CONFIG_PATH"], training=False)
    try:
        segan.convert_to_tflite(app.config["SEGAN_SAVED_WEIGHTS"], app.config["SEGAN_TFLITE"])
        segan_error = segan.load_interpreter(app.config["SEGAN_TFLITE"])
    except Exception as e:
        segan = None
    if segan_error is not None:
        segan = None

asr = SpeechToText(configs_path=app.config["CONFIG_PATH"])
asr.convert_to_tflite(app.config["SAVED_MODEL"], app.config["TFLITE_LENGTH"], app.config["TFLITE"])
asr_error = asr.load_interpreter(app.config["TFLITE"])


# asr_streaming = SpeechToText(configs_path=app.config["UNI_CONFIG_PATH"])
# asr_streaming_error = asr_streaming.load_model(app.config["MODEL_FILE"])


def predict(signal, streaming=False):
    signal = read_raw_audio(signal, asr.configs["speech_conf"]["sample_rate"])
    if not streaming:
        if asr_error:
            return asr_error
        if segan:
            signal = segan.generate_interpreter(signal)
        return asr.infer_single_interpreter(signal, app.config["TFLITE_LENGTH"])
    # if streaming and not asr_streaming_error:
    #   return asr_streaming.infer_single(signal)
    return "Streaming is not supported yet"


@asr_blueprint.route("/", methods=["GET"])
def hello():
    return "Hello, this is Huy Le Nguyen's ASR APIs"


@asr_blueprint.route("/asr", methods=["POST"])
@check_form_request
def inference():
    payload = request.files["payload"].read()
    transcript = predict(payload)
    return make_response(({"payload": transcript}, 200))


@asr_blueprint.route("/asrfile", methods=["POST"])
def file():
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
