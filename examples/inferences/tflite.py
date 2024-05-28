# Copyright 2020 Huy Le Nguyen (@nglehuy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow_text as tft
from tensorflow.lite.python import interpreter

from tensorflow_asr import tf
from tensorflow_asr.utils import cli_util, data_util, env_util

env_util.setup_logging()
logger = tf.get_logger()


def main(
    audio_file_path: str,
    tflite: str,
    sample_rate: int = 16000,
    blank: int = 0,
):
    wav = data_util.load_and_convert_to_wav(audio_file_path, sample_rate=sample_rate)
    signal = data_util.read_raw_audio(wav)
    signal = tf.reshape(signal, [1, -1])
    signal_length = tf.reshape(tf.shape(signal)[1], [1])

    tflitemodel = interpreter.InterpreterWithCustomOps(model_path=tflite, custom_op_registerers=tft.tflite_registrar.SELECT_TFTEXT_OPS)
    input_details = tflitemodel.get_input_details()
    output_details = tflitemodel.get_output_details()

    tflitemodel.resize_tensor_input(input_details[0]["index"], signal.shape, strict=True)
    tflitemodel.allocate_tensors()
    tflitemodel.set_tensor(input_details[0]["index"], signal)
    tflitemodel.set_tensor(input_details[1]["index"], signal_length)
    tflitemodel.set_tensor(input_details[2]["index"], tf.ones(input_details[2]["shape"], dtype=input_details[2]["dtype"]) * blank)
    tflitemodel.set_tensor(input_details[3]["index"], tf.zeros(input_details[3]["shape"], dtype=input_details[3]["dtype"]))
    tflitemodel.set_tensor(input_details[4]["index"], tf.zeros(input_details[4]["shape"], dtype=input_details[4]["dtype"]))

    tflitemodel.invoke()

    transcript = tflitemodel.get_tensor(output_details[0]["index"])
    tokens = tflitemodel.get_tensor(output_details[1]["index"])
    next_tokens = tflitemodel.get_tensor(output_details[2]["index"])
    if len(output_details) > 4:
        next_encoder_states = tflitemodel.get_tensor(output_details[3]["index"])
        next_decoder_states = tflitemodel.get_tensor(output_details[4]["index"])
    elif len(output_details) > 3:
        next_encoder_states = None
        next_decoder_states = tflitemodel.get_tensor(output_details[3]["index"])
    else:
        next_encoder_states = None
        next_decoder_states = None

    logger.info(f"Transcript: {transcript}")
    logger.info(f"Tokens: {tokens}")
    logger.info(f"Next tokens: {next_tokens}")
    logger.info(f"Next encoder states: {None if next_encoder_states is None else next_encoder_states.shape}")
    logger.info(f"Next decoder states: {None if next_decoder_states is None else next_decoder_states.shape}")


if __name__ == "__main__":
    cli_util.run(main)
