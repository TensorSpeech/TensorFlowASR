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

import tensorflow as tf

from tensorflow_asr.utils import cli_util, data_util

logger = tf.get_logger()


def main(
    file_path: str,
    tflite_path: str,
    previous_encoder_states_shape: list = None,
    previous_decoder_states_shape: list = None,
    blank_index: int = 0,
):
    tflitemodel = tf.lite.Interpreter(model_path=tflite_path)
    signal = data_util.read_raw_audio(file_path)
    signal = tf.reshape(signal, [1, -1])
    signal_length = tf.reshape(tf.shape(signal)[1], [1])

    input_details = tflitemodel.get_input_details()
    output_details = tflitemodel.get_output_details()
    tflitemodel.resize_tensor_input(input_details[0]["index"], signal.shape)
    tflitemodel.allocate_tensors()
    tflitemodel.set_tensor(input_details[0]["index"], signal)
    tflitemodel.set_tensor(input_details[1]["index"], signal_length)
    tflitemodel.set_tensor(input_details[2]["index"], tf.constant(blank_index, dtype=tf.int32))
    if previous_encoder_states_shape:
        tflitemodel.set_tensor(input_details[4]["index"], tf.zeros(previous_encoder_states_shape, dtype=tf.float32))
    if previous_decoder_states_shape:
        tflitemodel.set_tensor(input_details[5]["index"], tf.zeros(previous_decoder_states_shape, dtype=tf.float32))
    tflitemodel.invoke()
    hyp = tflitemodel.get_tensor(output_details[0]["index"])

    transcript = "".join([chr(u) for u in hyp])
    logger.info(f"Transcript: {transcript}")
    return transcript


if __name__ == "__main__":
    cli_util.run(main)
