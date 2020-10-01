# Copyright 2020 Huy Le Nguyen (@usimarit)
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

import argparse
import tensorflow as tf

from tiramisu_asr.featurizers.speech_featurizers import read_raw_audio

parser = argparse.ArgumentParser(prog="Conformer non streaming")

parser.add_argument("filename", metavar="FILENAME",
                    help="audio file to be played back")

parser.add_argument("--tflite", type=str, default=None,
                    help="Path to conformer tflite")

parser.add_argument("--blank", type=int, default=0,
                    help="Path to conformer tflite")

parser.add_argument("--statesize", type=int, default=320,
                    help="Path to conformer tflite")

args = parser.parse_args()

tflitemodel = tf.lite.Interpreter(model_path=args.tflite)

signal = read_raw_audio(args.filename)

input_details = tflitemodel.get_input_details()
output_details = tflitemodel.get_output_details()
tflitemodel.resize_tensor_input(input_details[0]["index"], signal.shape)
tflitemodel.allocate_tensors()
tflitemodel.set_tensor(input_details[0]["index"], signal)
tflitemodel.set_tensor(
    input_details[1]["index"],
    tf.constant(args.blank, dtype=tf.int32)
)
tflitemodel.set_tensor(
    input_details[2]["index"],
    tf.zeros([1, 2, 1, args.statesize], dtype=tf.float32)
)
tflitemodel.invoke()
hyp = tflitemodel.get_tensor(output_details[0]["index"])

print("".join([chr(u) for u in hyp]))
