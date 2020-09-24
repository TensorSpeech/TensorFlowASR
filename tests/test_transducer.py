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

import os
# import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
import tensorflow as tf

from tiramisu_asr.models.transducer import Transducer
from tiramisu_asr.featurizers.text_featurizers import CharFeaturizer
from tiramisu_asr.featurizers.speech_featurizers import TFSpeechFeaturizer, read_raw_audio

text_featurizer = CharFeaturizer({
    "vocabulary": None,
    "blank_at_zero": True,
    "beam_width": 5,
    "norm_score": True
})

speech_featurizer = TFSpeechFeaturizer({
    "sample_rate": 16000,
    "frame_ms": 25,
    "stride_ms": 10,
    "num_feature_bins": 80,
    "feature_type": "log_mel_spectrogram",
    "preemphasis": 0.97,
    "normalize_signal": True,
    "normalize_feature": True,
    "normalize_per_feature": False
})

inp = tf.keras.Input(shape=[None, 80, 1])
enc = tf.keras.layers.Reshape([-1, 80])(inp)
enc = tf.keras.layers.LSTM(256, return_sequences=True)(enc)

enc_model = tf.keras.Model(inputs=inp, outputs=enc)

model = Transducer(
    encoder=enc_model,
    vocabulary_size=text_featurizer.num_classes,
    embed_dim=256, embed_dropout=0.0, num_rnns=1, rnn_units=320,
    rnn_type="gru", layer_norm=True, joint_dim=320
)

model._build(speech_featurizer.shape)
model.summary(line_length=100)

model.save_weights("/tmp/transducer.h5")

model.add_featurizers(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer
)

features = tf.zeros(shape=[5, 50, 80, 1], dtype=tf.float32)
pred = model.recognize(features)
print(pred)
pred = model.recognize_beam(features)
print(pred)

# stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# logdir = '/tmp/logs/func/%s' % stamp
# writer = tf.summary.create_file_writer(logdir)
#
signal = read_raw_audio("/home/nlhuy/Desktop/test/11003.wav", speech_featurizer.sample_rate)
#
# tf.summary.trace_on(graph=True, profiler=True)
# hyps = model.recognize_tflite(signal, 0, tf.zeros([1, 2, 1, 320], dtype=tf.float32))
# with writer.as_default():
#     tf.summary.trace_export(
#         name="recognize_tflite",
#         step=0,
#         profiler_outdir=logdir)
#
# print(hyps[0])
#
# # hyps = model.recognize_beam(features)
#
#

# hyps = model.recognize_beam(tf.expand_dims(speech_featurizer.tf_extract(signal), 0))

# print(hyps)

# hyps = model.recognize_beam_tflite(signal)

# print(hyps.numpy().decode("utf-8"))

concrete_func = model.make_tflite_function(greedy=True).get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite = converter.convert()

tflitemodel = tf.lite.Interpreter(model_content=tflite)

input_details = tflitemodel.get_input_details()
output_details = tflitemodel.get_output_details()
tflitemodel.resize_tensor_input(input_details[0]["index"], signal.shape)
tflitemodel.allocate_tensors()
tflitemodel.set_tensor(input_details[0]["index"], signal)
tflitemodel.set_tensor(
    input_details[1]["index"],
    tf.constant(text_featurizer.blank, dtype=tf.int32)
)
tflitemodel.set_tensor(
    input_details[2]["index"],
    tf.zeros([1, 1, 1, 320], dtype=tf.float32)
)
tflitemodel.invoke()
hyp = tflitemodel.get_tensor(output_details[0]["index"])

print(hyp)
