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

import tensorflow as tf

from tiramisu_asr.models.transducer import Transducer
from tiramisu_asr.featurizers.text_featurizers import TextFeaturizer
from tiramisu_asr.featurizers.speech_featurizers import TFSpeechFeaturizer, read_raw_audio
from tiramisu_asr.utils.utils import merge_features_to_channels

text_featurizer = TextFeaturizer({
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
    "feature_type": "logfbank",
    "preemphasis": 0.97,
    # "delta": True,
    # "delta_delta": True,
    "normalize_signal": True,
    "normalize_feature": True,
    "normalize_per_feature": False,
    # "pitch": False,
})

inp = tf.keras.Input(shape=[None, 80, 1])
enc = merge_features_to_channels(inp)
enc = tf.keras.layers.LSTM(350, return_sequences=True)(enc)

enc_model = tf.keras.Model(inputs=inp, outputs=enc)

model = Transducer(
    encoder=enc_model,
    blank=0,
    vocabulary_size=text_featurizer.num_classes,
    embed_dim=350, embed_dropout=0.0, num_lstms=1, lstm_units=320, joint_dim=1024
)

model._build([1, 50, 80, 1])
model.summary(line_length=100)

model.save_weights("/tmp/transducer.h5")

model.add_featurizers(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer
)

features = tf.random.normal(shape=[5, 50, 80, 1], stddev=127., mean=247.)
hyps = model.recognize(features)

print(hyps)

signal = read_raw_audio("/home/nlhuy/Desktop/test/11003.wav", speech_featurizer.sample_rate)

# hyps = model.recognize_tflite(signal)
#
# print(hyps)

# hyps = model.recognize_beam(tf.expand_dims(speech_featurizer.tf_extract(signal), 0))

print(hyps)

# hyps = model.recognize_beam_tflite(signal)
#
# print(hyps.numpy().decode("utf-8"))
#
concrete_func = model.recognize_tflite.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func]
)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
converter.convert()
