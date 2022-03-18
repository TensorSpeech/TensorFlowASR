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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

logger = tf.get_logger()

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config_wp.j2")

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.speech_featurizers import SpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer
from tensorflow_asr.models.ctc.jasper import Jasper


def test_jasper():
    config = Config(DEFAULT_YAML)

    text_featurizer = CharFeaturizer(config.decoder_config)

    speech_featurizer = SpeechFeaturizer(config.speech_config)

    model = Jasper(vocab_size=text_featurizer.num_classes, **config.model_config)

    model.make(speech_featurizer.shape)
    model.summary()

    model.add_featurizers(speech_featurizer=speech_featurizer, text_featurizer=text_featurizer)

    concrete_func = model.make_tflite_function(greedy=False).get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.convert()

    logger.info("Converted successfully with beam search")

    concrete_func = model.make_tflite_function(greedy=True).get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.convert()

    logger.info("Converted successfully with greedy")
