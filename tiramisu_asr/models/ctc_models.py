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

from ..featurizers.speech_featurizers import SpeechFeaturizer
from ..featurizers.text_featurizers import TextFeaturizer
from .deepspeech2.DeepSpeech2 import DeepSpeech2

SUPPORTED_MODELS = {
    "ds2": DeepSpeech2
}


def create_ctc_model(model_config: dict,
                     speech_featurizer: SpeechFeaturizer,
                     text_featurizer: TextFeaturizer):
    feature_dim, channel_dim = speech_featurizer.compute_feature_dim()

    features = tf.keras.Input(shape=(None, feature_dim, channel_dim),
                              dtype=tf.float32, name="features")
    # Model
    base_model = SUPPORTED_MODELS.get(model_config["name"], None)
    if not base_model:
        raise ValueError(f"Model not supported: {model_config['name']}\n"
                         f"Supported models: {SUPPORTED_MODELS.keys()}")
    base_model = base_model(model_config["architecture"])
    outputs = base_model(features=features, streaming=False)
    # Fully connected layer
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(units=text_featurizer.num_classes, activation="linear",
                              use_bias=True), name="fully_connected")(outputs)

    model = tf.keras.Model(inputs=features, outputs=outputs, name=f"ctc_{model_config['name']}")
    return model

# def create_ctc_train_model(ctc_model, text_featurizer, name="ctc_train_model"):
#     input_length = tf.keras.Input(shape=(), dtype=tf.int32, name="input_length")
#     label_length = tf.keras.Input(shape=(), dtype=tf.int32, name="label_length")
#     label = tf.keras.Input(shape=(None,), dtype=tf.int32, name="label")
#
#     ctc_loss = tf.keras.layers.Lambda(
#         ctc_loss_keras, output_shape=(1,), arguments={"num_classes": text_featurizer.num_classes},
#         name="ctc_loss")([ctc_model.outputs[0], input_length, label, label_length])
#
#     return tf.keras.Model(inputs=(ctc_model.inputs, input_length, label, label_length),
#                           outputs=ctc_loss, name=name)
