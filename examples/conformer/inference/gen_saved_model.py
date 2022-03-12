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
import fire

from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")


from tensorflow_asr.configs.config import Config
from tensorflow_asr.helpers import featurizer_helpers
from tensorflow_asr.models.transducer.conformer import Conformer


def main(
    config: str = DEFAULT_YAML,
    h5: str = None,
    sentence_piece: bool = False,
    subwords: bool = False,
    output_dir: str = None,
):
    assert h5 and output_dir
    config = Config(config)
    tf.random.set_seed(0)
    tf.keras.backend.clear_session()

    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(
        config=config,
        subwords=subwords,
        sentence_piece=sentence_piece,
    )

    # build model
    conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
    conformer.make(speech_featurizer.shape)
    conformer.load_weights(h5, by_name=True)
    conformer.summary(line_length=100)
    conformer.add_featurizers(speech_featurizer, text_featurizer)

    class ConformerModule(tf.Module):
        def __init__(self, model: Conformer, name=None):
            super().__init__(name=name)
            self.model = model
            self.num_rnns = config.model_config["prediction_num_rnns"]
            self.rnn_units = config.model_config["prediction_rnn_units"]
            self.rnn_nstates = 2 if config.model_config["prediction_rnn_type"] == "lstm" else 1

        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def pred(self, signal):
            predicted = tf.constant(0, dtype=tf.int32)
            states = tf.zeros([self.num_rnns, self.rnn_nstates, 1, self.rnn_units], dtype=tf.float32)
            features = self.model.speech_featurizer.tf_extract(signal)
            encoded = self.model.encoder_inference(features)
            hypothesis = self.model._perform_greedy(encoded, tf.shape(encoded)[0], predicted, states, tflite=False)
            transcript = self.model.text_featurizer.indices2upoints(hypothesis.prediction)
            return transcript

    module = ConformerModule(model=conformer)
    tf.saved_model.save(module, export_dir=output_dir, signatures=module.pred.get_concrete_function())


if __name__ == "__main__":
    fire.Fire(main)
