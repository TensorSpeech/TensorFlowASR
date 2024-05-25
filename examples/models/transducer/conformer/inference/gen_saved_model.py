# # pylint: disable=no-member
# # Copyright 2020 Huy Le Nguyen (@nglehuy)
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import os

# import fire
# from tensorflow_asr import tf, keras

# from tensorflow_asr.configs import Config
# from tensorflow_asr.helpers import featurizer_helpers
# from tensorflow_asr.models.transducer.conformer import Conformer
# from tensorflow_asr.utils import env_util

# logger = env_util.setup_environment()

# DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config_wp.j2")


# def main(
#     config_path: str = DEFAULT_YAML,
#     saved: str = None,
#     output_dir: str = None,
# ):
#     assert saved and output_dir
#     tf.random.set_seed(0)
#     keras.backend.clear_session()

#     logger.info("Load config and featurizers ...")
#     config = Config(config_path)
#     speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(config=config)

#     logger.info("Build and load model ...")
#     conformer = Conformer(**config.model_config, vocab_size=text_featurizer.num_classes)
#     conformer.make(speech_featurizer.shape)
#     conformer.add_featurizers(speech_featurizer, text_featurizer)
#     conformer.load_weights(saved, by_name=True)
#     conformer.summary()

#     logger.info("Save model ...")
#     tf.saved_model.save(conformer, export_dir=output_dir, signatures=conformer.recognize_from_signal.get_concrete_function())


# if __name__ == "__main__":
#     fire.Fire(main)
