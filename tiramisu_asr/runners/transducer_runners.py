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
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from .base_runners import BaseTrainer
from ..losses.rnnt_losses import rnnt_loss
from ..models.transducer import Transducer
from ..featurizers.text_featurizers import TextFeaturizer


class TransducerTrainer(BaseTrainer):
    def __init__(self,
                 config: dict,
                 text_featurizer: TextFeaturizer,
                 is_mixed_precision: bool = False,
                 strategy: tf.distribute.Strategy = None):
        """
        Args:
            config: the 'running_config' part in YAML config file'
            text_featurizer: the TextFeaturizer instance
            is_mixed_precision: a boolean for using mixed precision or not
        """
        super(TransducerTrainer, self).__init__(config, strategy=strategy)
        self.text_featurizer = text_featurizer
        self.is_mixed_precision = is_mixed_precision

    def set_train_metrics(self):
        self.train_metrics = {
            "transducer_loss": tf.keras.metrics.Mean("train_transducer_loss", dtype=tf.float32)
        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            "transducer_loss": tf.keras.metrics.Mean("eval_transducer_loss", dtype=tf.float32)
        }

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        _, features, input_length, labels, label_length, pred_inp = batch

        with tf.GradientTape() as tape:
            logits = self.model([features, pred_inp], training=True)
            tape.watch(logits)
            per_train_loss = rnnt_loss(
                logits=logits, labels=labels, label_length=label_length,
                logit_length=(input_length // self.time_reduction_factor),
                blank=self.text_featurizer.blank
            )
            train_loss = tf.nn.compute_average_loss(per_train_loss,
                                                    global_batch_size=self.global_batch_size)

            if self.is_mixed_precision:
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss)

        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_train_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_metrics["transducer_loss"].update_state(per_train_loss)

    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        _, features, input_length, labels, label_length, pred_inp = batch

        logits = self.model([features, pred_inp], training=False)
        eval_loss = rnnt_loss(
            logits=logits, labels=labels, label_length=label_length,
            logit_length=(input_length // self.time_reduction_factor),
            blank=self.text_featurizer.blank
        )

        self.eval_metrics["transducer_loss"].update_state(eval_loss)

    def compile(self,
                model: Transducer,
                optimizer: any,
                time_reduction_factor: int = 1,
                max_to_keep: int = 10):
        with self.strategy.scope():
            self.model = model
            self.model.summary(line_length=100)
            self.optimizer = tf.keras.optimizers.get(optimizer)
            self.time_reduction_factor = time_reduction_factor
            if self.is_mixed_precision:
                self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, "dynamic")
        self.create_checkpoint_manager(max_to_keep, model=self.model, optimizer=self.optimizer)

    def fit(self, train_dataset, eval_dataset=None, eval_train_ratio=1):
        self.set_train_data_loader(train_dataset)
        self.set_eval_data_loader(eval_dataset, eval_train_ratio)
        self.load_checkpoint()
        self.run()

# class TransducerTFLite(BaseInferencer):
#     def __init__(self,
#                  speech_config: dict,
#                  decoder_config: dict,
#                  saved_path: str):
#         super(TransducerTFLite, self).__init__(saved_path=saved_path, from_weights=False)
#         self.speech_featurizer = SpeechFeaturizer(speech_config)
#         self.decoder_config = decoder_config
#         self.text_featurizer = TextFeaturizer(self.decoder_config["vocabulary"])
#         self.hyps = None
#         self.scorer = None
#
#     def clear(self):
#         self.hyps = None
#         self.encoder.reset_all_variables()
#         self.prediction.reset_all_variables()
#         self.joint.reset_all_variables()
#
#     def load_model(self):
#         try:
#             self.encoder = tf.lite.Interpreter(os.path.join(self.saved_path, "encoder.tflite"))
#             self.prediction = tf.lite.Interpreter(os.path.join(self.saved_path, "prediction.tflite"))
#             self.joint = tf.lite.Interpreter(os.path.join(self.saved_path, "joint.tflite"))
#             print("Transducer loaded")
#         except Exception as e:
#             raise Exception(e)
#
#     def convert_saved_model_to_tflite(self,
#                                       saved_model_path: str,
#                                       tflite_path: str):
#         print("Loading saved model ...")
#         try:
#             saved_model = tf.saved_model.load(saved_model_path)
#         except Exception as e:
#             raise Exception(e)
#
#         print("Converting to tflite ...")
#
#         # See: https://stackoverflow.com/a/55732431/11037553
#         # The time dimension can be resize using tf.lite.Interpreter.resize_tensor_input
#         concrete_func = saved_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
#         f, c = self.speech_featurizer.compute_feature_dim()
#         concrete_func.inputs[0].set_shape([None, None, f, c])
#         concrete_func.inputs[1].set_shape([None, None])
#         # Convert to tflite
#         converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
#         converter.experimental_new_converter = True
#         converter.optimizations = [tf.lite.Optimize.DEFAULT]
#         converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
#         tflite_model = converter.convert()
#
#         print("Writing to file ...")
#         if not os.path.exists(os.path.dirname(tflite_path)): os.makedirs(os.path.dirname(tflite_path))
#         with open(tflite_path, "wb") as tflite_out:
#             tflite_out.write(tflite_model)
#
#         print(f"Done converting to {tflite_path}")
#
#     def compile(self, duration: float, *args, **kwargs):
#         f, c = self.speech_featurizer.compute_feature_dim()
#         self.t = self.speech_featurizer.compute_time_dim(duration)
#         self.encoder.resize_tensor_input(self.encoder.get_input_details()[0]["index"], [1, self.t, f, c])
#         self.encoder.allocate_tensors()
#         self.prediction.resize_tensor_input(self.prediction.get_input_details()[0]["index"], [1, 1])
#         self.prediction.allocate_tensors()
#
#         self.encoder.set_tensor(self.encoder.get_input_details()[0]["index"], tf.random.normal(shape=[1, self.t, f, c]))
#         self.encoder.invoke()
#         enc_out = self.encoder.get_tensor(self.encoder.get_output_details()[0]["index"])
#         self.joint.resize_tensor_input(self.joint.get_input_details()[0]["index"], enc_out.shape)
#         self.encoder.reset_all_variables()
#
#         self.prediction.set_tensor(self.prediction.get_input_details()[0]["index"], tf.random.normal(shape=[1, 1], dtype=tf.int32))
#         self.prediction.invoke()
#         pred_out = self.prediction.get_tensor(self.prediction.get_output_details()[0]["index"])
#         self.joint.resize_tensor_input(self.joint.get_input_details()[1]["index"], pred_out.shape)
#         self.prediction.reset_all_variables()
#
#         self.joint.allocate_tensors()
#
#     def preprocess(self, audio):
#         signal = read_raw_audio(audio, self.speech_featurizer.sample_rate)
#         features = self.speech_featurizer.extract(signal)
#         return np.expand_dims(features, axis=0)
#
#     def postprocess(self, predicted_hyps):
#         return self.text_featurizer._idx_to_char(predicted_hyps[0]["yseq"])
#
#     def infer(self, audio, streaming=False):
#         features = self.preprocess(audio)
#         return self.call(features, streaming)
#
#     def call(self, features, streaming):
#         predicted_hyps = self.recognize_beam(features, self.decoder_config["beam_width"],
#                                              nbest=1, norm_score=True, kept_hyps=self.hyps)
#         if streaming:
#             self.hyps = predicted_hyps
#         else:
#             self.clear()
#
#         return self.postprocess(predicted_hyps)
