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
import numpy as np
import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from ..featurizers.speech_featurizers import SpeechFeaturizer, read_raw_audio
from ..featurizers.text_featurizers import TextFeaturizer
from ..losses.ctc_losses import ctc_loss
from .base_runners import BaseTrainer, BaseInferencer
from ..utils.utils import bytes_to_string


class CTCTrainer(BaseTrainer):
    """ Trainer for CTC Models """

    def __init__(self,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 config: dict,
                 is_mixed_precision: bool = False,
                 strategy: tf.distribute.Strategy = None):
        super(CTCTrainer, self).__init__(config=config, strategy=strategy)
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer
        self.is_mixed_precision = is_mixed_precision

    def set_train_metrics(self):
        self.train_metrics = {
            "ctc_loss": tf.keras.metrics.Mean("train_ctc_loss", dtype=tf.float32)
        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            "ctc_loss": tf.keras.metrics.Mean("eval_ctc_loss", dtype=tf.float32),
        }

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        _, features, input_length, labels, label_length, _ = batch

        with tf.GradientTape() as tape:
            y_pred = self.model(features, training=True)
            tape.watch(y_pred)
            per_train_loss = ctc_loss(
                y_true=labels, y_pred=y_pred,
                input_length=(input_length // self.model.time_reduction_factor),
                label_length=label_length,
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

        self.train_metrics["ctc_loss"].update_state(per_train_loss)

    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        _, features, input_length, labels, label_length, _ = batch

        logits = self.model(features, training=False)

        per_eval_loss = ctc_loss(
            y_true=labels, y_pred=logits,
            input_length=(input_length // self.model.time_reduction_factor),
            label_length=label_length,
            blank=self.text_featurizer.blank
        )

        # Update metrics
        self.eval_metrics["ctc_loss"].update_state(per_eval_loss)

    def compile(self, model: tf.keras.Model,
                optimizer: any,
                max_to_keep: int = 10):
        with self.strategy.scope():
            self.model = model
            self.model.summary(line_length=100)
            self.optimizer = tf.keras.optimizers.get(optimizer)
            if self.is_mixed_precision:
                self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer, "dynamic")
        self.create_checkpoint_manager(max_to_keep, model=self.model, optimizer=self.optimizer)

    def fit(self, train_dataset, eval_dataset=None, eval_train_ratio=1):
        self.set_train_data_loader(train_dataset)
        self.set_eval_data_loader(eval_dataset, eval_train_ratio)
        self.load_checkpoint()
        self.run()


class CTCTFLite(BaseInferencer):
    """
    TFLite Inferencer for CTC Models
    This class provides dynamic length audio recognition
    (expensive cost due to memory allocation everytime call infer function)
    """

    def __init__(self,
                 speech_config: dict,
                 decoder_config: dict,
                 saved_path: str):
        if "2.2" in tf.__version__: raise ImportError("CTCTFLite only works using tf-nightly")
        super(CTCTFLite, self).__init__(saved_path, False)
        self.speech_featurizer = SpeechFeaturizer(speech_config)

    def load_model(self):
        try:
            self.model = tf.lite.Interpreter(self.saved_path)
            print("Ctc dynamic loaded")
        except Exception as e:
            raise Exception(e)

    def compile(self):
        self.load_model()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

    def convert_saved_model_to_tflite(self,
                                      saved_model_path: str,
                                      tflite_path: str):
        print("Loading saved model ...")
        try:
            saved_model = tf.saved_model.load(saved_model_path)
        except Exception as e:
            raise Exception(e)

        print("Converting to tflite ...")

        # See: https://stackoverflow.com/a/55732431/11037553
        # The time dimension can be resize using tf.lite.Interpreter.resize_tensor_input
        concrete_func = saved_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        f, c = self.speech_featurizer.compute_feature_dim()
        concrete_func.inputs[0].set_shape([None, None, f, c])
        # Convert to tflite
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.experimental_new_converter = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()

        print("Writing to file ...")
        if not os.path.exists(os.path.dirname(tflite_path)): os.makedirs(
            os.path.dirname(tflite_path))
        with open(tflite_path, "wb") as tflite_out:
            tflite_out.write(tflite_model)

        print(f"Done converting to {tflite_path}")

    def preprocess(self, audio):
        signal = read_raw_audio(audio, self.speech_featurizer.sample_rate)
        features = self.speech_featurizer.extract(signal)
        return np.expand_dims(features, axis=0)

    def postprocess(self, probs: np.ndarray):
        decoded = self.decoder.decode(probs)
        return bytes_to_string(decoded.numpy())[0]

    def infer(self, audio):
        features = self.preprocess(audio)

        self.model.resize_tensor_input(self.input_details[0]["index"], features.shape)
        self.model.allocate_tensors()

        self.model.set_tensor(self.input_details[0]["index"], features)
        self.model.invoke()
        logits = self.model.get_tensor(self.output_details[0]["index"])

        return self.postprocess(logits)


class CtcStaticTFLite(BaseInferencer):
    """
    TFLite Inferencer for CTC Models
    This class provides static length audio recognition
    The length is specified in compile function (in seconds)
    """

    def __init__(self,
                 speech_config: dict,
                 decoder_config: dict,
                 saved_path: str):
        if "2.2" in tf.__version__: raise ImportError("CTCTFLite only works using tf-nightly")
        super(CtcStaticTFLite, self).__init__(saved_path, False)
        self.speech_featurizer = SpeechFeaturizer(speech_config)

    def load_model(self):
        try:
            self.model = tf.lite.Interpreter(self.saved_path)
            print("Ctc static Loaded")
        except Exception as e:
            raise Exception(e)

    def compile(self, duration: float):
        self.load_model()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        f, c = self.speech_featurizer.compute_feature_dim()
        self.t = self.speech_featurizer.compute_time_dim(duration)
        self.model.resize_tensor_input(self.input_details[0]["index"], [1, self.t, f, c])
        self.model.allocate_tensors()

    def convert_saved_model_to_tflite(self,
                                      saved_model_path: str,
                                      tflite_path: str):
        print("Loading saved model ...")
        try:
            saved_model = tf.saved_model.load(saved_model_path)
        except Exception as e:
            raise Exception(e)

        print("Converting to tflite ...")

        # See: https://stackoverflow.com/a/55732431/11037553
        # The time dimension can be resize using tf.lite.Interpreter.resize_tensor_input
        concrete_func = saved_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        f, c = self.speech_featurizer.compute_feature_dim()
        concrete_func.inputs[0].set_shape([None, None, f, c])
        # Convert to tflite
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.experimental_new_converter = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()

        print("Writing to file ...")
        if not os.path.exists(os.path.dirname(tflite_path)): os.makedirs(
            os.path.dirname(tflite_path))
        with open(tflite_path, "wb") as tflite_out:
            tflite_out.write(tflite_model)

        print(f"Done converting to {tflite_path}")

    def preprocess(self, audio):
        signal = read_raw_audio(audio, self.speech_featurizer.sample_rate)
        features = self.speech_featurizer.extract(signal)
        if features.shape[0] < self.t:
            features = np.pad(features, [[0, int(self.t - features.shape[0])], [0, 0], [0, 0]])
        else:
            features = features[:self.t, :, :]
        features = np.expand_dims(features, axis=0)
        return tf.convert_to_tensor(features)

    def postprocess(self, probs):
        decoded = self.decoder.decode(probs)
        return bytes_to_string(decoded.numpy())[0]

    def infer(self, audio):
        features = self.preprocess(audio)

        self.model.set_tensor(self.input_details[0]["index"], features)
        self.model.invoke()
        logits = self.model.get_tensor(self.output_details[0]["index"])

        return self.postprocess(logits)
