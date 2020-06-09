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
from __future__ import absolute_import

import os

import tensorflow as tf

from decoders.ctc_decoders import CTCDecoder
from featurizers.speech_featurizers import SpeechFeaturizer, read_raw_audio
from featurizers.text_featurizers import TextFeaturizer
from losses.ctc_losses import ctc_loss
from models.ctc_models import create_ctc_model
from runners.base_runners import BaseTrainer, BaseTester, BaseInferencer
from utils.metrics import ErrorRate, wer, cer
from utils.utils import bytes_to_string, print_one_line


class CTCTrainer(BaseTrainer):
    """ Trainer for CTC Models """

    def __init__(self,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 decoder: CTCDecoder,
                 config: dict,
                 is_mixed_precision: bool = False):
        super(CTCTrainer, self).__init__(config)
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer
        self.decoder = decoder
        self.is_mixed_precision = is_mixed_precision

        self.train_metrics = {
            "ctc_loss": tf.keras.metrics.Mean("train_ctc_loss", dtype=tf.float32)
        }
        self.eval_metrics = {
            "ctc_loss": tf.keras.metrics.Mean("eval_ctc_loss", dtype=tf.float32),
            "wer":      ErrorRate(func=wer, name="eval_word_error_rate", dtype=tf.float32),
            "cer":      ErrorRate(func=cer, name="eval_char_error_rate", dtype=tf.float32)
        }

    @tf.function
    def _train_step(self, batch):
        features, input_length, labels, label_length = batch

        with tf.GradientTape() as tape:
            y_pred = self.model(features, training=True)
            train_loss = ctc_loss(y_true=labels, y_pred=y_pred,
                                  input_length=input_length, label_length=label_length,
                                  num_classes=self.text_featurizer.num_classes)

            if self.is_mixed_precision:
                scaled_train_loss = self.optimizer.get_scaled_loss(train_loss)

        if self.is_mixed_precision:
            scaled_gradients = tape.gradient(scaled_train_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_metrics["ctc_loss"].update_state(train_loss)

        tf.py_function(lambda: self.tqdm.set_postfix_str(f"train_ctc_loss = {self.train_metrics['ctc_loss'].result().numpy():.4f}"), [], [])

    @tf.function
    def _eval_epoch(self):
        if not self.eval_data_loader: return

        tf.print("Evaluating ...")

        for eval_batch, batch in self.eval_data_loader.enumerate(start=1):
            self._eval_step(batch)
            print_one_line("Eval batch:", eval_batch, ", ctc_loss =", self.eval_metrics["ctc_loss"].result())

        tf.print("Finished evaluation at step", self.steps,
                 "gives eval_ctc_loss =", self.eval_metrics["ctc_loss"].result(),
                 "eval_wer =", self.eval_metrics["wer"].result(),
                 "eval_cer =", self.eval_metrics["cer"].result())
        # Write to tensorboard
        self._write_to_tensorboard(self.eval_metrics, self.steps, stage="eval")
        # Reset
        """Reset eval metrics after save it to tensorboard."""
        for metric in self.eval_metrics.keys():
            self.eval_metrics[metric].reset_states()

    @tf.function
    def _eval_step(self, batch):
        features, input_length, labels, label_length = batch
        logits = self.model(features, training=False)
        eval_loss = ctc_loss(y_true=labels, y_pred=logits,
                             input_length=input_length, label_length=label_length,
                             num_classes=self.text_featurizer.num_classes)
        pred = self.decoder.decode(probs=logits, input_length=input_length)
        labels = self.decoder.convert_to_string(labels)

        # Update metrics
        self.eval_metrics["ctc_loss"].update_state(eval_loss)
        self.eval_metrics["wer"].update_state(pred, labels)
        self.eval_metrics["cer"].update_state(pred, labels)

    def _exec_log_interval(self):
        self._write_to_tensorboard(self.train_metrics, self.steps, stage="train")
        """Reset train metrics after save it to tensorboard."""
        for metric in self.train_metrics.keys():
            self.train_metrics[metric].reset_states()

    def _save_model_architecture(self):
        with open(os.path.join(self.config["outdir"], "model.yaml"), "w") as f:
            f.write(self.model.to_yaml())

    def compile(self, model_config: dict, optimizer_config: dict):
        self.model = create_ctc_model(model_config, self.speech_featurizer, self.text_featurizer)
        tf.print(self.model.summary())
        self._save_model_architecture()
        self.optimizer = tf.keras.optimizers.get(optimizer_config)
        if self.is_mixed_precision:
            self.optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(self.optimizer, "dynamic")

    def fit(self, train_dataset, eval_dataset, max_to_keep=10):
        self.set_train_data_loader(train_dataset)
        self.set_eval_data_loader(eval_dataset)
        self.create_checkpoint_manager(max_to_keep, model=self.model, optimizer=self.optimizer)
        self.load_checkpoint()
        self.run()


class CTCTester(BaseTester):
    """ Tester for CTC Models """

    def __init__(self,
                 text_featurizer: TextFeaturizer,
                 decoder: CTCDecoder,
                 config: dict,
                 saved_path: str,
                 yaml_arch_path: str,
                 from_weights: bool = False):
        super(CTCTester, self).__init__(config, saved_path, yaml_arch_path, from_weights)
        self.text_featurizer = text_featurizer
        self.decoder = decoder
        self.test_metrics = {
            "ctc_loss": tf.keras.metrics.Mean("test_ctc_loss", dtype=tf.float32),
            "wer":      ErrorRate(func=wer, name="test_word_error_rate", dtype=tf.float32),
            "cer":      ErrorRate(func=cer, name="test_char_error_rate", dtype=tf.float32)
        }

    def _get_metrics(self):
        return f"test_ctc_loss = {self.test_metrics['ctc_loss'].result():.4f}, " \
               f"test_wer = {self.test_metrics['wer'].result():.4f}%, " \
               f"test_cer = {self.test_metrics['cer'].result():.4f}%"

    def _log_test(self, step):
        self._write_to_tensorboard(self.test_metrics, step, stage="test_" + self.decoder.name)
        tf.py_function(lambda: self.tqdm.set_postfix_str(self._get_metrics()), [], [])

    @tf.function
    def _test_step(self, batch):
        features, input_length, labels, label_length = batch
        logits = self.model(features, training=False)
        eval_loss = ctc_loss(y_true=labels, y_pred=logits,
                             input_length=input_length, label_length=label_length,
                             num_classes=self.text_featurizer.num_classes)
        pred = self.decoder.decode(probs=logits, input_length=input_length)
        labels = self.decoder.convert_to_string(labels)

        # Update metrics
        self.test_metrics["ctc_loss"].update_state(eval_loss)
        self.test_metrics["wer"].update_state(pred, labels)
        self.test_metrics["cer"].update_state(pred, labels)

    def finish(self):
        print(f"Test results: {self._get_metrics()}")


class CTCInferencer(BaseInferencer):
    """ Inferencer for CTC Models """

    def __init__(self,
                 config: dict,
                 saved_path: str,
                 yaml_arch_path: str,
                 from_weights: bool = False):
        super(CTCInferencer, self).__init__(saved_path, yaml_arch_path, from_weights)
        self.speech_featurizer = SpeechFeaturizer(config["speech_config"])
        self.decoder = CTCDecoder(config["decoder_config"], TextFeaturizer(config["vocabulary_file_path"]))

    def preprocess(self, audio):
        signal = read_raw_audio(audio, self.speech_featurizer.sample_rate)
        features = self.speech_featurizer.extract(signal)
        input_length = tf.cast(tf.shape(features)[0], tf.int32)
        return tf.expand_dims(features, axis=0), tf.expand_dims(input_length, axis=0)

    def postprocess(self, probs, input_length):
        decoded = self.decoder.decode(probs, input_length)
        return bytes_to_string(decoded.numpy())[0]

    def infer(self, audio):
        features, input_length = self.preprocess(audio)

        @tf.function
        def predict(features):
            return self.model(features, training=False)

        return self.postprocess(predict(features), input_length)
