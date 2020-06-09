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
import multiprocessing

import numpy as np
import soundfile as sf
import tensorflow as tf

from featurizers.speech_featurizers import deemphasis, read_raw_audio
from losses.segan_losses import generator_loss, discriminator_loss
from models.segan.Discriminator import create_discriminator
from models.segan.Generator import create_generator
from runners.base_runners import BaseTrainer, BaseTester, BaseInferencer
from utils.utils import slice_signal, merge_slices


class SeganTrainer(BaseTrainer):
    def __init__(self,
                 speech_config: dict,
                 training_config: dict,
                 is_mixed_precision: bool = False):
        super(SeganTrainer, self).__init__(config=training_config)
        self.speech_config = speech_config
        self.is_mixed_precision = is_mixed_precision

        self.train_metrics = {
            "g_l1_loss":  tf.keras.metrics.Mean("segan_g_l1_loss", dtype=tf.float32),
            "g_adv_loss": tf.keras.metrics.Mean("segan_g_adv_loss", dtype=tf.float32),
            "d_adv_loss": tf.keras.metrics.Mean("segan_d_adv_loss", dtype=tf.float32)
        }

    @tf.function
    def _train_step(self, batch):
        clean_wavs, noisy_wavs = batch
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            g_clean_wavs = self.generator(noisy_wavs, training=True)

            d_real_logit = self.discriminator({
                "clean": clean_wavs,
                "noisy": noisy_wavs,
            }, training=True)
            d_fake_logit = self.discriminator({
                "clean": g_clean_wavs,
                "noisy": noisy_wavs,
            }, training=True)

            _gen_l1_loss, _gen_adv_loss = generator_loss(y_true=clean_wavs,
                                                         y_pred=g_clean_wavs,
                                                         l1_lambda=self.config["l1_lambda"],
                                                         d_fake_logit=d_fake_logit)

            _disc_loss = discriminator_loss(d_real_logit, d_fake_logit)

            _gen_loss = _gen_l1_loss + _gen_adv_loss

            if self.is_mixed_precision:
                scaled_gen_loss = self.generator_optimizer.get_scaled_loss(_gen_loss)
                scaled_disc_loss = self.discriminator_optimizer.get_scaled_loss(_disc_loss)

        if self.is_mixed_precision:
            scaled_gen_grad = gen_tape.gradient(scaled_gen_loss, self.generator.trainable_variables)
            scaled_disc_grad = disc_tape.gradient(scaled_disc_loss, self.discriminator.trainable_variables)
            gradients_of_generator = self.generator_optimizer.get_unscaled_gradients(scaled_gen_grad)
            gradients_of_discriminator = self.discriminator_optimizer.get_unscaled_gradients(scaled_disc_grad)
        else:
            gradients_of_generator = gen_tape.gradient(_gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(_disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        self.train_metrics["g_l1_loss"].update_state(_gen_l1_loss)
        self.train_metrics["g_adv_loss"].update_state(_gen_adv_loss)
        self.train_metrics["d_adv_loss"].update_state(_disc_loss)

        tf.py_function(lambda: self.tqdm.set_postfix_str(
            f"g_l1_loss = {self.train_metrics['g_l1_loss'].result().numpy():.4f}, "
            f"g_adv_loss = {self.train_metrics['g_adv_loss'].result().numpy():.4f}, "
            f"d_loss = {self.train_metrics['d_adv_loss'].result().numpy():.4f}"
        ), [], [])

    def _eval_epoch(self):
        pass

    def _eval_step(self, batch):
        pass

    def _exec_log_interval(self):
        self._write_to_tensorboard(self.train_metrics, self.steps, stage="train")
        """Reset train metrics after save it to tensorboard."""
        for metric in self.train_metrics.keys():
            self.train_metrics[metric].reset_states()

    def _save_model_architecture(self):
        with open(os.path.join(self.config["outdir"], "generator.yaml"), "w") as f:
            f.write(self.generator.to_yaml())

    def compile(self, model_config: dict, optimizer_config: dict):
        self.generator = create_generator(g_enc_depths=model_config["g_enc_depths"],
                                          window_size=self.speech_config["window_size"],
                                          kwidth=model_config["kwidth"], ratio=model_config["ratio"])
        self.discriminator = create_discriminator(d_num_fmaps=model_config["d_num_fmaps"],
                                                  window_size=self.speech_config["window_size"],
                                                  kwidth=model_config["kwidth"], ratio=model_config["ratio"])
        print(self.generator.summary())
        self._save_model_architecture()
        self.generator_optimizer = tf.keras.optimizers.get(optimizer_config["generator"])
        self.discriminator_optimizer = tf.keras.optimizers.get(optimizer_config["discriminator"])
        if self.is_mixed_precision:
            self.generator_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(self.generator_optimizer, "dynamic")
            self.discriminator_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(self.discriminator_optimizer, "dynamic")

    def fit(self, train_dataset, max_to_keep=10):
        self.set_train_data_loader(train_dataset)
        self.create_checkpoint_manager(max_to_keep, generator=self.generator, gen_optimizer=self.generator_optimizer,
                                       discriminator=self.discriminator, disc_optimizer=self.discriminator_optimizer)
        self.load_checkpoint()
        self.run()


class SeganTester(BaseTester):
    def __init__(self,
                 speech_config: dict,
                 config: dict,
                 saved_path: str,
                 from_weights: bool = False):
        super(SeganTester, self).__init__(config, saved_path, None, from_weights)
        self.speech_config = speech_config
        try:
            from externals.semetrics.main import composite
            self.composite = composite
        except ImportError as e:
            print(f"Error: {e}\nPlease install https://github.com/usimarit/semetrics using ./setup.sh semetrics")
            return
        self.test_metrics = {
            "g_pesq": tf.keras.metrics.Mean("test_pesq", dtype=tf.float32),
            "g_csig": tf.keras.metrics.Mean("test_csig", dtype=tf.float32),
            "g_cbak": tf.keras.metrics.Mean("test_cbak", dtype=tf.float32),
            "g_covl": tf.keras.metrics.Mean("test_covl", dtype=tf.float32),
            "g_ssnr": tf.keras.metrics.Mean("test_ssnr", dtype=tf.float32),
            "n_pesq": tf.keras.metrics.Mean("test_noise_pesq", dtype=tf.float32),
            "n_csig": tf.keras.metrics.Mean("test_noise_csig", dtype=tf.float32),
            "n_cbak": tf.keras.metrics.Mean("test_noise_cbak", dtype=tf.float32),
            "n_covl": tf.keras.metrics.Mean("test_noise_covl", dtype=tf.float32),
            "n_ssnr": tf.keras.metrics.Mean("test_noise_ssnr", dtype=tf.float32)
        }

    def load_model_from_weights(self):
        try:
            self.model.load_weights(self.saved_path)
        except Exception as e:
            raise Exception(e)

    def compile(self, model_config: dict):
        self.model = create_generator(g_enc_depths=model_config["g_enc_depths"],
                                      window_size=self.speech_config["window_size"],
                                      kwidth=model_config["kwidth"], ratio=model_config["ratio"])
        if self.from_weights:
            self.load_model_from_weights()
        else:
            self.load_model()

    def _get_metrics(self):
        return f"pesq = {self.test_metrics['g_pesq'].result():.4f}, " \
               f"csig = {self.test_metrics['g_csig'].result():.4f}, " \
               f"cbak = {self.test_metrics['g_cbak'].result():.4f}, " \
               f"covl = {self.test_metrics['g_covl'].result():.4f}, " \
               f"ssnr = {self.test_metrics['g_ssnr'].result():.4f}"

    def _log_test(self, step):
        self._write_to_tensorboard(self.test_metrics, step, stage="test")
        tf.py_function(lambda: self.tqdm.set_postfix_str(self._get_metrics()), [], [])

    @tf.function
    def _test_step(self, batch):
        # Test only available for batch size = 1
        clean_wavs, noisy_wavs = batch
        g_wavs = self.model(noisy_wavs, training=False)

        results = tf.py_function(
            self._perform, inp=[clean_wavs, merge_slices(g_wavs), merge_slices(noisy_wavs)],
            Tout=tf.float32
        )

        for idx, key in enumerate(self.test_metrics.keys()):
            self.test_metrics[key].update_state(results[idx])

    def save_to_tmp(self, clean_signal, gen_signal, noisy_signal):
        sf.write("/tmp/clean_signal.wav", clean_signal, self.speech_config["sample_rate"])
        sf.write("/tmp/gen_signal.wav", gen_signal, self.speech_config["sample_rate"])
        sf.write("/tmp/noisy_signal.wav", noisy_signal, self.speech_config["sample_rate"])

    def _perform(self, clean_batch: tf.Tensor, gen_batch: tf.Tensor, noisy_batch: tf.Tensor) -> tf.Tensor:
        results = self._compare([clean_batch.numpy(), gen_batch.numpy(), noisy_batch.numpy()])
        return tf.convert_to_tensor(results, dtype=tf.float32)

    def _compare(self, data_slice) -> list:
        clean_slice, g_slice, n_slice = data_slice
        clean_slice = deemphasis(clean_slice, self.speech_config["preemphasis"])
        g_slice = deemphasis(g_slice, self.speech_config["preemphasis"])
        n_slice = deemphasis(n_slice, self.speech_config["preemphasis"])
        self.save_to_tmp(clean_slice, g_slice, n_slice)

        pesq_gen, csig_gen, cbak_gen, covl_gen, ssnr_gen = self.composite("/tmp/clean_signal.wav", "/tmp/gen_signal.wav")
        pesq_noisy, csig_noisy, cbak_noisy, covl_noisy, ssnr_noisy = self.composite("/tmp/clean_signal.wav", "/tmp/noisy_signal.wav")

        return [pesq_gen, csig_gen, cbak_gen, covl_gen, ssnr_gen, pesq_noisy, csig_noisy, cbak_noisy, covl_noisy, ssnr_noisy]

    def finish(self):
        print(f"Test results: {self._get_metrics()}")


class SeganInferencer(BaseInferencer):
    def __init__(self,
                 speech_config: dict,
                 saved_path: str,
                 yaml_arch_path: str,
                 from_weights: bool = False):
        super(SeganInferencer, self).__init__(saved_path, yaml_arch_path, from_weights)
        self.speech_config = speech_config

    def preprocess(self, audio):
        signal = read_raw_audio(audio, self.speech_config["sample_rate"])
        return slice_signal(signal, self.speech_config["window_size"], stride=1)

    def postprocess(self, signal):
        return deemphasis(signal.numpy(), self.speech_config["preemphasis"])

    def infer(self, audio):
        slice_signal = self.preprocess(audio)

        @tf.function
        def gen(sliced_signal):
            sliced_signal = tf.reshape(sliced_signal, [-1, self.speech_config["window_size"]])
            g_wavs = self.model(sliced_signal, training=False)
            return merge_slices(g_wavs)

        return self.postprocess(gen(slice_signal))
