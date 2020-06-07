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

import logging
import os

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

        gradients_of_generator = gen_tape.gradient(_gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(_disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        self.train_metrics["g_l1_loss"].update_state(_gen_l1_loss)
        self.train_metrics["g_adv_loss"].update_state(_gen_adv_loss)
        self.train_metrics["d_adv_loss"].update_state(_disc_loss)

    def _post_train_step(self):
        self.tqdm.set_postfix_str(f"train_g_l1_loss = {self.train_metrics['g_l1_loss'].result():.4f}, "
                                  f"train_g_adv_loss = {self.train_metrics['g_adv_loss'].result():.4f}, "
                                  f"train_d_loss = {self.train_metrics['d_adv_loss'].result():.4f}")

    def _eval_epoch(self):
        pass

    def _eval_step(self, batch):
        pass

    def _check_log_interval(self):
        if (self.steps % self.config["log_interval_steps"] == 0) or self.finish_training:
            self._write_to_tensorboard(self.train_metrics, self.steps, stage="train")

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
        logging.info(self.generator.summary())
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
                 yaml_arch_path: str,
                 from_weights: bool = False):
        super(SeganTester, self).__init__(config, saved_path, yaml_arch_path, from_weights)
        self.speech_config = speech_config
        try:
            from semetrics.main import pesq_mos as pesq, composite
            self.pesq = pesq
            self.composite = composite
        except ImportError as e:
            print(f"Error: {e}\nPlease install https://github.com/usimarit/semetrics")
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

    def _get_metrics(self):
        return (f"pesq = {self.test_metrics['pesq']:.4f}, ",
                f"csig = {self.test_metrics['csig']:.4f}, ",
                f"cbak = {self.test_metrics['cbak']:.4f}, ",
                f"covl = {self.test_metrics['covl']:.4f}, ",
                f"ssnr = {self.test_metrics['ssnr']:.4f}")

    def _post_process_step(self):
        self._write_to_tensorboard(self.test_metrics, self.test_steps_per_epoch, stage="test")
        self.test_data_loader.set_postfix_str(self._get_metrics())

    @tf.function
    def _test_step(self, batch):
        clean_wavs, noisy_wavs = batch
        g_wavs = self.model(noisy_wavs, training=False)
        (pesq_gen, csig_gen, cbak_gen, covl_gen, ssnr_gen,
         pesq_noisy, csig_noisy, cbak_noisy, covl_noisy, ssnr_noisy) = tf.py_function(
            self._compare, inp=[clean_wavs, g_wavs, noisy_wavs],
            Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                  tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        )

        self.test_metrics["g_pesq"].update_state(pesq_gen)
        self.test_metrics["g_csig"].update_state(csig_gen)
        self.test_metrics["g_cbak"].update_state(cbak_gen)
        self.test_metrics["g_covl"].update_state(covl_gen)
        self.test_metrics["g_ssnr"].update_state(ssnr_gen)
        self.test_metrics["n_pesq"].update_state(pesq_noisy)
        self.test_metrics["n_csig"].update_state(csig_noisy)
        self.test_metrics["n_cbak"].update_state(cbak_noisy)
        self.test_metrics["n_covl"].update_state(covl_noisy)
        self.test_metrics["n_ssnr"].update_state(ssnr_noisy)

    def save_to_tmp(self, clean_signal, gen_signal, noisy_signal):
        sf.write("/tmp/clean_signal.wav", clean_signal, self.speech_config["sample_rate"])
        sf.write("/tmp/gen_signal.wav", gen_signal, self.speech_config["sample_rate"])
        sf.write("/tmp/noisy_signal.wav", noisy_signal, self.speech_config["sample_rate"])

    def _compare(self, clean_wavs, g_wavs, noisy_wavs):
        g_wavs = g_wavs.numpy()
        clean_wavs = clean_wavs.numpy()
        noisy_wavs = noisy_wavs
        length = len(g_wavs)

        pesq_gen, pesq_noisy = (0., 0.)
        csig_gen, cbak_gen, covl_gen, ssnr_gen = (0., 0., 0., 0.)
        csig_noisy, cbak_noisy, covl_noisy, ssnr_noisy = (0., 0., 0., 0.)

        for clean_slice, g_slice, n_slice in zip(clean_wavs, g_wavs, noisy_wavs):
            g_slice = deemphasis(g_slice, self.speech_config["preemphasis"])
            clean_slice = deemphasis(clean_slice, self.speech_config["preemphasis"])
            n_slice = deemphasis(n_slice, self.speech_config["preemphasis"])
            self.save_to_tmp(clean_slice, g_slice, n_slice)
            pesq_gen += self.pesq("/tmp/clean_signal.wav", "/tmp/gen_signal.wav")
            pesq_noisy += self.pesq("/tmp/clean_signal.wav", "/tmp/noisy_signal.wav")
            _csig_gen, _cbak_gen, _covl_gen, _ssnr_gen = self.composite("/tmp/clean_signal.wav", "/tmp/gen_signal.wav")
            csig_gen += _csig_gen
            cbak_gen += _cbak_gen
            covl_gen += _covl_gen
            ssnr_gen += _ssnr_gen
            _csig_noisy, _cbak_noisy, _covl_noisy, _ssnr_noisy = self.composite("/tmp/clean_signal.wav", "/tmp/noisy_signal.wav")
            csig_noisy += _csig_noisy
            cbak_noisy += _cbak_noisy
            covl_noisy += _covl_noisy
            ssnr_noisy += _ssnr_noisy

        pesq_gen /= length
        pesq_noisy /= length
        csig_gen /= length
        cbak_gen /= length
        covl_gen /= length
        ssnr_gen /= length
        csig_noisy /= length
        cbak_noisy /= length
        covl_noisy /= length
        ssnr_noisy /= length
        return pesq_gen, csig_gen, cbak_gen, covl_gen, ssnr_gen, pesq_noisy, csig_noisy, cbak_noisy, covl_noisy, ssnr_noisy

    def finish(self):
        logging.info(f"Test results: {self._get_metrics()}")


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
