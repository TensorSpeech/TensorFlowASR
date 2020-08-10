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
from tqdm import tqdm
from colorama import Fore

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from ..featurizers.speech_featurizers import deemphasis, tf_merge_slices, read_raw_audio
from ..losses.segan_losses import generator_loss, discriminator_loss
from .base_runners import BaseTrainer, BaseTester
from ..utils.utils import shape_list


class SeganTrainer(BaseTrainer):
    def __init__(self,
                 speech_config: dict,
                 training_config: dict,
                 is_mixed_precision: bool = False,
                 strategy: tf.distribute.Strategy = None):
        self.speech_config = speech_config
        self.is_mixed_precision = is_mixed_precision
        self.deactivate_l1 = False
        self.deactivate_noise = False
        super(SeganTrainer, self).__init__(config=training_config, strategy=strategy)

    def set_train_metrics(self):
        self.train_metrics = {
            "g_l1_loss": tf.keras.metrics.Mean("train_g_l1_loss", dtype=tf.float32),
            "g_adv_loss": tf.keras.metrics.Mean("train_g_adv_loss", dtype=tf.float32),
            "d_adv_loss": tf.keras.metrics.Mean("train_d_adv_loss", dtype=tf.float32)
        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            "g_l1_loss": tf.keras.metrics.Mean("eval_g_l1_loss", dtype=tf.float32),
            "g_adv_loss": tf.keras.metrics.Mean("eval_g_adv_loss", dtype=tf.float32),
            "d_adv_loss": tf.keras.metrics.Mean("eval_d_adv_loss", dtype=tf.float32)
        }

    def save_model_weights(self):
        with self.strategy.scope():
            self.generator.save_weights(os.path.join(self.config["outdir"], "latest.h5"))

    def run(self):
        """Run training."""
        if self.steps.numpy() > 0: tf.print("Resume training ...")

        self.train_progbar = tqdm(
            initial=self.steps.numpy(), unit="batch", total=self.total_train_steps,
            position=0, leave=True,
            bar_format="{desc} |%s{bar:20}%s{r_bar}" % (Fore.GREEN, Fore.RESET),
            desc="[Train]"
        )

        while not self._finished():
            self._train_epoch()
            if self.epochs.numpy() >= self.config["l1_remove_epoch"] \
                    and self.deactivate_l1 is False:
                self.config["l1_lambda"] = 0.
                self.deactivate_l1 = True
            if self.epochs.numpy() >= self.config["denoise_epoch"] \
                    and self.deactivate_noise is False:
                self.config["noise_std"] *= self.config["noise_decay"]
                if self.config["noise_std"] < self.config["denoise_lbound"]:
                    self.config["noise_std"] = 0.
                    self.deactivate_noise = True

        self.save_checkpoint()

        self.train_progbar.close()
        print("> Finish training")

    def create_train_step(self):
        return tf.function(
            self._train_step,
            experimental_relax_shapes=True,
            input_signature=[(
                tf.TensorSpec([None, self.speech_config["window_size"]], dtype=tf.float32),
                tf.TensorSpec([None, self.speech_config["window_size"]], dtype=tf.float32)
            )]
        )

    def _train_step(self, batch):
        clean_wavs, noisy_wavs = batch
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            z = self.generator.get_z(shape_list(clean_wavs)[0])
            g_clean_wavs = self.generator([noisy_wavs, z], training=True)

            d_real_logit = self.discriminator(
                [clean_wavs, noisy_wavs],
                training=True,
                noise_std=self.config["noise_std"]
            )
            d_fake_logit = self.discriminator(
                [g_clean_wavs, noisy_wavs],
                training=True,
                noise_std=self.config["noise_std"]
            )

            gen_tape.watch(g_clean_wavs)
            disc_tape.watch([d_real_logit, d_fake_logit])

            _gen_l1_loss, _gen_adv_loss = generator_loss(y_true=clean_wavs,
                                                         y_pred=g_clean_wavs,
                                                         l1_lambda=self.config["l1_lambda"],
                                                         d_fake_logit=d_fake_logit)

            _disc_loss = discriminator_loss(d_real_logit, d_fake_logit)

            _gen_loss = _gen_l1_loss + _gen_adv_loss

            train_disc_loss = tf.nn.compute_average_loss(
                _disc_loss, global_batch_size=self.global_batch_size)
            train_gen_loss = tf.nn.compute_average_loss(
                _gen_loss, global_batch_size=self.global_batch_size)

            if self.is_mixed_precision:
                scaled_gen_loss = self.generator_optimizer.get_scaled_loss(train_gen_loss)
                scaled_disc_loss = self.discriminator_optimizer.get_scaled_loss(train_disc_loss)

        if self.is_mixed_precision:
            scaled_gen_grad = gen_tape.gradient(
                scaled_gen_loss, self.generator.trainable_variables)
            scaled_disc_grad = disc_tape.gradient(
                scaled_disc_loss, self.discriminator.trainable_variables)
            gradients_of_generator = self.generator_optimizer.get_unscaled_gradients(
                scaled_gen_grad)
            gradients_of_discriminator = self.discriminator_optimizer.get_unscaled_gradients(
                scaled_disc_grad)
        else:
            gradients_of_generator = gen_tape.gradient(
                train_gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(
                train_disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        self.train_metrics["g_l1_loss"].update_state(_gen_l1_loss)
        self.train_metrics["g_adv_loss"].update_state(_gen_adv_loss)
        self.train_metrics["d_adv_loss"].update_state(_disc_loss)

    def create_eval_step(self):
        return tf.function(
            self._eval_step,
            experimental_relax_shapes=True,
            input_signature=[(
                tf.TensorSpec([None, self.speech_config["window_size"]], dtype=tf.float32),
                tf.TensorSpec([None, self.speech_config["window_size"]], dtype=tf.float32)
            )]
        )

    def _eval_step(self, batch):
        clean_wavs, noisy_wavs = batch

        z = self.generator.get_z(shape_list(clean_wavs)[0])
        g_clean_wavs = self.generator([noisy_wavs, z], training=False)

        d_real_logit = self.discriminator([clean_wavs, noisy_wavs], training=False)
        d_fake_logit = self.discriminator([g_clean_wavs, noisy_wavs], training=False)

        _gen_l1_loss, _gen_adv_loss = generator_loss(y_true=clean_wavs,
                                                     y_pred=g_clean_wavs,
                                                     l1_lambda=self.config["l1_lambda"],
                                                     d_fake_logit=d_fake_logit)

        _disc_loss = discriminator_loss(d_real_logit, d_fake_logit)

        self.eval_metrics["g_l1_loss"].update_state(_gen_l1_loss)
        self.eval_metrics["g_adv_loss"].update_state(_gen_adv_loss)
        self.eval_metrics["d_adv_loss"].update_state(_disc_loss)

    def compile(self,
                generator: tf.keras.Model,
                discriminator: tf.keras.Model,
                optimizer_config: dict,
                max_to_keep: int = 10):
        with self.strategy.scope():
            self.generator = generator
            self.discriminator = discriminator
            self.generator_optimizer = tf.keras.optimizers.get(optimizer_config["generator"])
            self.discriminator_optimizer = tf.keras.optimizers.get(
                optimizer_config["discriminator"])
            if self.is_mixed_precision:
                self.generator_optimizer = mixed_precision.LossScaleOptimizer(
                    self.generator_optimizer, "dynamic")
                self.discriminator_optimizer = mixed_precision.LossScaleOptimizer(
                    self.discriminator_optimizer, "dynamic")
        self.create_checkpoint_manager(
            max_to_keep, generator=self.generator, gen_optimizer=self.generator_optimizer,
            discriminator=self.discriminator, disc_optimizer=self.discriminator_optimizer
        )

    def fit(self, train_dataset, eval_dataset=None, eval_train_ratio=1):
        self.set_train_data_loader(train_dataset)
        self.set_eval_data_loader(eval_dataset, eval_train_ratio)
        self.load_checkpoint()
        self.run()


class SeganTester(BaseTester):
    def __init__(self,
                 speech_config: dict,
                 config: dict):
        super(SeganTester, self).__init__(config)
        self.speech_config = speech_config
        try:
            from semetrics import composite
            self.composite = composite
        except ImportError as e:
            print(
                f"Error: {e}\nPlease run ./scripts/install_semetrics.sh")
            return
        self.test_noisy_dir = os.path.join(self.config["outdir"], "test", "noisy")
        self.test_gen_dir = os.path.join(self.config["outdir"], "test", "gen")
        if not os.path.exists(self.test_noisy_dir): os.makedirs(self.test_noisy_dir)
        if not os.path.exists(self.test_gen_dir): os.makedirs(self.test_gen_dir)
        self.test_results = os.path.join(self.config["outdir"], "test", "results.txt")
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

    def set_test_data_loader(self, test_dataset):
        """Set train data loader (MUST)."""
        self.clean_dir = test_dataset.clean_dir
        self.test_data_loader = test_dataset.create()

    def _test_epoch(self):
        if self.processed_records > 0:
            self.test_data_loader = self.test_data_loader.skip(self.processed_records)
        progbar = tqdm(initial=self.processed_records, total=None,
                       unit="batch", position=0, desc="[Test]")
        test_iter = iter(self.test_data_loader)
        while True:
            try:
                self._test_function(test_iter)
            except StopIteration:
                break
            except tf.errors.OutOfRangeError:
                break
            progbar.update(1)

        progbar.close()

    @tf.function(experimental_relax_shapes=True)
    def _test_function(self, iterator):
        batch = next(iterator)
        self._test_step(batch)

    @tf.function(experimental_relax_shapes=True)
    def _test_step(self, batch):
        # Test only available for batch size = 1
        clean_wav_path, noisy_wavs = batch
        g_wavs = self.model([noisy_wavs, self.model.get_z(shape_list(noisy_wavs)[0])],
                            training=False)

        results = tf.numpy_function(
            self._perform, inp=[clean_wav_path, tf_merge_slices(g_wavs),
                                tf_merge_slices(noisy_wavs)],
            Tout=tf.float32
        )

        for idx, key in enumerate(self.test_metrics.keys()):
            self.test_metrics[key].update_state(results[idx])

    def _perform(self,
                 clean_wav_path: bytes,
                 gen_signal: np.ndarray,
                 noisy_signal: np.ndarray) -> tf.Tensor:
        clean_wav_path = clean_wav_path.decode("utf-8")
        results = self._compare(clean_wav_path, gen_signal, noisy_signal)
        return tf.convert_to_tensor(results, dtype=tf.float32)

    def _save_to_outdir(self,
                        clean_wav_path: str,
                        gen_signal: np.ndarray,
                        noisy_signal: np.ndarray):
        gen_path = clean_wav_path.replace(self.clean_dir, self.test_gen_dir)
        noisy_path = clean_wav_path.replace(self.clean_dir, self.test_noisy_dir)
        try:
            os.makedirs(os.path.dirname(gen_path))
            os.makedirs(os.path.dirname(noisy_path))
        except Exception:
            pass
        # Avoid differences by writing original wav using sf
        clean_wav = read_raw_audio(clean_wav_path, self.speech_config["sample_rate"])
        sf.write("/tmp/clean.wav", clean_wav, self.speech_config["sample_rate"])
        sf.write(gen_path,
                 gen_signal,
                 self.speech_config["sample_rate"])
        sf.write(noisy_path,
                 noisy_signal,
                 self.speech_config["sample_rate"])
        return gen_path, noisy_path

    def _compare(self,
                 clean_wav_path: str,
                 gen_signal: np.ndarray,
                 noisy_signal: np.ndarray) -> list:
        gen_signal = deemphasis(gen_signal, self.speech_config["preemphasis"])
        noisy_signal = deemphasis(noisy_signal, self.speech_config["preemphasis"])

        gen_path, noisy_path = self._save_to_outdir(clean_wav_path, gen_signal, noisy_signal)

        (pesq_gen, csig_gen, cbak_gen,
         covl_gen, ssnr_gen) = self.composite("/tmp/clean.wav", gen_path)
        (pesq_noisy, csig_noisy, cbak_noisy,
         covl_noisy, ssnr_noisy) = self.composite("/tmp/clean.wav", noisy_path)

        return [pesq_gen, csig_gen, cbak_gen, covl_gen, ssnr_gen,
                pesq_noisy, csig_noisy, cbak_noisy, covl_noisy, ssnr_noisy]

    def finish(self):
        with open(self.test_results, "w", encoding="utf-8") as out:
            for idx, key in enumerate(self.test_metrics.keys()):
                out.write(f"{key} = {self.test_metrics[key].result().numpy():.2f}\n")
