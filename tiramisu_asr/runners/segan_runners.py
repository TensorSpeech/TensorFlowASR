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
import soundfile as sf
import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from ..featurizers.speech_featurizers import deemphasis, read_raw_audio
from ..losses.segan_losses import generator_loss, discriminator_loss
from ..models.segan import Discriminator, Generator, make_z_as_input
from .base_runners import BaseTrainer, BaseTester, BaseInferencer
from ..utils.utils import slice_signal, merge_slices, print_test_info, shape_list


class SeganTrainer(BaseTrainer):
    def __init__(self,
                 speech_config: dict,
                 training_config: dict,
                 is_mixed_precision: bool = False,
                 strategy: tf.distribute.Strategy = None):
        super(SeganTrainer, self).__init__(config=training_config, strategy=strategy)
        self.speech_config = speech_config
        self.is_mixed_precision = is_mixed_precision

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

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        clean_wavs, noisy_wavs = batch
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            z = self.generator.get_z(shape_list(clean_wavs)[0])
            g_clean_wavs = self.generator([noisy_wavs, z], training=True)

            d_real_logit = self.discriminator([clean_wavs, noisy_wavs], training=True)
            d_fake_logit = self.discriminator([g_clean_wavs, noisy_wavs], training=True)

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

    @tf.function(experimental_relax_shapes=True)
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

    def compile(self, model_config: dict, optimizer_config: dict, max_to_keep: int = 10):
        with self.strategy.scope():
            self.generator = Generator(
                g_enc_depths=model_config["g_enc_depths"],
                window_size=self.speech_config["window_size"],
                kwidth=model_config["kwidth"], ratio=model_config["ratio"]
            )
            self.generator._build()
            self.generator.summary(line_length=100)
            self.generator_optimizer = tf.keras.optimizers.get(optimizer_config["generator"])
            self.discriminator = Discriminator(
                d_num_fmaps=model_config["d_num_fmaps"],
                window_size=self.speech_config["window_size"],
                kwidth=model_config["kwidth"], ratio=model_config["ratio"]
            )
            self.discriminator._build()
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
                 config: dict,
                 saved_path: str,
                 from_weights: bool = False):
        super(SeganTester, self).__init__(config, saved_path, from_weights)
        self.speech_config = speech_config
        try:
            from semetrics import composite
            self.composite = composite
        except ImportError as e:
            print(
                f"Error: {e}\nPlease run ./scripts/install_semetrics.sh")
            return
        self.test_noisy = os.path.join(self.config["outdir"], "test", "noisy")
        self.test_gen = os.path.join(self.config["outdir"], "test", "gen")
        if not os.path.exists(self.test_noisy):
            os.makedirs(self.test_noisy)
        if not os.path.exists(self.test_gen):
            os.makedirs(self.test_gen)
        self.test_file = os.path.join(self.config["outdir"], "test", "results.txt")
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

    def compile(self, model_config: dict):
        self.model = Generator(
            g_enc_depths=model_config["g_enc_depths"],
            window_size=self.speech_config["window_size"],
            kwidth=model_config["kwidth"], ratio=model_config["ratio"]
        )
        self.model._build()
        if self.from_weights:
            self.load_model_from_weights()
        else:
            self.load_model()

    def _get_metrics(self):
        return "PESQ = ", self.test_metrics['g_pesq'].result(), \
               ", CSIG = ", self.test_metrics['g_csig'].result(), \
               ", CBAK = ", self.test_metrics['g_cbak'].result(), \
               ", COVL = ", self.test_metrics['g_covl'].result(), \
               ", SSNR = ", self.test_metrics['g_ssnr'].result(),

    @tf.function
    def _test_epoch(self):
        for idx, batch in self.test_data_loader.enumerate(start=1):
            self._test_step(batch)
            print_test_info(*self._get_metrics(), batches=idx)

    @tf.function
    def _test_step(self, batch):
        # Test only available for batch size = 1
        name, clean_wavs, noisy_wavs = batch
        g_wavs = self.model(noisy_wavs, training=False)

        results = tf.numpy_function(
            self._perform, inp=[clean_wavs, merge_slices(
                g_wavs), merge_slices(noisy_wavs), name],
            Tout=tf.float32
        )

        for idx, key in enumerate(self.test_metrics.keys()):
            self.test_metrics[key].update_state(results[idx])

    def _get_save_file_paths(self, name: str):
        return os.path.join(self.test_gen, name), os.path.join(self.test_noisy, name)

    def _save_to_tmp(self,
                     clean_signal: np.ndarray,
                     gen_signal: np.ndarray,
                     noisy_signal: np.ndarray,
                     name: str):
        gen_path, noisy_path = self._get_save_file_paths(name)
        sf.write("/tmp/clean_signal.wav", clean_signal, self.speech_config["sample_rate"])
        sf.write(gen_path, gen_signal, self.speech_config["sample_rate"])
        sf.write(noisy_path, noisy_signal, self.speech_config["sample_rate"])

    def _perform(self,
                 clean_batch: np.ndarray,
                 gen_batch: np.ndarray,
                 noisy_batch: np.ndarray,
                 name: bytes) -> tf.Tensor:
        name = name.decode("utf-8")
        results = self._compare([clean_batch, gen_batch, noisy_batch], name)
        return tf.convert_to_tensor(results, dtype=tf.float32)

    def _compare(self, data_slice: list, name: str) -> list:
        clean_slice, g_slice, n_slice = data_slice
        clean_slice = deemphasis(clean_slice, self.speech_config["preemphasis"])
        g_slice = deemphasis(g_slice, self.speech_config["preemphasis"])
        n_slice = deemphasis(n_slice, self.speech_config["preemphasis"])

        self._save_to_tmp(clean_slice, g_slice, n_slice, name)
        gen_path, noisy_path = self._get_save_file_paths(name)

        pesq_gen, csig_gen, cbak_gen, covl_gen, ssnr_gen = self.composite(
            "/tmp/clean_signal.wav", gen_path)
        pesq_noisy, csig_noisy, cbak_noisy, covl_noisy, ssnr_noisy = self.composite(
            "/tmp/clean_signal.wav", noisy_path)

        return [pesq_gen, csig_gen, cbak_gen, covl_gen, ssnr_gen,
                pesq_noisy, csig_noisy, cbak_noisy, covl_noisy, ssnr_noisy]

    def finish(self):
        with open(self.test_file, "w", encoding="utf-8") as out:
            for idx, key in enumerate(self.test_metrics.keys()):
                out.write(f"{key} = {self.test_metrics[key].result().numpy():.2f}\n")


class SeganInferencer(BaseInferencer):
    def __init__(self,
                 speech_config: dict,
                 model_config: dict,
                 saved_path: str,
                 from_weights: bool = False):
        super(SeganInferencer, self).__init__(saved_path, from_weights)
        self.speech_config = speech_config
        self.model_config = model_config

    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(self.saved_path)
            print("Model loaded")
        except Exception as e:
            raise Exception(e)

    def compile(self, *args, **kwargs):
        if self.from_weights:
            self.model = Generator(
                g_enc_depths=self.model_config["g_enc_depths"],
                window_size=self.speech_config["window_size"],
                kwidth=self.model_config["kwidth"], ratio=self.model_config["ratio"]
            )
            self.model._build()
            self.load_model_from_weights()
        else:
            self.load_model()

    def preprocess(self, audio):
        signal = read_raw_audio(audio, self.speech_config["sample_rate"])
        return (
            slice_signal(signal, self.speech_config["window_size"], stride=1),
            signal.shape[0]
        )

    def postprocess(self, signal: np.ndarray):
        return deemphasis(signal, self.speech_config["preemphasis"])

    def infer(self, audio):
        sliced_signal, original_length = self.preprocess(audio)

        @tf.function
        def gen(slices):
            slices = tf.reshape(slices, [-1, self.speech_config["window_size"]])
            g_wavs = self.model(slices, training=False)
            return merge_slices(g_wavs)

        output = gen(sliced_signal).numpy()
        output = output[:original_length + 1]

        return self.postprocess(output)


class SeganTFLite(BaseInferencer):
    def __init__(self,
                 speech_config: dict,
                 saved_path: str):
        super(SeganTFLite, self).__init__(saved_path, False)
        self.speech_config = speech_config

    def load_model(self):
        try:
            self.model = tf.lite.Interpreter(self.saved_path)
            self.model.allocate_tensors()
            print("Segan loaded")
        except Exception as e:
            raise Exception(e)

    @staticmethod
    def convert_saved_model_to_tflite(saved_model_path: str,
                                      model_config: dict,
                                      speech_config: dict,
                                      tflite_path: str):
        print("Loading saved model ...")
        try:
            saved_model = tf.keras.models.load_model(saved_model_path)
        except Exception as e:
            raise Exception(e)
        saved_model = make_z_as_input(saved_model, model_config, speech_config)

        print("Converting to tflite ...")
        converter = tf.lite.TFLiteConverter.from_keras_model(saved_model)
        converter.experimental_new_converter = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        print("Writing to file ...")
        if not os.path.exists(os.path.dirname(tflite_path)):
            os.makedirs(os.path.dirname(tflite_path))
        with open(tflite_path, "wb") as tflite_out:
            tflite_out.write(tflite_model)

        print(f"Done converting to {tflite_path}")

    def compile(self):
        self.load_model()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

    def preprocess(self, audio):
        signal = read_raw_audio(audio, self.speech_config["sample_rate"])
        slices = slice_signal(signal, self.speech_config["window_size"], stride=1)
        slices = np.reshape(slices, [-1, 1, self.speech_config["window_size"]])
        return slices, signal.shape[0]

    def postprocess(self, signal: np.ndarray):
        return deemphasis(signal, self.speech_config["preemphasis"])

    def infer(self, audio):
        slices, original_length = self.preprocess(audio)
        g_wavs = np.array([])

        for idx, s in enumerate(slices):
            # audio shape [1, window_size]
            self.model.set_tensor(self.input_details[0]['index'], s)

            # z shape [1, last_enc, 1, last_enc_channel]
            z = np.random.normal(0., 1., self.input_details[1]['shape']).astype(np.float32)
            self.model.set_tensor(self.input_details[1]['index'], z)

            self.model.invoke()

            output = self.model.get_tensor(self.output_details[0]['index'])
            g_wavs = np.concatenate([g_wavs, output[0]])

        g_wavs = g_wavs[:original_length + 1]

        return self.postprocess(g_wavs)
