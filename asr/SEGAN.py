from __future__ import absolute_import

import time
import os
import sys
import pathlib
import tensorflow as tf
from models.segan.Discriminator import create_discriminator, discriminator_loss
from models.segan.Generator import create_generator, generator_loss
from utils.Utils import get_segan_config, slice_signal, merge_slices
from utils.Metrics import pesq, csig, cbak, covl, ssnr
from featurizers.SpeechFeaturizer import deemphasis, preemphasis
from data.SeganDataset import SeganDataset


class SEGAN:
    def __init__(self, config_path, training=True):
        self.g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.d_num_fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]

        self.configs = get_segan_config(config_path)

        self.kwidth = self.configs["kwidth"]
        self.ratio = self.configs["ratio"]
        self.noise_std = self.configs["noise_std"]
        self.l1_lambda = self.configs["l1_lambda"]
        self.coeff = self.configs["pre_emph"]
        self.window_size = self.configs["window_size"]
        self.stride = self.configs["stride"]
        self.deactivated_noise = False

        self.generator = create_generator(g_enc_depths=self.g_enc_depths,
                                          window_size=self.window_size,
                                          kwidth=self.kwidth, ratio=self.ratio)

        if training:
            self.discriminator = create_discriminator(d_num_fmaps=self.d_num_fmaps,
                                                      window_size=self.window_size,
                                                      kwidth=self.kwidth,
                                                      ratio=self.ratio)

            self.generator_optimizer = tf.keras.optimizers.RMSprop(
                self.configs["g_learning_rate"])
            self.discriminator_optimizer = tf.keras.optimizers.RMSprop(
                self.configs["d_learning_rate"])

            self.writer = tf.summary.create_file_writer(self.configs["log_dir"])

            self.steps = tf.Variable(initial_value=0, trainable=False, shape=(), dtype=tf.int64)

            self.checkpoint = tf.train.Checkpoint(
                generator=self.generator,
                discriminator=self.discriminator,
                generator_optimizer=self.generator_optimizer,
                discriminator_optimizer=self.discriminator_optimizer,
                steps=self.steps
            )
            self.ckpt_manager = tf.train.CheckpointManager(
                self.checkpoint, self.configs["checkpoint_dir"], max_to_keep=5)

            print(self.generator.summary())
            print(self.discriminator.summary())

    def train(self, export_dir=None):
        train_dataset = SeganDataset(clean_data_dir=self.configs["clean_train_data_dir"],
                                     noises_dir=self.configs["noises_dir"],
                                     noise=self.configs["noise_conf"],
                                     window_size=self.window_size, stride=self.stride)

        tf_train_dataset = train_dataset.create(self.configs["batch_size"], coeff=self.coeff,
                                                sample_rate=self.configs["sample_rate"])

        epochs = self.configs["num_epochs"]

        initial_epoch = 0
        if self.ckpt_manager.latest_checkpoint:
            initial_epoch = int(self.ckpt_manager.latest_checkpoint.split('-')[-1])
            # restoring the latest checkpoint in checkpoint_path
            self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)

        @tf.function
        def train_step(clean_wavs, noisy_wavs):
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
                                                             l1_lambda=self.l1_lambda,
                                                             d_fake_logit=d_fake_logit)

                _disc_loss = discriminator_loss(d_real_logit, d_fake_logit)

                _gen_loss = _gen_l1_loss + _gen_adv_loss

            gradients_of_generator = gen_tape.gradient(_gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(_disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                             self.discriminator.trainable_variables))
            return _gen_l1_loss, _gen_adv_loss, _disc_loss

        for epoch in range(initial_epoch, epochs):
            start = time.time()
            g_l1_loss = []
            g_adv_loss = []
            d_loss = []

            if epoch > self.configs["denoise_epoch"] and self.deactivated_noise == False:
                self.noise_std = self.configs["noise_decay"] * self.noise_std
                if self.noise_std < self.configs["noise_std_lbound"]:
                    self.noise_std = 0.
                    self.deactivated_noise = True

            for step, (clean_wav, noisy_wav) in tf_train_dataset.enumerate(start=0):
                substart = time.time()
                gen_l1_loss, gen_adv_loss, disc_loss = train_step(clean_wav, noisy_wav)
                g_l1_loss.append(gen_l1_loss)
                g_adv_loss.append(gen_adv_loss)
                d_loss.append(disc_loss)
                sys.stdout.write("\033[K")
                print(f"\rEpoch: {epoch + 1}/{epochs}, step: {step}/{self.steps.numpy() / (epoch + 1)}, "
                      f"duration: {(time.time() - substart):.2f}s, "
                      f"gen_l1_loss = {gen_l1_loss}, gen_adv_loss = {gen_adv_loss}, "
                      f"disc_loss = {disc_loss}", end="")
                if self.writer and step % 500 == 0:
                    with self.writer.as_default():
                        tf.summary.scalar("g_l1_loss", tf.reduce_mean(g_l1_loss), step=(self.steps + step))
                        tf.summary.scalar("g_adv_loss", tf.reduce_mean(g_adv_loss), step=(self.steps + step))
                        tf.summary.scalar("d_loss", tf.reduce_mean(d_loss), step=(self.steps + step))

            self.steps.assign((epoch + 1) * (step + 1))

            self.ckpt_manager.save()
            print(f"\nSaved checkpoint at epoch {epoch + 1}", flush=True)
            print(f"Time for epoch {epoch + 1} is {time.time() - start} secs")

        if export_dir:
            self.save(export_dir)

    def test(self, export_dir: str, output_file_dir: str):
        test_dataset = SeganDataset(clean_data_dir=self.configs["clean_test_data_dir"],
                                    noises_dir=self.configs["noises_dir"],
                                    noise=self.configs["noise_conf"],
                                    window_size=self.window_size, stride=1)

        tf_test_dataset = test_dataset.create_test(coeff=self.coeff, sample_rate=self.configs["sample_rate"])

        msg = self.load_model(export_dir)
        if msg: raise Exception(msg)

        start = time.time()

        pesq_noisy, csig_noisy, cbak_noisy, covl_noisy, ssnr_noisy = 0
        pesq_gen, csig_gen, cbak_gen, covl_gen, ssnr_gen = 0

        try:
            from semetrics.main import pesq, composite, ssnr
            import soundfile as sf
        except ImportError as e:
            print(e)
            print("Please install https://github.com/usimarit/semetrics")
            return

        sr = self.configs["sample_rate"]

        def save_to_tmp(clean_signal, gen_signal, noisy_signal):
            sf.write("/tmp/clean_signal.wav", clean_signal, sr)
            sf.write("/tmp/gen_signal.wav", gen_signal, sr)
            sf.write("/tmp/noisy_signal.wav", noisy_signal, sr)

        for step, [clean_wav, noisy_wav] in tf_test_dataset.enumerate(start=1):
            gen_wav = self.generate(noisy_wav)
            clean_wav = clean_wav.numpy(); noisy_wav = noisy_wav.numpy()
            save_to_tmp(clean_wav, gen_wav, noisy_wav)

            pesq_gen += pesq(sr, "/tmp/clean_signal.wav", "/tmp/gen_signal.wav")
            pesq_gen += pesq(sr, "/tmp/clean_signal.wav", "/tmp/noisy_signal.wav")

            _csig_gen, _cbak_gen, _covl_gen = composite("/tmp/clean_signal.wav", "/tmp/gen_signal.wav")
            csig_gen += _csig_gen; cbak_gen += _cbak_gen; covl_gen += _covl_gen
            _csig_noisy, _cbak_noisy, _covl_noisy = composite("/tmp/clean_signal.wav", "/tmp/noisy_signal.wav")
            csig_noisy += _csig_noisy; cbak_noisy += _cbak_noisy; covl_noisy += _covl_noisy

            ssnr_gen += ssnr("/tmp/clean_signal.wav", "/tmp/gen_signal.wav", sr)
            ssnr_noisy += ssnr("/tmp/clean_signal.wav", "/tmp/noisy_signal.wav", sr)

            print(f"\rPESQ = {pesq_gen / step}, CSIG = {csig_gen / step}, "
                  "CBAK = {cbak_gen / step}, COVL = {covl_gen / step}, SSNR = {ssnr_gen / step}", end="")
            print(f"\rPESQ = {pesq_gen / step}, CSIG = {csig_gen / step}, "
                  "CBAK = {cbak_gen / step}, COVL = {covl_gen / step}, SSNR = {ssnr_gen / step}", end="")

        with open(output_file_dir, "w", encoding="utf-8") as fo:
            fo.write(f"PESQ_GEN = {(pesq_gen / step):.2f}, CSIG_GEN = {(csig_gen / step):.2f}, "
                     f"CBAK_GEN = {(cbak_gen / step):.2f}, COVL_GEN = {(covl_gen / step):.2f}, "
                     f"SSNR_GEN = {(ssnr_gen / step):.2f}\n")
            fo.write(f"PESQ_NOISY = {(pesq_noisy / step):.2f}, CSIG_NOISY = {(csig_noisy / step):.2f}, "
                     f"CBAK_NOISY = {(cbak_noisy / step):.2f}, COVL_NOISY = {(covl_noisy / step):.2f}, "
                     f"SSNR_NOISY = {(ssnr_noisy / step):.2f}\n")

        print(f"\nTime for testing is {time.time() - start} secs")

    def save_from_checkpoint(self, export_dir):
        if self.ckpt_manager.latest_checkpoint:
            # restoring the latest checkpoint in checkpoint_path
            self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
        else:
            raise ValueError("Model is not trained")

        self.save(export_dir)

    def save(self, export_dir):
        self.generator.save_weights(export_dir)

    def load_model(self, export_dir):
        try:
            self.generator.load_weights(export_dir)
        except Exception as e:
            return f"Model is not trained: {e}"
        return None

    def generate(self, signal):
        signal = preemphasis(signal, self.configs["pre_emph"])
        slices = slice_signal(signal, self.window_size, stride=1)

        @tf.function
        def gen(sliced_signal):
            sliced_signal = tf.reshape(sliced_signal, [-1, self.window_size])
            g_wavs = self.generator(sliced_signal, training=False)
            return merge_slices(g_wavs)

        signal = gen(tf.convert_to_tensor(slices)).numpy()
        return deemphasis(signal, self.configs["pre_emph"])

    def convert_to_tflite(self, export_file, output_file_path):
        if os.path.exists(output_file_path):
            return
        msg = self.load_model(export_file)
        print(msg)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.generator)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()

        tflite_model_dir = pathlib.Path(os.path.dirname(output_file_path))
        tflite_model_dir.mkdir(exist_ok=True, parents=True)

        tflite_model_file = tflite_model_dir / f"{os.path.basename(output_file_path)}"
        tflite_model_file.write_bytes(tflite_model)

    def load_interpreter(self, export_dir):
        try:
            self.generator = tf.lite.Interpreter(model_path=export_dir)
            self.generator.allocate_tensors()
        except Exception as e:
            return f"Model is not trained: {e}"
        return None

    def generate_interpreter(self, signal):
        signal = preemphasis(signal, self.configs["pre_emph"])
        slices = slice_signal(signal, self.window_size, stride=1)
        slices = tf.reshape(slices, [-1, self.window_size])

        input_index = self.generator.get_input_details()[0]["index"]
        output_index = self.generator.get_output_details()[0]["index"]

        self.generator.set_tensor(input_index, slices)
        self.generator.invoke()

        pred = self.generator.get_tensor(output_index)
        pred = merge_slices(pred)
        return deemphasis(pred.numpy(), self.configs["pre_emph"])
