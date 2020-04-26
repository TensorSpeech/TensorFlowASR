from __future__ import absolute_import

import time
import tensorflow as tf
from models.segan.Discriminator import create_discriminator, discriminator_loss
from models.segan.Generator import create_generator, generator_loss
from utils.Utils import get_segan_config, slice_signal, merge_slices
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

      self.checkpoint = tf.train.Checkpoint(
        generator=self.generator,
        discriminator=self.discriminator,
        generator_optimizer=self.generator_optimizer,
        discriminator_optimizer=self.discriminator_optimizer
      )
      self.ckpt_manager = tf.train.CheckpointManager(
        self.checkpoint, self.configs["checkpoint_dir"], max_to_keep=5)

      print(self.generator.summary())
      print(self.discriminator.summary())

  def train(self, export_dir=None):
    train_dataset = SeganDataset(clean_data_dir=self.configs["clean_train_data_dir"],
                                 noisy_data_dir=self.configs["noisy_train_data_dir"],
                                 window_size=self.window_size, stride=self.stride)

    tf_train_dataset = train_dataset.create(self.configs["batch_size"], coeff=self.coeff)

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

    num_batch = None

    for epoch in range(initial_epoch, epochs):
      start = time.time()
      batch_idx = 1

      if epoch > self.configs["denoise_epoch"] and self.deactivated_noise == False:
        self.noise_std = self.configs["noise_decay"] * self.noise_std
        if self.noise_std < self.configs["noise_std_lbound"]:
          self.noise_std = 0.
          self.deactivated_noise = True

      for clean_wav, noisy_wav in tf_train_dataset:
        gen_l1_loss, gen_adv_loss, disc_loss = train_step(clean_wav, noisy_wav)
        print(f"Epoch: {epoch + 1}/{epochs}, batch: {batch_idx}/{num_batch}, "
              f"gen_l1_loss = {gen_l1_loss}, gen_adv_loss = {gen_adv_loss}, "
              f"disc_loss = {disc_loss}", end="\r", flush=True)
        batch_idx += 1

      num_batch = batch_idx

      self.ckpt_manager.save()
      print(f"\nSaved checkpoint at epoch {epoch + 1}", flush=True)
      print(f"Time for epoch {epoch + 1} is {time.time() - start} secs")

    if export_dir:
      self.save(export_dir)

  def test(self):
    test_dataset = SeganDataset(clean_data_dir=self.configs["clean_test_data_dir"],
                                noisy_data_dir=self.configs["noisy_test_data_dir"],
                                window_size=self.window_size, stride=self.stride)

    tf_test_dataset = test_dataset.create(self.configs["batch_size"])

    if self.ckpt_manager.latest_checkpoint:
      # restoring the latest checkpoint in checkpoint_path
      self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
    else:
      raise ValueError("Model is not trained")

    @tf.function
    def test_step(clean_wavs, noisy_wavs):
      g_clean_wavs = self.generator(noisy_wavs, training=False)

      d_real_logit = self.discriminator(clean_wavs, noisy_wavs, training=False)
      d_fake_logit = self.discriminator(g_clean_wavs, noisy_wavs, training=False)

      _gen_loss = self.generator.loss(y_true=clean_wavs,
                                      y_pred=g_clean_wavs,
                                      l1_lambda=self.l1_lambda,
                                      d_fake_logit=d_fake_logit)

      _disc_loss = self.discriminator.loss(d_real_logit, d_fake_logit)
      # Evaluation methods
      return _gen_loss, _disc_loss

    start = time.time()
    batch_idx = 0

    for clean_wav, noisy_wav in tf_test_dataset:
      gen_loss, disc_loss = test_step(clean_wav, noisy_wav)
      print(f"batch: {batch_idx}, gen_loss = {gen_loss}, disc_loss = {disc_loss}")
      batch_idx += 1

    print(f"Time for testing is {time.time() - start} secs")

  def save_from_checkpoint(self, export_dir):
    if self.ckpt_manager.latest_checkpoint:
      # restoring the latest checkpoint in checkpoint_path
      self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
    else:
      raise ValueError("Model is not trained")

    self.save(export_dir)

  def save(self, export_dir):
    self.generator.save_weights(export_dir)

  def generate(self, signal):
    slices = slice_signal(signal, self.window_size, stride=1)

    @tf.function
    def gen(sliced_signal):
      sliced_signal = tf.reshape(sliced_signal, [-1, self.window_size])
      g_wavs = self.generator(sliced_signal, training=False)
      return merge_slices(g_wavs)

    return gen(tf.convert_to_tensor(slices))
