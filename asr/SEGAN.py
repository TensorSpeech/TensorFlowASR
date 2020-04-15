from __future__ import absolute_import

import time
import tensorflow as tf
from models.segan.Discriminator import Discriminator
from models.segan.Generator import Generator


class SEGAN:
  def __init__(self, configs):
    self.g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
    self.d_num_fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]

    self.configs = configs

    self.kwidth = self.configs["kwidth"]
    self.ratio = self.configs["ratio"]
    self.noise_std = self.configs["noise_std"]
    self.l1_lambda = self.configs["l1_lambda"]

    self.generator = Generator(g_enc_depths=self.g_enc_depths,
                               kwidth=self.kwidth, ratio=self.ratio)

    self.discriminator = Discriminator(d_num_fmaps=self.d_num_fmaps,
                                       noise_std=self.noise_std,
                                       kwidth=self.kwidth,
                                       pooling=self.ratio)

    self.generator_optimizer = tf.keras.optimizers.RMSprop(self.configs["g_learning_rate"])
    self.discriminator_optimizer = tf.keras.optimizers.RMSprop(self.configs["d_learning_rate"])

  def train(self, dataset, epochs):

    self.checkpoint = tf.train.Checkpoint(
      generator=self.generator,
      discriminator=self.discriminator,
      generator_optimizer=self.generator_optimizer,
      discriminator_optimizer=self.discriminator_optimizer
    )
    self.ckpt_manager = tf.train.CheckpointManager(
      self.checkpoint, self.configs["checkpoint_dir"], max_to_keep=5)

    @tf.function
    def train_step(clean_wavs, noisy_wavs):
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        g_clean_wavs = self.generator(noisy_wavs, training=True)

        d_real_logit = self.discriminator(clean_wavs, noisy_wavs, training=True)
        d_fake_logit = self.discriminator(g_clean_wavs, noisy_wavs, training=True)

        gen_loss = self.generator.loss(y_true=clean_wavs,
                                       y_pred=g_clean_wavs,
                                       l1_lambda=self.l1_lambda,
                                       d_fake_logit=d_fake_logit)

        disc_loss = self.discriminator.loss(d_real_logit, d_fake_logit)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss

    for epoch in range(epochs):
      start = time.time()
      batch_idx = 0

      for clean_wav, noisy_wav in dataset:
        gen_loss, disc_loss = train_step(clean_wav, noisy_wav)
        print(f"{epoch + 1}/{epochs}, batch: {batch_idx}, gen_loss = {gen_loss}, disc_loss = {disc_loss}")

      self.ckpt_manager.save()

      print(f"Time for epoch {epoch + 1} is {time.time() - start} secs")
