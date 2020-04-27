from __future__ import absolute_import

import os
import time
import tensorflow as tf

from models.CTCModel import CTCModel
from decoders.Decoders import create_decoder
from featurizers.TextFeaturizer import TextFeaturizer
from utils.Utils import get_asr_config, check_key_in_dict, \
  bytes_to_string, wer, cer, scalar_summary
from data.Dataset import Dataset


class SpeechToText:
  def __init__(self, configs_path, noise_filter=None):
    self.configs = get_asr_config(configs_path)
    self.text_featurizer = TextFeaturizer(self.configs["vocabulary_file_path"])
    self.decoder = create_decoder(
      decoder_config=self.configs["decoder"],
      index_to_token=self.text_featurizer.index_to_token,
      num_classes=self.text_featurizer.num_classes,
      vocab_array=self.text_featurizer.vocab_array)
    self.model = CTCModel(
      num_classes=self.text_featurizer.num_classes,
      sample_rate=self.configs["sample_rate"],
      frame_ms=self.configs["frame_ms"],
      stride_ms=self.configs["stride_ms"],
      num_feature_bins=self.configs["num_feature_bins"],
      learning_rate=self.configs["learning_rate"],
      min_lr=self.configs["min_lr"],
      base_model=self.configs["base_model"],
      streaming_size=self.configs["streaming_size"])
    self.noise_filter = noise_filter
    self.writer = None

  def train_and_eval(self, model_file=None):
    print("Training and evaluating model ...")
    self.ckpt = tf.train.Checkpoint(model=self.model.model,
                                    optimizer=self.model.optimizer)
    self.ckpt_manager = tf.train.CheckpointManager(
      self.ckpt, self.configs["checkpoint_dir"], max_to_keep=5)
    check_key_in_dict(dictionary=self.configs,
                      keys=["train_data_transcript_paths",
                            "eval_data_transcript_paths"])
    train_dataset = Dataset(data_path=self.configs["train_data_transcript_paths"], mode="train")
    eval_dataset = Dataset(data_path=self.configs["eval_data_transcript_paths"], mode="eval")

    augmentations = []
    if "augmentations" in self.configs.keys():
      augmentations = self.configs["augmentations"]
      augmentations.append(None)

    tf_train_dataset = train_dataset(text_featurizer=self.text_featurizer,
                                     sample_rate=self.configs["sample_rate"],
                                     preemph=self.configs["pre_emph"],
                                     batch_size=self.configs["batch_size"],
                                     augmentations=augmentations)
    tf_train_dataset_sorted = train_dataset(text_featurizer=self.text_featurizer,
                                            sample_rate=self.configs["sample_rate"],
                                            preemph=self.configs["pre_emph"],
                                            batch_size=self.configs["batch_size"],
                                            augmentations=augmentations, sortagrad=True)
    tf_eval_dataset = eval_dataset(text_featurizer=self.text_featurizer,
                                   sample_rate=self.configs["sample_rate"],
                                   preemph=self.configs["pre_emph"],
                                   batch_size=self.configs["batch_size"])

    self.model.summary()

    initial_epoch = 0
    if self.ckpt_manager.latest_checkpoint:
      initial_epoch = int(self.ckpt_manager.latest_checkpoint.split('-')[-1])
      # restoring the latest checkpoint in checkpoint_path
      self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

    if "log_dir" in self.configs.keys():
      with open(os.path.join(self.configs["log_dir"], "model.json"), "w") as f:
        f.write(self.model.to_json())
      self.writer = tf.summary.create_file_writer(os.path.join(self.configs["log_dir"], "train"))

    @tf.function
    def train_step(features, y_true, lab_length):
      with tf.GradientTape() as tape:
        y_pred, inp_length = self.model(features, training=True)
        _loss = self.model.loss(y_true=y_true, y_pred=y_pred, input_length=inp_length, label_length=lab_length)
      gradients = tape.gradient(_loss, self.model.model.trainable_variables)
      self.model.optimizer.apply_gradients(zip(gradients, self.model.model.trainable_variables))
      return _loss

    @tf.function
    def eval_step(features, y_true, lab_length):
      y_pred, inp_length = self.model(features, training=False)
      _loss = self.model.loss(y_true=y_true, y_pred=y_pred, input_length=inp_length, label_length=lab_length)
      return _loss

    epochs = self.configs["num_epochs"]
    num_batch = None

    for epoch in range(initial_epoch, epochs, 1):
      if epoch == 0:
        dataset = tf_train_dataset_sorted
      else:
        dataset = tf_train_dataset

      eval_loss = []
      epoch_train_loss = []
      batch_idx = 1
      start = time.time()

      for feature, transcript, label_length in dataset:
        train_loss = train_step(feature, transcript, label_length)
        epoch_train_loss.append(train_loss)
        tf.print(f"Epoch: {epoch + 1}/{epochs}, batch: {batch_idx}/{num_batch}, "
                 f"train_loss = {train_loss}", end="\r")
        batch_idx += 1

      num_batch = batch_idx

      for feature, transcript, label_length in tf_eval_dataset:
        _eval_loss = eval_step(feature, transcript, label_length)
        eval_loss.append(_eval_loss)

      eval_loss = tf.reduce_mean(eval_loss)
      epoch_train_loss = tf.reduce_mean(epoch_train_loss)
      tf.print(f"\nEpoch: {epoch + 1}/{epochs}, eval_loss = {eval_loss}")

      self.ckpt_manager.save()
      tf.print(f"\nSaved checkpoint at epoch {epoch + 1}")
      time_epoch = time.time() - start
      tf.print(f"Time for epoch {epoch + 1} is {time_epoch} secs")

      if self.writer:
        with self.writer.as_default():
          scalar_summary("train_loss", epoch_train_loss, step=epoch)
          scalar_summary("eval_loss", eval_loss, step=epoch)
          scalar_summary("epoch_time", time_epoch, step=epoch)
          self.writer.flush()

    if model_file:
      self.model.save(model_file)

  def test(self, model_file, output_file_path):
    print("Testing model ...")
    check_key_in_dict(dictionary=self.configs,
                      keys=["test_data_transcript_paths"])
    test_dataset = Dataset(
      data_path=self.configs["test_data_transcript_paths"],
      mode="test")
    self.load_model(model_file)
    tf_test_dataset = test_dataset(text_featurizer=self.text_featurizer,
                                   sample_rate=self.configs["sample_rate"],
                                   preemph=self.configs["pre_emph"],
                                   batch_size=self.configs["batch_size"])

    def test_step(features, transcripts):
      predictions = self.predict(features)
      predictions = bytes_to_string(predictions.numpy())

      transcripts = self.decoder.convert_to_string(transcripts)

      b_wer = 0.0
      b_wer_count = 0.0
      b_cer = 0.0
      b_cer_count = 0.0

      for idx, decoded in enumerate(predictions):
        print(f"Pred: {decoded}")
        print(f"Groundtruth: {transcripts[idx]}")
        _wer, _wer_count = wer(decode=decoded, target=transcripts[idx])
        _cer, _cer_count = cer(decode=decoded, target=transcripts[idx])
        b_wer += _wer
        b_cer += _cer
        b_wer_count += _wer_count
        b_cer_count += _cer_count

      return b_wer, b_wer_count, b_cer, b_cer_count

    total_wer = 0.0
    wer_count = 0.0
    total_cer = 0.0
    cer_count = 0.0

    for feature, label, _ in tf_test_dataset:
      batch_wer, batch_wer_count, batch_cer, batch_cer_count = test_step(feature, label)
      total_wer += batch_wer
      total_cer += batch_cer
      wer_count += batch_wer_count
      cer_count += batch_cer_count

    results = (total_wer / wer_count, total_cer / cer_count)

    print(f"WER: {results[0]}, CER: {results[-1]}")

    with open(output_file_path, "w", encoding="utf-8") as of:
      of.write("WER: " + str(results[0]) + "\n")
      of.write("CER: " + str(results[-1]) + "\n")

  def infer(self, input_file_path, model_file, output_file_path):
    print("Infering ...")
    self.load_model(model_file)
    tf_infer_dataset = Dataset(data_path=input_file_path,
                               mode="infer")
    tf_infer_dataset = tf_infer_dataset(batch_size=self.configs["batch_size"],
                                        text_featurizer=self.text_featurizer,
                                        preemph=self.configs["pre_emph"],
                                        sample_rate=self.configs["sample_rate"])

    def infer_step(feature):
      prediction = self.predict(feature)
      return bytes_to_string(prediction.numpy())

    for features in tf_infer_dataset:
      predictions = infer_step(features)

      with open(output_file_path, "a", encoding="utf-8") as of:
        of.write("Predictions\n")
        for pred in predictions:
          of.write(pred + "\n")

  def infer_single(self, signal):
    signal = tf.expand_dims(signal, axis=0)
    pred = self.predict(signal)
    return bytes_to_string(pred.numpy())[0]

  def load_model(self, model_file):
    tf.compat.v1.set_random_seed(0)
    try:
      self.model.load_model(model_file)
    except Exception as e:
      return f"Model is not trained: {e}"
    return None

  def load_model_from_weights(self, model_file):
    tf.compat.v1.set_random_seed(0)
    try:
      self.model.load_weights(model_file)
    except Exception as e:
      return f"Model is not trained: {e}"
    return None

  @tf.function
  def predict(self, signal):
    logits, logit_length = self.model(signal, training=False)
    return self.decoder(probs=logits, input_length=logit_length)

  def save_model(self, model_file):
    self.model.save(model_file)
