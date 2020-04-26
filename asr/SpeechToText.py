from __future__ import absolute_import

import os
import time
import tensorflow as tf

from models.CTCModel import CTCModel
from decoders.Decoders import create_decoder
from featurizers.SpeechFeaturizer import SpeechFeaturizer
from featurizers.TextFeaturizer import TextFeaturizer
from utils.Utils import get_asr_config, check_key_in_dict, \
  bytes_to_string, get_length, wer, cer, scalar_summary
from data.Dataset import Dataset


class SpeechToText:
  def __init__(self, configs_path, noise_filter=None):
    self.configs = get_asr_config(configs_path)
    self.speech_featurizer = SpeechFeaturizer(
      sample_rate=self.configs["sample_rate"],
      frame_ms=self.configs["frame_ms"],
      stride_ms=self.configs["stride_ms"],
      num_feature_bins=self.configs["num_feature_bins"],
      feature_type=self.configs["feature_type"])
    self.text_featurizer = TextFeaturizer(self.configs["vocabulary_file_path"])
    self.decoder = create_decoder(
      decoder_config=self.configs["decoder"],
      index_to_token=self.text_featurizer.index_to_token,
      vocab_array=self.text_featurizer.vocab_array)
    self.model = CTCModel(
      speech_featurizer=self.speech_featurizer,
      num_classes=self.text_featurizer.num_classes,
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
    train_dataset = Dataset(data_path=self.configs["train_data_transcript_paths"],
                            num_classes=self.text_featurizer.num_classes, mode="train")
    eval_dataset = Dataset(data_path=self.configs["eval_data_transcript_paths"],
                           num_classes=self.text_featurizer.num_classes, mode="eval")

    augmentations = []
    if "augmentations" in self.configs.keys():
      augmentations = self.configs["augmentations"]
      augmentations.append(None)

    tf_train_dataset = train_dataset(speech_featurizer=self.speech_featurizer,
                                     text_featurizer=self.text_featurizer,
                                     batch_size=self.configs["batch_size"],
                                     augmentations=augmentations)
    tf_train_dataset_sorted = train_dataset(speech_featurizer=self.speech_featurizer,
                                            text_featurizer=self.text_featurizer,
                                            batch_size=self.configs["batch_size"],
                                            augmentations=augmentations, sort=True)
    tf_eval_dataset = eval_dataset(speech_featurizer=self.speech_featurizer,
                                   text_featurizer=self.text_featurizer,
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
        print(f"Epoch: {epoch + 1}/{epochs}, batch: {batch_idx}/{num_batch}, "
              f"train_loss = {train_loss}", end="\r", flush=True)
        batch_idx += 1

      num_batch = batch_idx

      for feature, transcript, label_length in tf_eval_dataset:
        _eval_loss = eval_step(feature, transcript, label_length)
        eval_loss.append(_eval_loss)

      eval_loss = tf.reduce_mean(eval_loss)
      epoch_train_loss = tf.reduce_mean(epoch_train_loss)
      print(f"\nEpoch: {epoch + 1}/{epochs}, eval_loss = {eval_loss}")

      self.ckpt_manager.save()
      print(f"\nSaved checkpoint at epoch {epoch + 1}")
      time_epoch = time.time() - start
      print(f"Time for epoch {epoch + 1} is {time_epoch} secs")

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
    tf_test_dataset = test_dataset(
      speech_featurizer=self.speech_featurizer,
      text_featurizer=self.text_featurizer,
      batch_size=self.configs["batch_size"])

    def test_step(features, transcripts):
      predictions = self.predict(features)

      transcripts = self.decoder.convert_to_string(transcripts)

      b_wer = 0.0
      b_wer_count = 0.0
      b_cer = 0.0
      b_cer_count = 0.0

      for idx, decoded in enumerate(predictions):
        print(decoded)
        _wer, _wer_count = wer(decode=decoded, target=transcripts[idx])
        _cer, _cer_count = cer(decode=decoded, target=transcripts[idx])
        b_wer += _wer
        b_cer += _cer
        b_wer_count += _wer_count
        b_cer_count += _cer_count

      print(f"batch_wer: {b_wer / b_wer_count}, batch_cer: {b_cer / b_cer_count}")
      return b_wer, b_wer_count, b_cer, b_cer_count

    total_wer = 0.0
    wer_count = 0.0
    total_cer = 0.0
    cer_count = 0.0

    for feature, label in tf_test_dataset:
      batch_wer, batch_wer_count, batch_cer, batch_cer_count = test_step(feature, label)
      total_wer += batch_wer
      total_cer += batch_cer
      wer_count += batch_wer_count
      cer_count += batch_cer_count

    results = (total_wer / wer_count, total_cer / cer_count)

    with open(output_file_path, "w", encoding="utf-8") as of:
      of.write("WER: " + str(results[0]) + "\n")
      of.write("CER: " + str(results[-1]) + "\n")

  def infer(self, input_file_path, model_file, output_file_path):
    print("Infering ...")
    self.load_model(model_file)
    tf_infer_dataset = Dataset(data_path=input_file_path,
                               mode="infer")
    tf_infer_dataset = tf_infer_dataset(
      speech_featurizer=self.speech_featurizer,
      batch_size=self.configs["batch_size"])

    def infer_step(feature):
      prediction = self.predict(feature)
      return bytes_to_string(prediction.numpy())

    for features in tf_infer_dataset:
      predictions = infer_step(features)

      with open(output_file_path, "a", encoding="utf-8") as of:
        of.write("Predictions\n")
        for pred in predictions:
          of.write(pred + "\n")

  def infer_single(self, audio, sample_rate=None,
                   channels=None, streaming=False):
    if sample_rate and channels:
      features = self.speech_featurizer.compute_speech_features(
        audio, sr=sample_rate, channels=channels)
    else:
      features = self.speech_featurizer.compute_speech_features(audio)
    features = tf.expand_dims(features, axis=0)
    if streaming:
      features = tf.pad(
        features,
        [[0, 0],
         [0, int(self.configs["streaming_size"]) - features.shape[1]],
         [0, 0],
         [0, 0]],
        "CONSTANT")
    pred = self.predict(features)

    return pred[0]

  def load_model(self, model_file):
    tf.compat.v1.set_random_seed(0)
    try:
      self.model.load_model(model_file)
    except Exception as e:
      raise ValueError("Model is not trained: ", e)
    return None

  def load_model_from_weights(self, model_file):
    tf.compat.v1.set_random_seed(0)
    try:
      self.model.load_weights(model_file)
    except Exception as e:
      raise ValueError("Model is not trained: ", e)
    return None

  def predict(self, features):
    logits = self.model(features, training=False)
    return self.decoder.decode(probs=logits, input_length=get_length(logits))
