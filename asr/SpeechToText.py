from __future__ import absolute_import

import os
import tempfile
import tensorflow as tf

from models.CTCModel import CTCModel
from decoders.Decoders import create_decoder
from featurizers.SpeechFeaturizer import SpeechFeaturizer
from featurizers.TextFeaturizer import TextFeaturizer
from utils.Utils import get_asr_config, check_key_in_dict, \
  bytes_to_string, get_length, wer, cer
from utils.TimeHistory import TimeHistory
from utils.Checkpoint import Checkpoint
from data.Dataset import Dataset


class SpeechToText:
  def __init__(self, configs_path):
    self.configs = get_asr_config(configs_path)
    self.speech_featurizer = SpeechFeaturizer(
      sample_rate=self.configs["sample_rate"],
      frame_ms=self.configs["frame_ms"],
      stride_ms=self.configs["stride_ms"],
      num_feature_bins=self.configs["num_feature_bins"],
      feature_type=self.configs["feature_type"])
    self.text_featurizer = TextFeaturizer(
      self.configs["vocabulary_file_path"])
    self.decoder = create_decoder(
      decoder_config=self.configs["decoder"],
      index_to_token=self.text_featurizer.index_to_token,
      vocab_array=self.text_featurizer.vocab_array)
    self.model = CTCModel(
      num_classes=self.text_featurizer.num_classes,
      num_feature_bins=self.speech_featurizer.num_feature_bins,
      learning_rate=self.configs["learning_rate"],
      min_lr=self.configs["min_lr"],
      base_model=self.configs["base_model"],
      streaming_size=self.configs["streaming_size"])

  def train_and_eval(self, model_file=None):
    print("Training and evaluating model ...")
    self.ckpt = tf.train.Checkpoint(model=self.model.model,
                                    optimizer=self.model.optimizer)
    self.ckpt_manager = tf.train.CheckpointManager(
      self.ckpt, self.configs["checkpoint_dir"], max_to_keep=5)
    check_key_in_dict(dictionary=self.configs,
                      keys=["train_data_transcript_paths",
                            "eval_data_transcript_paths"])
    train_dataset = Dataset(
      data_path=self.configs["train_data_transcript_paths"],
      mode="train", train_sort=True)
    eval_dataset = Dataset(
      data_path=self.configs["eval_data_transcript_paths"],
      mode="eval")
    if "augmentations" in self.configs.keys():
      augmentations = self.configs["augmentations"]
      augmentations.append(None)
    train_dataset = train_dataset(
      speech_featurizer=self.speech_featurizer,
      text_featurizer=self.text_featurizer,
      batch_size=self.configs["batch_size"],
      augmentations=augmentations)
    eval_dataset = eval_dataset(
      speech_featurizer=self.speech_featurizer,
      text_featurizer=self.text_featurizer,
      batch_size=self.configs["batch_size"])

    self.model.summary()

    initial_epoch = 0
    if self.ckpt_manager.latest_checkpoint:
      initial_epoch = int(self.ckpt_manager.latest_checkpoint.split('-')[-1])
      # restoring the latest checkpoint in checkpoint_path
      self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

    cp_callback = Checkpoint(ckpt_manager=self.ckpt_manager)
    callbacks = [cp_callback]

    if "log_dir" in self.configs.keys():
      tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=self.configs["log_dir"], histogram_freq=1,
        update_freq=500, write_images=True)
      callbacks.append(tb_callback)
      csv_callback = tf.keras.callbacks.CSVLogger(
        filename=os.path.join(self.configs["log_dir"], "training.log"),
        append=True)
      callbacks.append(csv_callback)
      time_callback = TimeHistory(os.path.join(self.configs["log_dir"],
                                               "time.log"))
      callbacks.append(time_callback)
      with open(os.path.join(self.configs["log_dir"],
                             "model.json"), "w") as f:
        f.write(self.model.to_json())

    self.model.compile(optimizer=self.model.optimizer,
                       loss=self.model.loss)

    self.model.fit(
      x=train_dataset, epochs=self.configs["num_epochs"],
      validation_data=eval_dataset, shuffle="batch",
      initial_epoch=initial_epoch,
      callbacks=callbacks)

    if model_file:
      self.model.save(model_file)

  def test(self, model_file, output_file_path):
    tf.compat.v1.set_random_seed(0)
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

    @tf.function
    def test_step(features, transcripts):
      logits = self.model(features, training=False)
      logits = logits.numpy()
      print(logits)
      predictions = self.decoder.decode(
        probs=logits,
        input_length=tf.squeeze(get_length(logits), -1))

      print(predictions)

      transcripts = transcripts.numpy()

      b_wer = 0.0
      b_wer_count = 0.0
      b_cer = 0.0
      b_cer_count = 0.0

      for idx, decoded in enumerate(predictions):
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
    tf.compat.v1.set_random_seed(0)
    print("Infering ...")
    self.model = tf.keras.models.load_model(model_file)
    tf_infer_dataset = Dataset(data_path=input_file_path,
                               mode="infer")
    tf_infer_dataset = tf_infer_dataset(
      speech_featurizer=self.speech_featurizer,
      batch_size=self.configs["batch_size"])
    logits = self.model.predict(x=tf_infer_dataset)
    predictions = self.decoder.decode(
      probs=logits,
      input_length=tf.squeeze(get_length(logits), -1))

    with open(output_file_path, "w", encoding="utf-8") as of:
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
    logits = self.model.predict(x=features, batch_size=1)
    predictions = self.decoder.decode(
      probs=logits,
      input_length=tf.squeeze(get_length(logits), -1))

    return predictions[0]

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
