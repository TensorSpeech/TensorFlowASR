from __future__ import absolute_import

import os
import gc
import sys
import time
import tensorflow as tf

from models.CTCModel import create_ctc_model, ctc_loss, ctc_loss_1, ctc_loss_keras, ctc_loss_keras_2
from decoders.Decoders import create_decoder
from featurizers.TextFeaturizer import TextFeaturizer
from utils.Utils import get_asr_config, check_key_in_dict, bytes_to_string, wer, cer
from featurizers.SpeechFeaturizer import compute_mfcc_feature
from utils.Checkpoint import Checkpoint
from utils.TimeHistory import TimeHistory
from data.Dataset import Dataset


class SpeechToText:
  def __init__(self, configs_path, noise_filter=None):
    self.configs = get_asr_config(configs_path)
    self.text_featurizer = TextFeaturizer(self.configs["vocabulary_file_path"])
    self.decoder = create_decoder(decoder_config=self.configs["decoder"],
                                  index_to_token=self.text_featurizer.index_to_token,
                                  num_classes=self.text_featurizer.num_classes,
                                  vocab_array=self.text_featurizer.vocab_array)
    self.model, self.optimizer = create_ctc_model(num_classes=self.text_featurizer.num_classes,
                                                  last_activation=self.configs["last_activation"],
                                                  base_model=self.configs["base_model"],
                                                  streaming_size=self.configs["streaming_size"],
                                                  speech_conf=self.configs["speech_conf"])
    self.noise_filter = noise_filter
    self.writer = None

  def _create_checkpoints(self):
    self.ckpt = tf.train.Checkpoint(model=self.model,
                                    optimizer=self.optimizer)
    self.ckpt_manager = tf.train.CheckpointManager(
      self.ckpt, self.configs["checkpoint_dir"], max_to_keep=None)

  def train_and_eval(self, model_file=None):
    print("Training and evaluating model ...")
    self._create_checkpoints()

    check_key_in_dict(dictionary=self.configs,
                      keys=["tfrecords_dir", "checkpoint_dir", "augmentations",
                            "log_dir", "train_data_transcript_paths"])
    augmentations = self.configs["augmentations"]
    augmentations.append(None)

    train_dataset = Dataset(data_path=self.configs["train_data_transcript_paths"],
                            tfrecords_dir=self.configs["tfrecords_dir"], mode="train")
    tf_train_dataset = train_dataset(text_featurizer=self.text_featurizer,
                                     speech_conf=self.configs["speech_conf"],
                                     batch_size=self.configs["batch_size"],
                                     augmentations=augmentations)

    tf_eval_dataset = None

    if "eval_data_transcript_paths" in self.configs.keys():
      eval_dataset = Dataset(data_path=self.configs["eval_data_transcript_paths"],
                             tfrecords_dir=self.configs["tfrecords_dir"], mode="eval")
      tf_eval_dataset = eval_dataset(text_featurizer=self.text_featurizer,
                                     speech_conf=self.configs["speech_conf"],
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

    if self.configs["last_activation"] == "linear":
      loss = ctc_loss
    else:
      loss = ctc_loss_1

    @tf.function
    def train_step(features, inp_length, y_true, lab_length):
      with tf.GradientTape() as tape:
        y_pred = self.model(features, training=True)
        _loss = loss(y_true=y_true, y_pred=y_pred,
                     input_length=inp_length, label_length=lab_length,
                     num_classes=self.text_featurizer.num_classes)
      gradients = tape.gradient(_loss, self.model.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
      return _loss

    def eval_step(features, inp_length, y_true, lab_length):
      @tf.function
      def sub_eval_step():
        y_pred = self.model(features, training=False)
        _loss = loss(y_true=y_true, y_pred=y_pred,
                     input_length=inp_length, label_length=lab_length,
                     num_classes=self.text_featurizer.num_classes)
        _pred = self.decoder(probs=y_pred, input_length=inp_length,
                             last_activation=self.configs["last_activation"])
        return _loss, _pred

      _val_loss, _eval_pred = sub_eval_step()
      predictions = bytes_to_string(_eval_pred.numpy())
      transcripts = self.decoder.convert_to_string(y_true)

      b_wer = 0.0
      b_wer_count = 0.0

      for idx, decoded in enumerate(predictions):
        _wer, _wer_count = wer(decode=decoded, target=transcripts[idx])
        _cer, _cer_count = cer(decode=decoded, target=transcripts[idx])
        b_wer += _wer
        b_wer_count += _wer_count

      return _val_loss, b_wer, b_wer_count

    epochs = self.configs["num_epochs"]
    num_batch = None

    for epoch in range(initial_epoch, epochs, 1):
      epoch_eval_loss = None
      epoch_eval_wer = None
      batch_idx = 1
      start = time.time()

      for step, [feature, input_length, transcript, label_length] in tf_train_dataset.enumerate(start=1):
        train_loss = train_step(feature, input_length, transcript, label_length)
        sys.stdout.write("\033[K")
        print(f"\rEpoch: {epoch + 1}/{epochs}, batch: {batch_idx}/{num_batch}, train_loss = {train_loss}", end="")
        batch_idx += 1
        if self.writer:
          with self.writer.as_default():
            tf.summary.scalar("train_loss", train_loss, step=(epoch * epochs + step))
        gc.collect()

      num_batch = batch_idx

      if tf_eval_dataset:
        print("Validating ... ", end="")
        eval_loss_count = 0
        epoch_eval_loss = 0.0
        total_wer = 0.0
        wer_count = 0.0
        for feature, input_length, transcript, label_length in tf_eval_dataset:
          _eval_loss, _wer, _wer_count = eval_step(feature, input_length, transcript, label_length)
          epoch_eval_loss += _eval_loss
          eval_loss_count += 1
          total_wer += _wer
          wer_count += _wer_count
        epoch_eval_loss = epoch_eval_loss / eval_loss_count
        epoch_eval_wer = total_wer / wer_count
        print(f"val_loss = {epoch_eval_loss}, wer = {epoch_eval_wer}")

      self.ckpt_manager.save()
      print(f"\nSaved checkpoint at epoch {epoch + 1}")
      time_epoch = time.time() - start
      tf.print(f"Time for epoch {epoch + 1} is {time_epoch} secs")

      if self.writer:
        with self.writer.as_default():
          if epoch_eval_loss and epoch_eval_wer:
            tf.summary.scalar("eval_loss", epoch_eval_loss, step=epoch)
            tf.summary.scalar("eval_wer", epoch_eval_wer, step=epoch)
          tf.summary.scalar("epoch_time", time_epoch, step=epoch)

    if model_file:
      self.model.save(model_file)

  def keras_train_and_eval(self, model_file=None):
    print("Training and evaluating model ...")
    self._create_checkpoints()

    check_key_in_dict(dictionary=self.configs,
                      keys=["tfrecords_dir", "checkpoint_dir", "augmentations",
                            "log_dir", "train_data_transcript_paths", "eval_data_transcript_paths"])
    augmentations = self.configs["augmentations"]
    augmentations.append(None)

    train_dataset = Dataset(data_path=self.configs["train_data_transcript_paths"],
                            tfrecords_dir=self.configs["tfrecords_dir"], mode="train", is_keras=True)
    tf_train_dataset = train_dataset(text_featurizer=self.text_featurizer,
                                     speech_conf=self.configs["speech_conf"],
                                     batch_size=self.configs["batch_size"],
                                     augmentations=augmentations)

    eval_dataset = Dataset(data_path=self.configs["eval_data_transcript_paths"],
                           tfrecords_dir=self.configs["tfrecords_dir"], mode="eval", is_keras=True)
    tf_eval_dataset = eval_dataset(text_featurizer=self.text_featurizer,
                                   speech_conf=self.configs["speech_conf"],
                                   batch_size=self.configs["batch_size"])

    self.model.summary()

    initial_epoch = 0
    if self.ckpt_manager.latest_checkpoint:
      initial_epoch = int(self.ckpt_manager.latest_checkpoint.split('-')[-1])
      # restoring the latest checkpoint in checkpoint_path
      self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

    if self.configs["last_activation"] == "linear":
      def keras_loss(y_true, y_pred):
        return ctc_loss_keras_2(y_true, y_pred, num_classes=self.text_featurizer.num_classes)

      self.model.compile(optimizer=self.optimizer, loss=keras_loss)
    else:
      self.model.compile(optimizer=self.optimizer, loss=ctc_loss_keras)

    callback = [Checkpoint(self.ckpt_manager)]
    if "log_dir" in self.configs.keys():
      with open(os.path.join(self.configs["log_dir"], "model.json"), "w") as f:
        f.write(self.model.to_json())
      callback.append(TimeHistory(os.path.join(self.configs["log_dir"], "time.txt")))

    self.model.fit(x=tf_train_dataset, epochs=self.configs["num_epochs"],
                   validation_data=tf_eval_dataset, shuffle="batch",
                   initial_epoch=initial_epoch, callbacks=callback)

    if model_file:
      self.save_model(model_file)

  def test(self, model_file, output_file_path):
    print("Testing model ...")
    check_key_in_dict(dictionary=self.configs,
                      keys=["test_data_transcript_paths", "tfrecords_dir"])
    test_dataset = Dataset(data_path=self.configs["test_data_transcript_paths"],
                           tfrecords_dir=self.configs["tfrecords_dir"],
                           mode="test")
    msg = self.load_model(model_file)
    if msg:
      raise Exception(msg)
    tf_test_dataset = test_dataset(text_featurizer=self.text_featurizer,
                                   speech_conf=self.configs["speech_conf"],
                                   batch_size=self.configs["batch_size"])

    def test_step(features, inp_length, transcripts):
      predictions = self.predict(features, inp_length)
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

    for feature, input_length, label, _ in tf_test_dataset:
      batch_wer, batch_wer_count, batch_cer, batch_cer_count = test_step(feature, input_length, label)
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
    check_key_in_dict(dictionary=self.configs,
                      keys=["tfrecords_dir"])
    msg = self.load_model(model_file)
    if msg:
      raise Exception(msg)
    tf_infer_dataset = Dataset(data_path=input_file_path,
                               tfrecords_dir=self.configs["tfrecords_dir"],
                               mode="infer")
    tf_infer_dataset = tf_infer_dataset(batch_size=self.configs["batch_size"],
                                        text_featurizer=self.text_featurizer,
                                        speech_conf=self.configs["speech_conf"])

    def infer_step(feature, input_length):
      prediction = self.predict(feature, input_length)
      return bytes_to_string(prediction.numpy())

    for features, inp_length in tf_infer_dataset:
      predictions = infer_step(features, inp_length)

      with open(output_file_path, "a", encoding="utf-8") as of:
        of.write("Predictions\n")
        for pred in predictions:
          of.write(pred + "\n")

  def infer_single(self, signal):
    features = compute_mfcc_feature(signal, **self.configs["speech_conf"])
    features = tf.expand_dims(features, axis=-1)
    input_length = tf.cast(tf.shape(features)[0], tf.int32)
    pred = self.predict(tf.expand_dims(features, 0), tf.expand_dims(input_length, 0))
    return bytes_to_string(pred.numpy())[0]

  def load_model(self, model_file):
    try:
      self.model = tf.saved_model.load(model_file)
    except Exception as e:
      return f"Model is not trained: {e}"
    return None

  def load_model_from_weights(self, model_file):
    try:
      self.model.load_weights(model_file)
    except Exception as e:
      return f"Model is not trained: {e}"
    return None

  @tf.function
  def predict(self, feature, input_length):
    logits = self.model(feature, training=False)
    return self.decoder(probs=logits, input_length=input_length,
                        last_activation=self.configs["last_activation"])

  def save_model(self, model_file):
    self.model.save(model_file)

  def save_from_checkpoint(self, model_file):
    self._create_checkpoints()
    if len(self.ckpt_manager.checkpoints) <= 0:
      raise ValueError("No checkpoint to save from")
    self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
    self.save_model(model_file)
