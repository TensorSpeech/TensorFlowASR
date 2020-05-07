from __future__ import absolute_import

import os
import gc
import sys
import time
import tensorflow as tf

from models.CTCModel import create_ctc_model, ctc_loss, ctc_loss_1, create_ctc_train_model
from decoders.Decoders import create_decoder
from featurizers.TextFeaturizer import TextFeaturizer
from utils.Utils import get_asr_config, check_key_in_dict, bytes_to_string, wer, cer
from featurizers.SpeechFeaturizer import compute_mfcc_feature, preemphasis, normalize_signal, normalize_audio_feature
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

  def _create_checkpoints(self, model):
    self.ckpt = tf.train.Checkpoint(model=model,
                                    optimizer=self.optimizer)
    self.ckpt_manager = tf.train.CheckpointManager(
      self.ckpt, self.configs["checkpoint_dir"], max_to_keep=None)

  @tf.function
  def train(self, model, dataset, optimizer, loss, num_classes, epoch, num_epochs):
    for step, [features, input_length, label, label_length] in dataset.enumerate(start=1):
      start = time.time()
      with tf.GradientTape() as tape:
        y_pred = self.model(features, training=True)
        train_loss = loss(y_true=label, y_pred=y_pred,
                          input_length=input_length, label_length=label_length,
                          num_classes=num_classes)
      gradients = tape.gradient(train_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

      sys.stdout.write("\033[K")
      tf.print("\rEpoch: ", epoch, "/", num_epochs, ", step: ", step,
               ", duration: ", int(time.time() - start), "s",
               ", train_loss = ", train_loss,
               sep="", end="", output_stream=sys.stdout)

      if self.writer:
        with self.writer.as_default():
          tf.summary.scalar("train_loss", train_loss, step=(epoch * num_epochs + step))
      gc.collect()

  def validate(self, model, decoder, dataset, loss, num_classes, last_activation):
    eval_loss_count = 0
    epoch_eval_loss = 0.0
    total_wer = 0.0
    wer_count = 0.0

    @tf.function
    def val_step(features, inp_length, y_true, lab_length):
      y_pred = model(features, training=False)
      _loss = loss(y_true=y_true, y_pred=y_pred,
                   input_length=inp_length, label_length=lab_length,
                   num_classes=num_classes)
      _pred = decoder(probs=y_pred, input_length=inp_length, last_activation=last_activation)
      return _loss, _pred

    for feature, input_length, transcript, label_length in dataset:
      _val_loss, _eval_pred = val_step(feature, input_length, transcript, label_length)
      predictions = bytes_to_string(_eval_pred.numpy())
      transcripts = self.decoder.convert_to_string(transcript)

      for idx, decoded in enumerate(predictions):
        _wer, _wer_count = wer(decode=decoded, target=transcripts[idx])
        total_wer += _wer
        wer_count += _wer_count

      epoch_eval_loss += _val_loss
      eval_loss_count += 1

    epoch_eval_loss = epoch_eval_loss / eval_loss_count
    epoch_eval_wer = total_wer / wer_count
    return epoch_eval_loss, epoch_eval_wer

  def train_and_eval(self, model_file=None):
    print("Training and evaluating model ...")
    self._create_checkpoints(self.model)

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

    epochs = self.configs["num_epochs"]

    for epoch in range(initial_epoch, epochs, 1):
      epoch_eval_loss = None
      epoch_eval_wer = None
      start = time.time()

      self.train(self.model, tf_train_dataset, self.optimizer, loss,
                 self.text_featurizer.num_classes, epoch, epochs)

      print(f"\nEnd training on epoch = {epoch}")

      if tf_eval_dataset:
        print("Validating ... ", end="")
        epoch_eval_loss, epoch_eval_wer = self.validate(
          self.model, self.decoder, tf_eval_dataset, loss,
          self.text_featurizer.num_classes, self.configs["last_activation"]
        )
        print(f"val_loss = {epoch_eval_loss}, val_wer = {epoch_eval_wer}")

      self.ckpt_manager.save()
      print(f"Saved checkpoint at epoch {epoch + 1}")
      time_epoch = time.time() - start
      print(f"Time for epoch {epoch + 1} is {time_epoch} secs")

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

    train_model = create_ctc_train_model(self.model, last_activation=self.configs["last_activation"],
                                         num_classes=self.text_featurizer.num_classes)
    self._create_checkpoints(train_model)

    self.model.summary()

    initial_epoch = 0
    if self.ckpt_manager.latest_checkpoint:
      initial_epoch = int(self.ckpt_manager.latest_checkpoint.split('-')[-1])
      # restoring the latest checkpoint in checkpoint_path
      self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

    train_model.compile(optimizer=self.optimizer, loss={"ctc_loss": lambda y_true, y_pred: y_pred})

    callback = [Checkpoint(self.ckpt_manager)]
    if "log_dir" in self.configs.keys():
      with open(os.path.join(self.configs["log_dir"], "model.json"), "w") as f:
        f.write(self.model.to_json())
      callback.append(TimeHistory(os.path.join(self.configs["log_dir"], "time.txt")))
      callback.append(tf.keras.callbacks.TensorBoard(log_dir=self.configs["log_dir"]))

    train_model.fit(x=tf_train_dataset, epochs=self.configs["num_epochs"],
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

  def test_with_noise_filter(self, model_file, output_file_path):
    print("Testing model ...")
    if not self.noise_filter:
      raise ValueError("noise_filter must be defined")

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
                                   batch_size=1, feature_extraction=False)

    def test_step(signal, label):
      prediction = self.infer_single(signal)
      label = self.decoder.convert_to_string_single(label)

      print(f"Pred: {prediction}")
      print(f"Groundtruth: {label}")
      _wer, _wer_count = wer(decode=prediction, target=label)
      _cer, _cer_count = cer(decode=prediction, target=label)
      return _wer, _wer_count, _cer, _cer_count

    total_wer = 0.0
    wer_count = 0.0
    total_cer = 0.0
    cer_count = 0.0

    for signal, label in tf_test_dataset.as_numpy_iterator():
      batch_wer, batch_wer_count, batch_cer, batch_cer_count = test_step(signal, label)
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
    if self.noise_filter:
      signal = self.noise_filter.generate(signal)

    if self.configs["speech_conf"]["normalize_signal"]:
      signal = normalize_signal(signal)
    signal = preemphasis(signal, self.configs["speech_conf"]["pre_emph"])
    features = compute_mfcc_feature(signal, self.configs["speech_conf"])
    if self.configs["speech_conf"]["normalize_feature"]:
      features = normalize_audio_feature(features)

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

  def save_from_checkpoint(self, model_file, idx, is_builtin=False):
    if is_builtin:
      train_model = create_ctc_train_model(
        self.model, last_activation=self.configs["last_activation"],
        num_classes=self.text_featurizer.num_classes
      )
    else:
      train_model = self.model
    self._create_checkpoints(train_model)
    if len(self.ckpt_manager.checkpoints) <= 0:
      raise ValueError("No checkpoint to save from")
    if idx == -1:
      self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
    else:
      self.ckpt.restore(self.ckpt_manager.checkpoints[idx])
    self.save_model(model_file)
