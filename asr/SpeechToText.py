from __future__ import absolute_import

import os
import tempfile
import tensorflow as tf

from models.CTCModel import create_ctc_model
from decoders.Decoders import create_decoder
from featurizers.SpeechFeaturizer import SpeechFeaturizer
from featurizers.TextFeaturizer import TextFeaturizer
from utils.Utils import get_config, check_key_in_dict, \
  bytes_to_string, get_length, wer, cer
from utils.TimeHistory import TimeHistory
from data.Dataset import Dataset


class SpeechToText:
  def __init__(self, configs_path, mode="train"):
    self.mode = mode
    self.configs = get_config(configs_path)
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
    self.model = create_ctc_model(
      num_classes=self.text_featurizer.num_classes,
      num_feature_bins=self.speech_featurizer.num_feature_bins,
      learning_rate=self.configs["learning_rate"],
      min_lr=self.configs["min_lr"],
      base_model=self.configs["base_model"],
      streaming_size=self.configs["streaming_size"])

  def __call__(self, *args, **kwargs):
    if self.mode not in ["infer_single", "infer_streaming"]:
      check_key_in_dict(dictionary=kwargs, keys=["model_file"])
    if self.mode == "train":
      self.__train_and_eval(model_file=kwargs["model_file"])
    elif self.mode == "test":
      check_key_in_dict(dictionary=kwargs, keys=["output_file_path"])
      self.__test(model_file=kwargs["model_file"],
                  output_file_path=kwargs["output_file_path"])
    elif self.mode == "infer":
      check_key_in_dict(dictionary=kwargs,
                        keys=["input_file_path", "output_file_path"])
      self.__infer(input_file_path=kwargs["input_file_path"],
                   model_file=kwargs["model_file"],
                   output_file_path=kwargs["output_file_path"])
    elif self.mode in ["infer_single", "infer_streaming"]:
      check_key_in_dict(dictionary=kwargs, keys=["audio"])
      if isinstance(kwargs["audio"], str):
        return self.__infer_single(audio=kwargs["audio"])
      return self.__infer_single(audio=kwargs["audio"],
                                 sample_rate=kwargs["sample_rate"],
                                 channels=kwargs["channels"])
    else:
      raise ValueError(
        "'mode' must be either 'train', 'test', 'infer' or "
        "'infer_streaming")

  def __train_and_eval(self, model_file):
    print("Training and evaluating model ...")
    check_key_in_dict(dictionary=self.configs,
                      keys=["train_data_transcript_paths",
                            "eval_data_transcript_paths"])
    train_dataset = Dataset(
      data_path=self.configs["train_data_transcript_paths"],
      mode="train")
    eval_dataset = Dataset(
      data_path=self.configs["eval_data_transcript_paths"],
      mode="eval")
    if "augmentations" in self.configs.keys():
      augmentations = self.configs["augmentations"]

      # Augmentation must have a None element representing original
      # data
      def check_no_augment():
        for au in augmentations:
          if au is None:
            return True
        return False

      if not check_no_augment():
        augmentations.append(None)
    else:
      augmentations = [None]
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

    # Must save whole model because optimizer's state needs to be
    # reloaded when resuming training
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=os.path.join(self.configs["checkpoint_dir"],
                            "ckpt.{epoch:02d}.h5"),
      save_weights_only=False, verbose=1, monitor='val_loss',
      save_best_only=True, mode='min', save_freq='epoch')
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

    latest = tf.train.latest_checkpoint(
      self.configs["checkpoint_dir"])
    if latest is not None:
      self.model = tf.keras.models.load_model(latest)
      initial_epoch = int(latest.split(".")[1])
    else:
      initial_epoch = 0

    self.model.fit(
      x=train_dataset, epochs=self.configs["num_epochs"],
      validation_data=eval_dataset, shuffle="batch",
      initial_epoch=initial_epoch,
      callbacks=callbacks)

    self.model.save(model_file)

  def __test(self, model_file, output_file_path):
    print("Testing model ...")
    check_key_in_dict(dictionary=self.configs,
                      keys=["test_data_transcript_paths"])
    test_dataset = Dataset(
      data_path=self.configs["test_data_transcript_paths"],
      mode="test")
    self.model = tf.keras.models.load_model(model_file)
    tf_test_dataset = test_dataset(
      speech_featurizer=self.speech_featurizer,
      text_featurizer=self.text_featurizer,
      batch_size=self.configs["batch_size"])

    self.model.summary()
    logits = self.model.predict(x=tf_test_dataset)
    predictions = self.decoder.decode(
      probs=logits,
      input_length=tf.squeeze(get_length(logits), -1))

    total_wer = 0.0
    wer_count = 0.0
    total_cer = 0.0
    cer_count = 0.0

    for idx, labels in enumerate(test_dataset.entries):
      _wer, _wer_count = wer(decode=predictions[idx],
                             target=labels[-1])
      _cer, _cer_count = cer(decode=predictions[idx],
                             target=labels[-1])
      total_wer += _wer
      total_cer += _cer
      wer_count += _wer_count
      cer_count += _cer_count

    results = (total_wer / wer_count, total_cer / cer_count)
    print("WER: ", results[0])
    print("CER: ", results[-1])

    with open(output_file_path, "w", encoding="utf-8") as of:
      of.write("WER: " + str(results[0]) + "\n")
      of.write("CER: " + str(results[-1]) + "\n")

  def __infer(self, input_file_path, model_file, output_file_path):
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

  def __infer_single(self, audio, sample_rate=None, channels=None):
    if sample_rate and channels:
      features = self.speech_featurizer.compute_speech_features(
        audio, sr=sample_rate, channels=channels)
    else:
      features = self.speech_featurizer.compute_speech_features(audio)
    features = tf.expand_dims(features, axis=0)
    if self.mode == "infer_streaming":
      features = tf.pad(features,
                        [[0, 0],
                         [0, self.configs["streaming_size"] - features.shape[1]],
                         [0, 0],
                         [0, 0]],
                        "CONSTANT")
    logits = self.model.predict(x=features, batch_size=1)
    predictions = self.decoder.decode(
      probs=logits,
      input_length=tf.squeeze(get_length(logits), -1))

    return predictions[0]

  def save_infer_model(self, model_file, input_file_path):
    assert self.mode in ["infer", "infer_single", "infer_streaming"], \
      "Mode must be either infer, infer_single or infer_streaming"
    trained_model = tf.keras.models.load_model(input_file_path)
    tempdir = os.path.join(tempfile.gettempdir(), "asr.tf")
    trained_model.save_weights(tempdir)
    self.model.load_weights(tempdir)
    self.model.save(model_file)

  def load_infer_model(self, model_file):
    assert self.mode in ["infer", "infer_single", "infer_streaming"], \
      "Mode must be either infer, infer_single or infer_streaming"
    try:
      self.model = tf.keras.models.load_model(model_file)
    except Exception:
      return "Model is not trained"
    return None

  def save_infer_model_from_weights(self, model_file):
    assert self.mode in ["infer", "infer_single", "infer_streaming"], \
      "Mode must be either infer, infer_single or infer_streaming"
    self.model.save(model_file)

  def load_infer_model_from_weights(self, model_file):
    assert self.mode in ["infer", "infer_single", "infer_streaming"], \
      "Mode must be either infer, infer_single or infer_streaming"
    try:
      self.model.load_weights(model_file)
    except Exception:
      return "Model is not trained"
    return None

  def save_model(self, model_file):
    self.model.save(model_file)
