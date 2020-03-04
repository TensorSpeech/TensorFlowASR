from __future__ import absolute_import

import os
import tensorflow as tf

from models.CTCModel import create_ctc_model
from decoders.Decoders import create_decoder
from featurizers.SpeechFeaturizer import SpeechFeaturizer
from featurizers.TextFeaturizer import TextFeaturizer
from utils.Utils import get_config, check_key_in_dict
from data.Dataset import Dataset


class SpeechToText:
  def __init__(self, configs_path, mode="train"):
    self.mode = mode
    self.configs = get_config(configs_path)
    self.speech_featurizer = SpeechFeaturizer(
      sample_rate=self.configs["sample_rate"],
      frame_ms=self.configs["frame_ms"],
      stride_ms=self.configs["stride_ms"],
      num_feature_bins=self.configs["num_feature_bins"])
    self.text_featurizer = TextFeaturizer(
      self.configs["vocabulary_file_path"])
    self.decoder = create_decoder(
      name=self.configs["decoder"],
      index_to_token=self.text_featurizer.index_to_token,
      beam_width=self.configs["beam_width"],
      lm_path=self.configs["lm_path"])
    self.model = create_ctc_model(
      num_classes=self.text_featurizer.num_classes,
      num_feature_bins=self.speech_featurizer.num_feature_bins,
      learning_rate=self.configs["learning_rate"],
      base_model=self.configs["base_model"],
      decoder=self.decoder, mode=self.mode,
      min_lr=self.configs["min_lr"])

  def __call__(self, *args, **kwargs):
    check_key_in_dict(dictionary=kwargs, keys=["model_file"])
    if self.mode == "train":
      self.__train_and_eval(model_file=kwargs["model_file"])
    elif self.mode == "test":
      check_key_in_dict(dictionary=kwargs, keys=["output_file_path"])
      self.__test(model_file=kwargs["model_file"],
                  output_file_path=kwargs["output_file_path"])
    elif self.mode == "infer":
      check_key_in_dict(dictionary=kwargs,
                        keys=["speech_file_path",
                              "output_file_path"])
      self.__infer(speech_file_path=kwargs["speech_file_path"],
                   model_file=kwargs["model_file"],
                   output_file_path=kwargs["output_file_path"])
    elif self.mode == "infer_single":
      check_key_in_dict(dictionary=kwargs, keys=["features"])
      self.__infer_single(features=kwargs["features"],
                          model_file=kwargs["model_file"])
    elif self.mode == "infer_streaming":
      pass
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
    tf_train_dataset = train_dataset(
      speech_featurizer=self.speech_featurizer,
      text_featurizer=self.text_featurizer,
      batch_size=self.configs["batch_size"],
      augmentations=augmentations)

    tf_eval_dataset = eval_dataset(
      speech_featurizer=self.speech_featurizer,
      text_featurizer=self.text_featurizer,
      batch_size=self.configs["batch_size"])
    self.model.summary()
    checkpoint_prefix = os.path.join(self.configs["checkpoint_dir"],
                                     "ckpt_{epoch}")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_prefix,
      save_weights_only=True, verbose=1, monitor='val_loss',
      save_best_only=True, mode='min', save_freq='epoch')
    callbacks = [cp_callback]
    if "log_dir" in self.configs.keys():
      tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=self.configs["log_dir"], histogram_freq=1,
        update_freq=500, write_images=True)
      callbacks.append(tb_callback)
    latest = tf.train.latest_checkpoint(
      self.configs["checkpoint_dir"])
    if latest is not None:
      self.model.load_weights(latest)
      initial_epoch = int(latest.split("_")[-1])
    else:
      initial_epoch = 0
    self.model.fit(
      x=tf_train_dataset, epochs=self.configs["num_epochs"],
      validation_data=tf_eval_dataset, shuffle="batch",
      initial_epoch=initial_epoch,
      callbacks=callbacks)

    self.model.save_weights(filepath=model_file, save_format='tf')

  def __test(self, model_file, output_file_path):
    print("Testing model ...")
    check_key_in_dict(dictionary=self.configs, keys=[
      "test_data_transcript_paths"])
    test_dataset = Dataset(
      data_path=self.configs["test_data_transcript_paths"],
      mode="test")
    self.model.load_weights(filepath=model_file)
    tf_test_dataset = test_dataset(
      speech_featurizer=self.speech_featurizer,
      text_featurizer=self.text_featurizer,
      batch_size=self.configs["batch_size"])
    if "log_dir" in self.configs.keys():
      tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=self.configs["log_dir"], histogram_freq=1,
        update_freq=500, write_images=True)
      callbacks = [tb_callback]
    else:
      callbacks = []
    self.model.summary()
    error_rates = self.model.predict(x=tf_test_dataset,
                                     callbacks=callbacks)

    total_wer = 0
    total_cer = 0

    for er in error_rates:
      total_wer += er[0]
      total_cer += er[1]

    results = (
      total_wer / len(error_rates), total_cer / len(error_rates))
    print("WER: ", results[0])
    print("CER: ", results[-1])

    with open(output_file_path, "w", encoding="utf-8") as of:
      of.write("WER: " + str(results[0]) + "\n")
      of.write("CER: " + str(results[-1]) + "\n")

  def __infer(self, speech_file_path, model_file, output_file_path):
    print("Infering ...")
    self.model.load_weights(filepath=model_file)
    tf_infer_dataset = Dataset(data_path=speech_file_path,
                               mode="infer")
    tf_infer_dataset = tf_infer_dataset(
      speech_featurizer=self.speech_featurizer, batch_size=1)
    predictions = self.model.predict(x=tf_infer_dataset)

    print(predictions)

    with open(output_file_path, "w", encoding="utf-8") as of:
      for pred in predictions:
        of.write(pred + "\n")

  def __infer_single(self, features, model_file):
    input_length = tf.convert_to_tensor(
      features.get_shape().as_list()[0], dtype=tf.int64)
    self.model.load_weights(filepath=model_file)
    input = (
      {
        "features"    : features,
        "input_length": input_length
      },
      -1
    )
    return self.model.predict(x=input, batch_size=1)[0]

  def save_model(self, model_file):
    latest = tf.train.latest_checkpoint(
      self.configs["checkpoint_dir"])
    if latest is None:
      raise ValueError("No checkpoint found")
    self.model.load_weights(latest)
    self.model.save_weights(filepath=model_file, save_format='tf')

  def __infer_streaming(self, input_buffer,
                        model_file, output_buffer):
    pass
