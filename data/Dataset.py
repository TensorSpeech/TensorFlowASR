from __future__ import absolute_import

import tensorflow as tf
from featurizers.SpeechFeaturizer import read_raw_audio, preemphasis, interp


class Dataset:
  def __init__(self, data_path, mode="train"):
    self.data_path = data_path
    self.mode = mode

  def __call__(self, text_featurizer, sample_rate=16000, preemph=0.95, batch_size=32,
               repeat=1, augmentations=tuple([None]), sort=False):
    entries = self.__create_entries(sort)
    if self.mode == "train":
      return self.__create_dataset(entries=entries, text_featurizer=text_featurizer,
                                   sample_rate=sample_rate, preemph=preemph,
                                   batch_size=batch_size, repeat=repeat, augmentations=augmentations)
    if self.mode in ["eval", "test", "infer"]:
      return self.__create_dataset(entries=entries, text_featurizer=text_featurizer,
                                   sample_rate=sample_rate, preemph=preemph,
                                   batch_size=batch_size, augmentations=[None])
    raise ValueError("Mode must be 'train', 'eval' or 'infer'")

  def __create_entries(self, sort=False):
    lines = []
    for file_path in self.data_path:
      with tf.io.gfile.GFile(file_path, "r") as f:
        temp_lines = f.read().splitlines()
        # Skip the header of csv file
        lines += temp_lines[1:]
    # The files is "\t" seperated
    lines = [line.split("\t", 2) for line in lines]
    if sort:
      lines.sort(key=lambda item: int(item[1]))
    return [tuple(line) for line in lines]

  def __create_dataset(self, entries, text_featurizer, sample_rate,
                       batch_size, augmentations, repeat=1, preemph=None):
    if not isinstance(augmentations, list) and \
        not isinstance(augmentations, tuple):
      raise ValueError("augmentation must be a list or a tuple")

    def _gen_data():
      for audio_file, _, transcript in entries:
        for au in augmentations:
          signal = read_raw_audio(audio_file, sample_rate)
          signal = interp(signal)
          if au is not None:
            signal = au(signal=signal, sample_rate=sample_rate)
          if preemph:
            signal = preemphasis(signal, preemph)
          labels = text_featurizer.compute_label_features(transcript)
          label_length = tf.cast(tf.shape(labels)[0], tf.int32)

          yield signal, labels, label_length

    dataset = tf.data.Dataset.from_generator(
      _gen_data,
      output_types=(
        tf.float32,
        tf.int32,
        tf.int32
      ),
      output_shapes=(
        tf.TensorShape([None]),
        tf.TensorShape([None]),
        tf.TensorShape([])
      )
    )
    # Repeat and batch the dataset
    dataset = dataset.repeat(repeat)
    # # Padding the features to its max length dimensions
    dataset = dataset.padded_batch(
      batch_size=batch_size,
      padded_shapes=(
        tf.TensorShape([None]),
        tf.TensorShape([None]),
        tf.TensorShape([])
      ),
      padding_values=(
        0.,
        text_featurizer.num_classes - 1,
        0
      )
    )
    # Prefetch to improve speed of input length
    dataset = dataset.prefetch(4)
    return dataset
