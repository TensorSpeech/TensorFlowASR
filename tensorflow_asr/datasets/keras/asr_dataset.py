# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from ..asr_dataset import ASRDataset, ASRTFRecordDataset, ASRSliceDataset, AUTOTUNE, TFRECORD_SHARDS
from ..base_dataset import BUFFER_SIZE
from ...featurizers.speech_featurizers import SpeechFeaturizer
from ...featurizers.text_featurizers import TextFeaturizer
from ...utils.utils import get_num_batches
from ...augmentations.augments import Augmentation


class ASRDatasetKeras(ASRDataset):
    """ Keras Dataset for ASR using Generator """

    @tf.function
    def parse(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        """
        Returns:
            path, features, input_lengths, labels, label_lengths, pred_inp
        """
        if self.use_tf: data = self.tf_preprocess(path, audio, indices)
        else: data = self.preprocess(path, audio, indices)

        path, features, input_length, label, label_length, prediction, prediction_length = data

        return (
            {
                "path": path,
                "input": features,
                "input_length": input_length,
                "prediction": prediction,
                "prediction_length": prediction_length
            },
            {
                "label": label,
                "label_length": label_length
            }
        )

    def process(self, dataset, batch_size):
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)

        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)

        # PADDED BATCH the dataset
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                {
                    "path": tf.TensorShape([]),
                    "input": tf.TensorShape(self.speech_featurizer.shape),
                    "input_length": tf.TensorShape([]),
                    "prediction": tf.TensorShape(self.text_featurizer.prepand_shape),
                    "prediction_length": tf.TensorShape([])
                },
                {
                    "label": tf.TensorShape(self.text_featurizer.shape),
                    "label_length": tf.TensorShape([])
                },
            ),
            padding_values=(
                {
                    "path": "",
                    "input": 0.,
                    "input_length": 0,
                    "prediction": self.text_featurizer.blank,
                    "prediction_length": 0
                },
                {
                    "label": self.text_featurizer.blank,
                    "label_length": 0
                }
            ),
            drop_remainder=self.drop_remainder
        )

        # PREFETCH to improve speed of input length
        dataset = dataset.prefetch(AUTOTUNE)
        self.total_steps = get_num_batches(self.total_steps, batch_size, drop_remainders=self.drop_remainder)
        return dataset


class ASRTFRecordDatasetKeras(ASRDatasetKeras, ASRTFRecordDataset):
    """ Keras Dataset for ASR using TFRecords """

    def __init__(self,
                 data_paths: list,
                 tfrecords_dir: str,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 stage: str,
                 augmentations: Augmentation = Augmentation(None),
                 tfrecords_shards: int = TFRECORD_SHARDS,
                 cache: bool = False,
                 shuffle: bool = False,
                 use_tf: bool = False,
                 drop_remainder: bool = True,
                 buffer_size: int = BUFFER_SIZE,
                 **kwargs):
        ASRTFRecordDataset.__init__(
            self, stage=stage, speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
            data_paths=data_paths, tfrecords_dir=tfrecords_dir, augmentations=augmentations, cache=cache, shuffle=shuffle,
            tfrecords_shards=tfrecords_shards, drop_remainder=drop_remainder, buffer_size=buffer_size, use_tf=use_tf
        )
        ASRDatasetKeras.__init__(
            self, stage=stage, speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
            data_paths=data_paths, augmentations=augmentations, cache=cache, shuffle=shuffle,
            drop_remainder=drop_remainder, buffer_size=buffer_size, use_tf=use_tf
        )

    @tf.function
    def parse(self, record: tf.Tensor):
        feature_description = {
            "path": tf.io.FixedLenFeature([], tf.string),
            "audio": tf.io.FixedLenFeature([], tf.string),
            "indices": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, feature_description)
        return ASRDatasetKeras.parse(self, **example)

    def process(self, dataset: tf.data.Dataset, batch_size: int):
        return ASRDatasetKeras.process(self, dataset, batch_size)


class ASRSliceDatasetKeras(ASRDatasetKeras, ASRSliceDataset):
    """ Keras Dataset for ASR using Slice """

    def __init__(self,
                 stage: str,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 data_paths: list,
                 augmentations: Augmentation = Augmentation(None),
                 cache: bool = False,
                 shuffle: bool = False,
                 use_tf: bool = False,
                 drop_remainder: bool = True,
                 buffer_size: int = BUFFER_SIZE,
                 **kwargs):
        ASRSliceDataset.__init__(
            self, stage=stage, speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
            data_paths=data_paths, augmentations=augmentations, cache=cache, shuffle=shuffle,
            drop_remainder=drop_remainder, buffer_size=buffer_size, use_tf=use_tf
        )
        ASRDatasetKeras.__init__(
            self, stage=stage, speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
            data_paths=data_paths, augmentations=augmentations, cache=cache, shuffle=shuffle,
            drop_remainder=drop_remainder, buffer_size=buffer_size, use_tf=use_tf
        )

    @tf.function
    def parse(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        return ASRDatasetKeras.parse(self, path, audio, indices)

    def process(self, dataset: tf.data.Dataset, batch_size: int):
        return ASRDatasetKeras.process(self, dataset, batch_size)
