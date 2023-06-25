# Copyright 2023 Huy Le Nguyen (@nglehuy)
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

import numpy as np
import tensorflow as tf

from tensorflow_asr.featurizers.text_featurizers import TextFeaturizer
from tensorflow_asr.utils import file_util, metric_util


class MetricLogger(tf.keras.callbacks.Callback):
    def __init__(self, text_featurizer: TextFeaturizer, predict_output: str = None):
        super().__init__()
        self.text_featurizer = text_featurizer
        self.predict_output = predict_output
        self.__reset_metrics__()

    def __reset_metrics__(self):
        # loss
        self.loss = (0, 0)
        # greedy
        self.gwer = (0, 0)
        self.gcer = (0, 0)
        # beam search
        self.bwer = (0, 0)
        self.bcer = (0, 0)

    def __agg_loss__(self, losses, logs):
        self.loss = tf.nest.map_structure(np.add, self.loss, (np.sum(losses), len(losses)))
        logs["loss"] = np.divide(self.loss[0], self.loss[1])
        return logs

    def __agg_greedy__(self, transcripts, targets, logs):
        self.gwer = tf.nest.map_structure(np.add, self.gwer, metric_util.execute_wer(transcripts, targets))
        self.gcer = tf.nest.map_structure(np.add, self.gcer, metric_util.execute_cer(transcripts, targets))
        logs["greedy_wer"] = np.divide(self.gwer[0], self.gwer[1])
        logs["greedy_cer"] = np.divide(self.gcer[0], self.gcer[1])
        return logs

    def __agg_beamsearch__(self, transcripts, targets, logs):
        self.bwer = tf.nest.map_structure(np.add, self.bwer, metric_util.execute_wer(transcripts, targets))
        self.bcer = tf.nest.map_structure(np.add, self.bcer, metric_util.execute_cer(transcripts, targets))
        logs["beamsearch_wer"] = np.divide(self.bwer[0], self.bwer[1])
        logs["beamsearch_cer"] = np.divide(self.bcer[0], self.bcer[1])
        return logs

    def on_train_begin(self, logs=None):
        self.__reset_metrics__()

    def on_train_batch_end(self, batch, logs=None):
        if logs is None:
            return
        logs = self.__agg_loss__(logs.pop("loss"), logs=logs)

    def on_test_begin(self, logs=None):
        self.__reset_metrics__()

    def on_test_batch_end(self, batch, logs=None):
        if logs is None:
            return
        losses = logs.pop("loss")
        logs = self.__agg_loss__(losses, logs=logs)
        transcripts = self.text_featurizer.detokenize(logs.pop("_gtokens")).numpy()
        targets = self.text_featurizer.detokenize(logs.pop("_labels")).numpy()
        logs = self.__agg_greedy__(transcripts, targets, logs)

    def on_epoch_end(self, epoch, logs=None):
        print(logs)

    def on_predict_begin(self, logs=None):
        self.__reset_metrics__()
        self.predict_output = file_util.preprocess_paths(self.predict_output)
        self.predict_result_file = tf.io.gfile.GFile(self.predict_output, "w")
        self.predict_result_file.write("GROUNDTRUTH\tGREEDY\tBEAMSEARCH\n")  # header

    def on_predict_batch_end(self, batch, logs=None):
        if logs is None:
            return
        gtranscripts = self.text_featurizer.detokenize(logs.pop("_gtokens")).numpy()
        btranscripts = self.text_featurizer.detokenize(logs.pop("_btokens")).numpy()
        targets = self.text_featurizer.detokenize(logs.pop("_labels")).numpy()
        logs = self.__agg_greedy__(gtranscripts, targets, logs)
        logs = self.__agg_beamsearch__(btranscripts, targets, logs)
        for i, groundtruth in enumerate(targets):
            greedy, beamsearch = gtranscripts[i], btranscripts[i]
            self.predict_result_file.write(f"{groundtruth}\t{greedy}\t{beamsearch}\n")

    def on_predict_end(self, logs=None):
        self.predict_result_file.close()
