# Copyright 2020 Huy Le Nguyen (@usimarit) and Huy Phan (@pquochuy)
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

from tiramisu_asr.runners.base_runners import BaseTester


class MultiConformersTester(BaseTester):

    @tf.function
    def _test_step(self, batch):
        """
        One testing step
        Args:
            batch: a step fed from test dataset

        Returns:
            (file_paths, groundtruth, greedy, beamsearch, beamsearch_lm) each has shape [B]
        """
        file_paths, lms, lgs, _, labels, _, _ = batch

        with tf.device("/CPU:0"):  # avoid copy tf.string
            labels = self.model.text_featurizer.iextract(labels)
            greed_pred = self.model.recognize([lms, lgs])
            beam_pred = self.model.recognize_beam([lms, lgs], lm=False)
            beam_lm_pred = self.model.recognize_beam([lms, lgs], lm=True)

        return file_paths, labels, greed_pred, beam_pred, beam_lm_pred
