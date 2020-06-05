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
from __future__ import absolute_import

import abc
from utils.utils import preprocess_paths
from augmentations.augments import UserAugmentation


class BaseDataset(metaclass=abc.ABCMeta):
    """ Based dataset for all models """

    def __init__(self,
                 data_paths: list,
                 augmentations: dict,
                 shuffle: bool = False,
                 stage: str = "train"):
        self.data_paths = preprocess_paths(data_paths)
        self.augmentations = UserAugmentation(augmentations)
        self.shuffle = shuffle
        self.stage = stage
        self.num_samples = 0

    @abc.abstractmethod
    def parse(self, record, augment=False):
        pass

    @abc.abstractmethod
    def create(self, batch_size):
        pass
