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
import abc

from ..augmentations.augments import UserAugmentation
from ..utils.utils import preprocess_paths


class BaseDataset(metaclass=abc.ABCMeta):
    """ Based dataset for all models """

    def __init__(self,
                 data_paths: list,
                 augmentations: dict = None,
                 cache: bool = False,
                 shuffle: bool = False,
                 stage: str = "train"):
        self.data_paths = preprocess_paths(data_paths) if data_paths else []
        self.augmentations = UserAugmentation(augmentations)  # apply augmentation
        self.cache = cache  # whether to cache WHOLE transformed dataset to memory
        self.shuffle = shuffle  # whether to shuffle tf.data.Dataset
        self.stage = stage  # for defining tfrecords files
        self.total_steps = None  # for better training visualization

    @abc.abstractmethod
    def parse(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def create(self, batch_size):
        raise NotImplementedError()
