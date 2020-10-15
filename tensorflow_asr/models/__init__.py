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
import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self, name, **kwargs):
        super(Model, self).__init__(name=name, **kwargs)

    @abc.abstractmethod
    def _build(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def call(self, inputs, training=False, **kwargs):
        raise NotImplementedError()
