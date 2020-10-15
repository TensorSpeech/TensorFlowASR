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
from ...utils.utils import merge_two_last_dims


class Merge2LastDims(tf.keras.layers.Layer):
    def __init__(self, name: str = "merge_two_last_dims", **kwargs):
        super(Merge2LastDims, self).__init__(name=name, **kwargs)

    def call(self, inputs, **kwargs):
        return merge_two_last_dims(inputs)

    def get_config(self):
        config = super(Merge2LastDims, self).get_config()
        return config
