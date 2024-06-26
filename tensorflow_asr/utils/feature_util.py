# Copyright 2020 Huy Le Nguyen (@nglehuy)
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

from tensorflow_asr import tf


def float_feature(
    list_of_floats,
):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def int64_feature(
    list_of_ints,
):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def bytestring_feature(
    list_of_bytestrings,
):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))
