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

import tensorflow as tf


def shape_list(x, out_type=tf.int32):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def shape_list_per_replica(x, per_replica_batch_size):
    shapes = x.shape.as_list()
    shapes[0] = int(per_replica_batch_size)
    return shapes


def get_shape_invariants(tensor):
    shapes = shape_list(tensor)
    return tf.TensorShape([i if isinstance(i, int) else None for i in shapes])


def get_float_spec(tensor):
    shape = get_shape_invariants(tensor)
    return tf.TensorSpec(shape, dtype=tf.float32)


def get_dim(tensor, i):
    """Get value of tensor shape[i] preferring static value if available."""
    return tf.compat.dimension_value(tensor.shape[i]) or tf.shape(tensor)[i]
