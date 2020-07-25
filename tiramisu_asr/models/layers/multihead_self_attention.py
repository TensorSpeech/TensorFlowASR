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
from tensorflow_addons.layers import MultiHeadAttention


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self,
                 head_size,
                 num_heads,
                 output_size=None,
                 dropout=0.1,
                 name="rel_pos_multihead_self_attention",
                 **kwargs):
        super(MultiHeadSelfAttention, self).__init__(name=name, **kwargs)
        self.multihead_attention = MultiHeadAttention(
            head_size=head_size,
            num_heads=num_heads,
            output_size=output_size,
            dropout=dropout
        )

    def call(self, inputs, training=False, **kwargs):
        output = self.multihead_attention([inputs, inputs], training=training)
        return output

    def get_config(self):
        conf = super(MultiHeadSelfAttention, self).get_config()
        conf.update(self.multihead_attention.get_config())
        return conf
