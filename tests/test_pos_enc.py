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
import matplotlib.pyplot as plt
from tensorflow_asr.models.layers.positional_encoding import PositionalEncodingConcat
from tensorflow_asr.models.layers.multihead_attention import RelPositionMultiHeadAttention

pos_encoding = PositionalEncodingConcat.encode(500, 144)
print(pos_encoding.shape)

plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 144))
plt.ylabel('Position')
plt.colorbar()
plt.show()

rel = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])[None, None, ...]
rel_shift = RelPositionMultiHeadAttention.relative_shift(rel)
print(tf.reduce_all(tf.equal(rel, rel_shift)))

plt.figure(figsize=(15, 5))

plt.subplot(2, 1, 1)
plt.imshow(rel[0][0])
plt.colorbar()

plt.subplot(2, 1, 2)
plt.imshow(rel_shift[0][0])
plt.colorbar()
plt.show()
