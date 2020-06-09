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
""" Augmentation on spectrogram: http://arxiv.org/abs/1904.08779 """
import numpy as np


# import tensorflow as tf
# import tensorflow_addons as tfa


def freq_masking(spectrogram: np.ndarray, num_freq_mask: int = 1,
                 freq_mask_param: int = 10) -> np.ndarray:
    """
    Masking the frequency channels (make features on some channel 0)
    :param spectrogram: shape (time_steps, num_feature_bins, 1)
    :param num_freq_mask: number of frequency masks, default 1
    :param freq_mask_param: parameter F of frequency masking, default 10
    :return: a tensor that's applied freq masking
    """
    freq_mask_param = freq_mask_param \
        if 0 <= freq_mask_param <= spectrogram.shape[1] \
        else spectrogram.shape[1]
    for _ in range(num_freq_mask):
        freq = np.random.randint(0, freq_mask_param + 1)
        freq0 = np.random.randint(0, spectrogram.shape[1] - freq)
        spectrogram[:, freq0:freq0 + freq] = 0  # masking
    return spectrogram


def time_masking(spectrogram: np.ndarray, num_time_mask: int = 1,
                 time_mask_param: int = 50,
                 p_upperbound: float = 1.0) -> np.ndarray:
    """
    Masking the time steps (make features on some time steps 0)
    :param spectrogram: shape (time_steps, num_feature_bins, 1)
    :param num_time_mask: number of time masks, default 1
    :param time_mask_param: parameter W of time masking, default 50
    :param p_upperbound: an upperbound so that the number of masked time
    steps must not exceed p_upperbound * total_time_steps, default 1.0
    :return: a tensor that's applied time masking
    """
    assert 0.0 <= p_upperbound <= 1.0, "0.0 <= p_upperbound <= 1.0"
    time_mask_param = time_mask_param \
        if 0 <= time_mask_param <= spectrogram.shape[0] \
        else spectrogram.shape[0]
    for _ in range(num_time_mask):
        time = np.random.randint(0, time_mask_param + 1)
        if time > p_upperbound * spectrogram.shape[0]:
            time = int(p_upperbound * spectrogram.shape[0])
        time0 = np.random.randint(0, spectrogram.shape[0] - time)
        spectrogram[time0:time0 + time, :] = 0
    return spectrogram

# def time_warping(spectrogram: np.ndarray, time_warp_param: int = 50) -> np.ndarray:
#     """
#     Warping the spectrogram as image with 2 point along the middle
#     horizontal line with a distance to the left or right
#     :param spectrogram: shape (time_steps, num_feature_bins, 1)
#     :param time_warp_param: parameter W of time warping, default 50
#     :return: a tensor that's applied time warping
#     """
#     time_warp_param = time_warp_param \
#         if 0 <= time_warp_param <= spectrogram.shape[0] \
#         else spectrogram.shape[0]
#     vertical = int(spectrogram.shape[1])
#     # Choose a random source point
#     a = time_warp_param
#     b = spectrogram.shape[0] - time_warp_param
#     if a > b:
#         a, b = b, a
#     h_source_point = np.random.randint(a, b)
#     distance = int(np.random.uniform(0.0, time_warp_param + 1.0))
#     direction = np.random.randint(1, 3)
#     # Choose a random destination point
#     h_dest_point = h_source_point + distance if direction == 1 else h_source_point - distance
#     h_dest_point = 0 if h_dest_point < 0 else spectrogram.shape[0] if h_dest_point > spectrogram.shape[0] else h_dest_point
#     if h_source_point == h_dest_point:
#         return spectrogram
#     # Expand to shape (1, time_steps, num_feature_bins, channels)
#     spectrogram = tf.expand_dims(spectrogram, axis=0)
#     spectrogram = tf.cast(spectrogram, dtype=tf.float32)
#     # Convert to tensor with dtype=float32 to avoid TypeError
#     source_control_point_locations = tf.constant([[[h_source_point, vertical], [h_source_point, 0], [h_source_point, vertical // 2]]],
#                                                  dtype=tf.float32)
#     dest_control_point_locations = tf.constant([[[h_dest_point, vertical], [h_dest_point, 0], [h_dest_point, vertical // 2]]],
#                                                dtype=tf.float32)
#     try:
#         spectrogram, _ = tfa.image.sparse_image_warp(
#             image=spectrogram,
#             source_control_point_locations=source_control_point_locations,
#             dest_control_point_locations=dest_control_point_locations,
#             interpolation_order=2,
#             num_boundary_points=1
#         )
#     except Exception:
#         pass
#     spectrogram = tf.squeeze(spectrogram, axis=0)
#     return spectrogram.numpy()
