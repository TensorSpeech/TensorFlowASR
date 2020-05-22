""" Augmentation on spectrogram: http://arxiv.org/abs/1904.08779 """
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def freq_masking(spectrogram: np.ndarray, num_freq_mask: int = 1,
                 freq_mask_param: int = 10) -> np.ndarray:
  """
  Masking the frequency channels (make features on some channel 0)
  :param spectrogram: shape (time_steps, num_feature_bins, 1)
  :param num_freq_mask: number of frequency masks, default 1
  :param freq_mask_param: parameter F of frequency masking, default 10
  :return: a tensor that's applied freq masking
  """
  assert 0 <= freq_mask_param <= spectrogram.shape[1], \
    "0 <= freq_mask_param <= num_feature_bins"
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
  for _ in range(num_time_mask):
    time = np.random.randint(0, time_mask_param + 1)
    if time > p_upperbound * spectrogram.shape[0]:
      time = int(p_upperbound * spectrogram.shape[0])
    time0 = np.random.randint(0, spectrogram.shape[0] - time)
    spectrogram[time0:time0 + time, :] = 0
  return spectrogram


@tf.function
def time_warping(spectrogram: np.ndarray, time_warp_param=None) -> np.ndarray:
  """
  Warping the spectrogram as image with 2 point along the middle
  horizontal line with a distance to the left or right
  :param spectrogram: shape (time_steps, num_feature_bins, 1)
  :param time_warp_param: parameter W of time warping, default 50
  :param direction: "left" or "right", default "right"
  :return: a tensor that's applied time warping
  """
  # Expand to shape (1, time_steps, num_feature_bins, 1)
  spectrogram = tf.expand_dims(spectrogram, axis=0)
  time_warp_param = time_warp_param if time_warp_param else spectrogram.shape[1]
  assert 0 <= time_warp_param <= spectrogram.shape[1], \
    "time_warp_param >= 0 and must not exceed time steps"
  vertical = int(spectrogram.shape[2] / 2)
  # Choose a random source point
  h_source_point = np.random.randint(0, time_warp_param)
  # Choose a random destination point
  h_dest_point = np.random.randint(0, spectrogram.shape[1])
  # Convert to tensor with dtype=float32 to avoid TypeError
  source_control_point_locations = tf.constant([[[h_source_point, vertical]]],
                                               dtype=tf.float32)
  dest_control_point_locations = tf.constant([[[h_dest_point, vertical]]],
                                             dtype=tf.float32)
  spectrogram, _ = tfa.image.sparse_image_warp(
    image=spectrogram,
    source_control_point_locations=source_control_point_locations,
    dest_control_point_locations=dest_control_point_locations,
    interpolation_order=2,
    num_boundary_points=1
  )
  spectrogram = tf.squeeze(spectrogram, axis=0)
  return spectrogram.numpy()
