""" Augmentation on spectrogram: http://arxiv.org/abs/1904.08779 """
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def freq_masking(spectrogram: tf.Tensor, num_freq_mask: int = 1, freq_mask_param: int = 10) -> tf.Tensor:
    """
    Masking the frequency channels (make features on some channel 0)
    :param spectrogram: shape (time_steps, num_feature_bins, 1)
    :param num_freq_mask: number of frequency masks, default 1
    :param freq_mask_param: parameter F of frequency masking, default 10
    :return: a tensor that's applied freq masking
    """
    assert 0 <= freq_mask_param <= spectrogram.shape[1], "0 <= freq_mask_param <= num_feature_bins"
    spectrogram = spectrogram.numpy()  # convert to numpy to use index
    for idx in range(num_freq_mask):
        freq = np.random.randint(0, freq_mask_param)
        freq0 = np.random.randint(0, spectrogram.shape[1] - freq)
        spectrogram[:, freq0:freq0 + freq] = 0  # masking
    return tf.convert_to_tensor(spectrogram)


def time_masking(spectrogram: tf.Tensor, num_time_mask: int = 1, time_mask_param: int = 50,
                 p_upperbound: float = 1.0) -> tf.Tensor:
    """
    Masking the time steps (make features on some time steps 0)
    :param spectrogram: shape (time_steps, num_feature_bins, 1)
    :param num_time_mask: number of time masks, default 1
    :param time_mask_param: parameter W of time masking, default 50
    :param p_upperbound: an upperbound so that the number of masked time steps must not exceed p_upperbound * total_time_steps, default 1.0
    :return: a tensor that's applied time masking
    """
    assert 0.0 <= p_upperbound <= 1.0, "0.0 <= p_upperbound <= 1.0"
    spectrogram = spectrogram.numpy()  # convert to numpy to use index
    for idx in range(num_time_mask):
        time = np.random.randint(0, time_mask_param)
        if time > p_upperbound * spectrogram.shape[0]:
            time = int(p_upperbound * spectrogram.shape[0])
        time0 = np.random.randint(0, spectrogram.shape[0] - time)
        spectrogram[time0:time0 + time, :] = 0
    return tf.convert_to_tensor(spectrogram)


def time_warping(spectrogram: tf.Tensor, time_warp_param: int = 50, direction: str = "right") -> tf.Tensor:
    """
    Warping the spectrogram as image with 2 point along the middle horizontal line with a distance to the left or right
    :param spectrogram: shape (time_steps, num_feature_bins, 1)
    :param time_warp_param: parameter W of time warping, default 50
    :param direction: "left" or "right", default "right"
    :return: a tensor that's applied time warping
    """
    spectrogram = tf.expand_dims(spectrogram, axis=0)  # Expand to shape (1, time_steps, num_feature_bins, 1)
    assert direction == "left" or direction == "right", "direction must be either 'left' or 'right'"
    assert 0 <= time_warp_param <= spectrogram.shape[1], "time_warp_param >= 0 and must not exceed time steps"
    vertical = int(spectrogram.shape[2] / 2)
    # Choose a random source point
    if time_warp_param > spectrogram.shape[1] - time_warp_param:
        h_source_point = np.random.randint(spectrogram.shape[1] - time_warp_param, time_warp_param)
    elif time_warp_param < spectrogram.shape[1] - time_warp_param:
        h_source_point = np.random.randint(time_warp_param, spectrogram.shape[1] - time_warp_param)
    else:
        h_source_point = time_warp_param
    distance = np.random.randint(0, time_warp_param)
    # Calculate destination point along the direction with a distance
    if direction == "left":
        h_dest_point = h_source_point - distance
        h_dest_point = 0 if h_dest_point < 0 else h_dest_point
        h_source_point, h_dest_point = h_dest_point, h_source_point
    else:
        h_dest_point = h_source_point + distance
        h_dest_point = spectrogram.shape[1] if h_dest_point > spectrogram.shape[1] else h_dest_point
    # Convert to tensor with dtype=float32 to avoid TypeError
    source_control_point_locations = tf.constant([[[h_source_point, vertical]]], dtype=tf.float32)
    dest_control_point_locations = tf.constant([[[h_dest_point, vertical]]], dtype=tf.float32)
    spectrogram, _ = tfa.image.sparse_image_warp(
        image=spectrogram,
        source_control_point_locations=source_control_point_locations,
        dest_control_point_locations=dest_control_point_locations,
        interpolation_order=2,
        num_boundary_points=1
    )
    spectrogram = tf.squeeze(spectrogram, axis=0)
    return spectrogram
