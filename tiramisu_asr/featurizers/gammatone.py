# Copyright 2020 Huy Le Nguyen (@usimarit) and Huy Phan (@pquochuy)
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
""" This code is inspired from https://github.com/detly/gammatone """

import numpy as np
import tensorflow as tf

pi = tf.constant(np.pi, dtype=tf.float32)


def fft_weights(
        nfft,
        fs,
        nfilts,
        width,
        fmin,
        fmax,
        maxlen):
    """
    :param nfft: the source FFT size
    :param sr: sampling rate (Hz)
    :param nfilts: the number of output bands required (default 64)
    :param width: the constant width of each band in Bark (default 1)
    :param fmin: lower limit of frequencies (Hz)
    :param fmax: upper limit of frequencies (Hz)
    :param maxlen: number of bins to truncate the rows to

    :return: a tuple `weights`, `gain` with the calculated weight matrices and
             gain vectors

    Generate a matrix of weights to combine FFT bins into Gammatone bins.

    Note about `maxlen` parameter: While wts has nfft columns, the second half
    are all zero. Hence, aud spectrum is::

        fft2gammatonemx(nfft,sr)*abs(fft(xincols,nfft))

    `maxlen` truncates the rows to this many bins.

    | (c) 2004-2009 Dan Ellis dpwe@ee.columbia.edu  based on rastamat/audspec.m
    | (c) 2012 Jason Heeris (Python implementation)
    """
    ucirc = tf.exp(1j * 2 * pi * tf.range(0, nfft / 2 + 1) / nfft)[None, ...]

    # Common ERB filter code factored out
    cf_array = erb_space(fmin, fmax, nfilts)[::-1]

    _, A11, A12, A13, A14, _, _, _, B2, gain = (
        make_erb_filters(fs, cf_array, width).T
    )

    A11, A12, A13, A14 = A11[..., None], A12[..., None], A13[..., None], A14[..., None]

    r = tf.sqrt(B2)
    theta = 2 * pi * cf_array / fs
    pole = (r * tf.exp(1j * theta))[..., None]

    GTord = 4

    weights = (
        tf.abs(ucirc + A11 * fs) * tf.abs(ucirc + A12 * fs)
        * tf.abs(ucirc + A13 * fs) * tf.abs(ucirc + A14 * fs)
        * tf.abs(fs * (pole - ucirc) * (pole.conj() - ucirc)) ** (-GTord)
        / gain[..., None]
    )

    weights = tf.pad(weights, [[0, 0], [0, nfft - weights.shape()[-1]]])

    weights = weights[:, 0:int(maxlen)]

    return weights, gain
