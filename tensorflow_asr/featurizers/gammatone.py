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

from ..utils.utils import shape_list

pi = tf.constant(np.pi, dtype=tf.complex64)

DEFAULT_FILTER_NUM = 100
DEFAULT_LOW_FREQ = 100
DEFAULT_HIGH_FREQ = 44100 / 4


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
    ucirc = tf.exp(1j * 2 * pi * tf.cast(tf.range(0, nfft / 2 + 1),
                                         tf.complex64) / nfft)[None, ...]

    # Common ERB filter code factored out
    cf_array = erb_space(fmin, fmax, nfilts)[::-1]

    _, A11, A12, A13, A14, _, _, _, B2, gain = make_erb_filters(fs, cf_array, width)

    A11, A12, A13, A14 = A11[..., None], A12[..., None], A13[..., None], A14[..., None]

    r = tf.cast(tf.sqrt(B2), tf.complex64)
    theta = 2 * pi * cf_array / fs
    pole = (r * tf.exp(1j * theta))[..., None]

    GTord = 4

    weights = (
        tf.abs(ucirc + A11 * fs) * tf.abs(ucirc + A12 * fs)
        * tf.abs(ucirc + A13 * fs) * tf.abs(ucirc + A14 * fs)
        * tf.abs(fs * (pole - ucirc) * (tf.math.conj(pole) - ucirc)) ** (-GTord)
        / tf.cast(gain[..., None], tf.float32)
    )

    weights = tf.pad(weights, [[0, 0], [0, nfft - shape_list(weights)[-1]]])

    weights = weights[:, 0:int(maxlen)]

    return tf.transpose(weights, perm=[1, 0])


def erb_point(low_freq, high_freq, fraction):
    """
    Calculates a single point on an ERB scale between ``low_freq`` and
    ``high_freq``, determined by ``fraction``. When ``fraction`` is ``1``,
    ``low_freq`` will be returned. When ``fraction`` is ``0``, ``high_freq``
    will be returned.

    ``fraction`` can actually be outside the range ``[0, 1]``, which in general
    isn't very meaningful, but might be useful when ``fraction`` is rounded a
    little above or below ``[0, 1]`` (eg. for plot axis labels).
    """
    # Change the following three parameters if you wish to use a different ERB
    # scale. Must change in MakeERBCoeffs too.
    # TODO: Factor these parameters out
    ear_q = 9.26449  # Glasberg and Moore Parameters
    min_bw = 24.7

    # All of the following expressions are derived in Apple TR #35, "An
    # Efficient Implementation of the Patterson-Holdsworth Cochlear Filter
    # Bank." See pages 33-34.
    erb_point = (
        -ear_q * min_bw
        + tf.exp(
            fraction * (
                -tf.math.log(high_freq + ear_q * min_bw)
                + tf.math.log(low_freq + ear_q * min_bw)
            )
        ) *
        (high_freq + ear_q * min_bw)
    )

    return tf.cast(erb_point, tf.complex64)


def erb_space(
        low_freq=DEFAULT_LOW_FREQ,
        high_freq=DEFAULT_HIGH_FREQ,
        num=DEFAULT_FILTER_NUM):
    """
    This function computes an array of ``num`` frequencies uniformly spaced
    between ``high_freq`` and ``low_freq`` on an ERB scale.

    For a definition of ERB, see Moore, B. C. J., and Glasberg, B. R. (1983).
    "Suggested formulae for calculating auditory-filter bandwidths and
    excitation patterns," J. Acoust. Soc. Am. 74, 750-753.
    """
    return erb_point(
        low_freq,
        high_freq,
        tf.range(1, num + 1, dtype=tf.float32) / num
    )


def make_erb_filters(fs, centre_freqs, width=1.0):
    """
    This function computes the filter coefficients for a bank of 
    Gammatone filters. These filters were defined by Patterson and Holdworth for
    simulating the cochlea. 

    The result is returned as a :class:`ERBCoeffArray`. Each row of the
    filter arrays contains the coefficients for four second order filters. The
    transfer function for these four filters share the same denominator (poles)
    but have different numerators (zeros). All of these coefficients are
    assembled into one vector that the ERBFilterBank can take apart to implement
    the filter.

    The filter bank contains "numChannels" channels that extend from
    half the sampling rate (fs) to "lowFreq". Alternatively, if the numChannels
    input argument is a vector, then the values of this vector are taken to be
    the center frequency of each desired filter. (The lowFreq argument is
    ignored in this case.)

    Note this implementation fixes a problem in the original code by
    computing four separate second order filters. This avoids a big problem with
    round off errors in cases of very small cfs (100Hz) and large sample rates
    (44kHz). The problem is caused by roundoff error when a number of poles are
    combined, all very close to the unit circle. Small errors in the eigth order
    coefficient, are multiplied when the eigth root is taken to give the pole
    location. These small errors lead to poles outside the unit circle and
    instability. Thanks to Julius Smith for leading me to the proper
    explanation.

    Execute the following code to evaluate the frequency response of a 10
    channel filterbank::

        fcoefs = MakeERBFilters(16000,10,100);
        y = ERBFilterBank([1 zeros(1,511)], fcoefs);
        resp = 20*log10(abs(fft(y')));
        freqScale = (0:511)/512*16000;
        semilogx(freqScale(1:255),resp(1:255,:));
        axis([100 16000 -60 0])
        xlabel('Frequency (Hz)'); ylabel('Filter Response (dB)');

    | Rewritten by Malcolm Slaney@Interval.  June 11, 1998.
    | (c) 1998 Interval Research Corporation
    |
    | (c) 2012 Jason Heeris (Python implementation)
    """
    T = 1 / fs
    # Change the followFreqing three parameters if you wish to use a different
    # ERB scale. Must change in ERBSpace too.
    # TODO: factor these out
    ear_q = 9.26449  # Glasberg and Moore Parameters
    min_bw = 24.7
    order = 1

    erb = width * ((centre_freqs / ear_q) ** order + min_bw ** order) ** (1 / order)
    B = 1.019 * 2 * pi * erb

    arg = 2 * centre_freqs * pi * T
    vec = tf.exp(2j * arg)

    A0 = T
    A2 = 0
    B0 = 1
    B1 = -2 * tf.cos(arg) / tf.exp(B * T)
    B2 = tf.exp(-2 * B * T)

    rt_pos = tf.cast(tf.sqrt(3 + 2 ** 1.5), tf.complex64)
    rt_neg = tf.cast(tf.sqrt(3 - 2 ** 1.5), tf.complex64)

    common = -T * tf.exp(-(B * T))

    # TODO: This could be simplified to a matrix calculation involving the
    # constant first term and the alternating rt_pos/rt_neg and +/-1 second
    # terms
    k11 = tf.cos(arg) + rt_pos * tf.sin(arg)
    k12 = tf.cos(arg) - rt_pos * tf.sin(arg)
    k13 = tf.cos(arg) + rt_neg * tf.sin(arg)
    k14 = tf.cos(arg) - rt_neg * tf.sin(arg)

    A11 = common * k11
    A12 = common * k12
    A13 = common * k13
    A14 = common * k14

    gain_arg = tf.exp(1j * arg - B * T)

    gain = tf.cast(tf.abs(
        (vec - gain_arg * k11)
        * (vec - gain_arg * k12)
        * (vec - gain_arg * k13)
        * (vec - gain_arg * k14)
        * (T * tf.exp(B * T)
           / (-1 / tf.exp(B * T) + 1 + vec * (1 - tf.exp(B * T)))
           )**4
    ), tf.complex64)

    allfilts = tf.ones_like(centre_freqs, dtype=tf.complex64)

    fcoefs = tf.stack([
        A0 * allfilts, A11, A12, A13, A14, A2 * allfilts,
        B0 * allfilts, B1, B2,
        gain
    ], axis=1)

    return tf.transpose(fcoefs, perm=[1, 0])
