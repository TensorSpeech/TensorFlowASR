# pylint: disable=redefined-builtin,method-hidden,invalid-overridden-method
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:29:39 2023
"""

# Copyright 2021 Alexey Tochin
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
# ==============================================================================

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Type, Union

import numpy as np
import tensorflow as tf
from cached_property import cached_property

inf = tf.constant(np.inf)


def logit_to_logproba(logit: tf.Tensor, axis: int) -> tf.Tensor:
    """Converts logits to logarithmic probabilities:
        logit_to_logproba(x) = x - log (sum along axis (exp(x))

    Args:
        logit:  tf.Tensor, dtype = tf.float32
        axis: integer, like for tf.reduce_logsumexp

    Returns:    tf.Tensor, of the same shape and size as input logit
    """
    log_probas = logit - tf.reduce_logsumexp(input_tensor=logit, axis=axis, keepdims=True)
    return log_probas


def apply_logarithmic_mask(tensor: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Masks a logarithmic representation of a tensor, namely
    1. Keeps the value of tensor unchanged for True values of mask
    2. Replace the value of tensor by -tf.inf for False values of mask

    Args:
        tensor: tf.Tensor, dtype = tf.float32 of the same shape as mask or broadcastable
        mask:   tf.Tensor, dbool = tf.float32 of the same shape as tensor or broadcastable

    Returns:    tf.Tensor, dtype = tf.float32 of the same shape as tensor
    """
    return tensor + tf.math.log(tf.cast(mask, dtype=tf.float32))


def logsumexp(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """A numerically stable version of elementwise function
        logsumexp(x, y) = log (e ** x + e ** y)

    Args:
        x:      tf.Tensor of the shape and size as y or broadcastable
        y:      tf.Tensor of the shape and size as x or broadcastable

    Returns:    tf.Tensor of the shape and size as x and y
    """
    return tf.where(
        condition=x < y,
        x=y + tf.math.softplus(x - y),
        y=tf.where(condition=x > y, x=x + tf.math.softplus(y - x), y=x + np.log(2.0)),
    )


def subexp(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """A numerically stable version of elementwise function
        subexp(x,y) := exp x - exp y

    Args:
        x:      tf.Tensor, shape broadcastable to y
        y:      tf.Tensor, shape broadcastable to x

    Returns:    tf.Tensor, shape, the same as x and y
    """
    return tf.where(
        condition=x > y,
        x=-tf.exp(x) * tf.math.expm1(y - x),
        y=tf.where(
            condition=x < y,
            x=tf.exp(y) * tf.math.expm1(x - y),
            y=tf.zeros_like(x),
        ),
    )


def unsorted_segment_logsumexp(data: tf.Tensor, segment_ids: tf.Tensor, num_segments: Union[int, tf.Tensor]) -> tf.Tensor:
    """Computes the logarithmic sum of exponents along segments of a tensor
    like other operators from tf.math.unsorted_segment_* family.

    Args:
        data:           tf.Tensor,  shape = [...] + data_dims,
        segment_ids:    tf.Tensor,  shape = [...], dtype = tf.int32
        num_segments:   tf.Tensor,  shape = [], dtype = tf.int32

    Returns:            tf.Tensor,  shape = [num_segments] + data_dims, for the same type as data
    """
    data_max = tf.math.unsorted_segment_max(data=data, segment_ids=segment_ids, num_segments=num_segments)
    data_normed = data - tf.gather(params=data_max, indices=segment_ids)
    output = data_max + tf.math.log(
        tf.math.unsorted_segment_sum(
            data=tf.exp(data_normed),
            segment_ids=segment_ids,
            num_segments=num_segments,
        )
    )
    return output


def pad_until(tensor: tf.Tensor, desired_size: Union[tf.Tensor, int], axis: int, pad_value: Union[tf.Tensor, int, float, bool] = 0) -> tf.Tensor:
    """Pads tensor until desired dimension from right,

    Args:
        tensor:         tf.Tensor, of any shape and type
        desired_size:   tf.Tensor or pythonic static integer
        axis:           pythonic static integer for pad axes
        pad_value:      tf.Tensor or pythonic numerical for padding

    Returns:            tf.Tensor, the same shape as tensor except axis that equals to desired_size.
    """
    rank = len(tensor.shape)
    if axis >= rank:
        raise ValueError()

    current_size = tf.shape(tensor)[axis]
    paddings = [[0, 0]] * axis + [[0, tf.maximum(desired_size - current_size, 0)]] + [[0, 0]] * (rank - axis - 1)
    return tf.pad(tensor=tensor, paddings=paddings, constant_values=pad_value)


def insert_zeros(tensor: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Inserts zeros into tensor before each masked element.
    For example:
    ```python
        output = insert_zeros(
            tensor =  tf.constant([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]], dtype = tf.int32),
            mask = tf.constant([[False, True, False, False, True], [False, True,  True, True,  False]]),
        )
        # -> [[1, 0, 2, 3, 4, 0, 5, 0], [10, 0, 20, 0, 30, 0, 40, 50]]
        # We insert 0s 2, 5, 20, 30, and 40 because their positions in input tensor corresponds to True value
        in mask.
    ```

    Args:
        tensor: tf.Tensor, shape = [batch, length], any type and the same shape as mask
        mask:   tf.Tensor, shape = [batch, length], dtype = tf.bool and the same shape as tensor

    Returns:    tf.Tensor, shape = [batch, length + max_num_insertions],
                where max_num_insertions is the maximal number of True values along the 0 batch dimension of mask.
                dtype = same as input tensor
    """
    batch_size = tf.shape(tensor)[0]
    length = tf.shape(mask)[1]

    delta = tf.cumsum(tf.cast(mask, dtype=tf.int32), exclusive=False, axis=1)
    max_num_insertions = tf.reduce_max(delta[:, -1])

    y, x = tf.meshgrid(tf.range(length), tf.range(batch_size))
    y = y + delta
    indices = tf.reshape(tf.stack([x, y], 2), [-1, 2])

    output = tf.scatter_nd(indices=indices, updates=tf.reshape(tensor, shape=[-1]), shape=tf.stack([batch_size, length + max_num_insertions]))

    return output


def unfold(
    init_tensor: tf.Tensor,
    iterfunc: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    num_iters: Union[int, tf.Tensor],
    d_i: int,
    element_shape: tf.TensorShape,
    swap_memory: bool = False,
    name: str = "unfold",
) -> tf.Tensor:
    """Calculates a tensor by iterations over i that is the concatenation
        for d_i = +1:
            init_tensor
            iterfunc(init_tensor, 0)
            iterfunc(iterfunc(init_tensor, 0), 1)
            ...
            ..., num_iters - 1)
            ..., num_iters - 1), num_iters)
        for d_i = -1:
            ..., 2), 1), 0)
            ..., 2), 1)
            ...
            iterfunc(iterfunc(init_tensor, num_iters - 1), num_iters - 2)
            iterfunc(init_tensor, num_iters - 1)
            init_tensor
    For example:
    ```python
        unfold(
            init_tensor=tf.constant(0),
            iterfunc=lambda x, i: x + i,
            num_iters=5,
            d_i=1,
            element_shape=tf.TensorShape([]),
        )
        # -> [0, 0, 1, 3, 6, 10]
    ```

    Args:
        init_tensor:    tf.Tensor, of any shape that is the initial value of the iterations.
        iterfunc:       tf.Tensor, tf.Tensor -> tf.Tensor, that is the iteration function
                            from and onto the same shape as init_tensor
        num_iters:      tf.Tensor or static integer that is the number of iterations
        d_i:            either +1 or -1, where
                            +1 corresponds for the iterations from 0 to num_iters inclusive
                            -1 corresponds for the iterations from num_iters to 0 inclusive
        element_shape:  tf.TensorShape([]) that is the shape of init_tensor
        swap_memory:    the same as for tf.while_loop, argument
        name:           str, local tensor names scope

    Returns:            tf.Tensor, shape = [num_iters + 1] + init_tensor.shape
                        dtype the same as init_tensor
    """
    assert d_i in {-1, 1}
    positive_direction = d_i == 1

    with tf.name_scope(name):
        num_iters = tf.convert_to_tensor(num_iters)

        tensor_array = tf.TensorArray(
            dtype=init_tensor.dtype,
            size=num_iters + 1,
            element_shape=element_shape,
            clear_after_read=False,
            infer_shape=True,
            dynamic_size=False,
        )
        tensor_array = tensor_array.write(0 if positive_direction else num_iters, init_tensor)

        def body(i, tensor_slice):
            last_value = tensor_slice.read(i if positive_direction else i + 1)
            new_value = iterfunc(last_value, i)
            tensor_slice = tensor_slice.write(i + 1 if positive_direction else i, new_value)
            return i + d_i, tensor_slice

        n = tf.constant(0, dtype=tf.int32) if positive_direction else num_iters - 1
        _, array_out = tf.while_loop(
            cond=lambda i, _: tf.constant(True),
            body=body,
            loop_vars=(n, tensor_array),
            maximum_iterations=num_iters,
            swap_memory=swap_memory,
            name="unfold_while_loop",
        )
        return array_out.stack()


def reduce_max_with_default(input_tensor: tf.Tensor, default: tf.Tensor) -> tf.Tensor:
    """A version of tf.reduce_max function that supports default values for zero size input.
    Support axis=None case only that corresponds to scalar output

    Args:
        input_tensor:   tf.Tensor, of any shape and numerical type
        default:        tf.Tensor, shape = [], dtype the same as input_tensor

    Returns:            tf.Tensor, shape = [], dtype the same as input_tensor
    """
    total_size = tf.shape(tf.reshape(input_tensor, [-1]))[0]
    return tf.where(condition=total_size > 0, x=tf.reduce_max(input_tensor), y=default)


def expand_many_dims(input: tf.Tensor, axes: List[int]) -> tf.Tensor:
    """Analogous of tf.expand_dims for multiple new dimensions.
    Like for tf.expand_dims no new memory allocated for the output tensor.

    For example:
        expand_many_dims(tf.zeros(shape=[5, 1, 3]), axes=[0, 4, 5]).shape
        # -> [1, 5, 1, 3, 1, 1]

    Args:
        input:  tf.Tensor of any rank shape and type
        axes:   list of integer that are supposed to be the indexes of new dimensions.

    Returns:    tf.Tensor of the same type an input and rank = rank(input) + len(axes)
    """
    tensor = input
    for axis in axes:
        tensor = tf.expand_dims(input=tensor, axis=axis)

    return tensor


def smart_transpose(a: tf.Tensor, perm=List[int]) -> tf.Tensor:
    """Extension of tf.transpose.
    Parameter perm may be shorter list than rank on input tensor a.
    This case all dimensions that are beyond the list perm remain unchanged.

    For example:
        smart_transpose(tf.zeros(shape=[2, 3, 4, 5, 6]), [2, 1, 0]).shape
        # -> [4, 3, 2, 5, 6]

    Args:
        a:      tf.Tensor of any rank shape and type
        perm:   list of integers like for tf.transpose but in may be shorter than the shape of a.

    Returns:    tf.Tensor of the same type and rank as th input tensor a.
    """
    if len(perm) > len(a.shape):
        raise ValueError(f"Tensor with shape '{a.shape}' cannot be reshaped to '{perm}'")

    perm_rest = list(range(len(perm), len(a.shape)))

    return tf.transpose(a=a, perm=perm + perm_rest)


def smart_reshape(tensor: tf.Tensor, shape: List[Optional[Union[int, tf.Tensor]]]) -> tf.Tensor:
    """A version of tf.reshape.
    1. The ouput tensor is always of the same rank as input tensor.
    2. The parameter shape is supposed to be a list that is smaller or equal
    than the tensor shape.
    3. The list shape may contain None, that means "keep this dimension unchanged".
    4. The list shape is appended with None value to be of the same length as the input tensor shape.
    5. Like for tf.reshape output tensor does not requre new memory for allocation.

    For example:
    ```python
        smart_reshape(
            tensor=tf.zeros(shape=[2, 3, 4, 5]),
            shape=[8, None, 1]
        )
        # -> tf.Tensor([8, 3, 1, 5])
    ```

    Args:
        tensor: tf.Tensor of any shape and type
        shape:  list of optional static of dynamic integrates

    Returns:    tf.Tensor of the same typey and rank as the input tensor
    """
    if len(shape) > len(tensor.shape):
        raise ValueError(f"Tensor with shape {tensor.shape} cannot be reshaped to {shape}.")

    shape = shape + [None] * (len(tensor.shape) - len(shape))

    original_shape = tf.shape(tensor)
    new_shape = []
    for index, dim in enumerate(shape):
        if dim is None:
            new_shape.append(original_shape[index])
        else:
            new_shape.append(dim)

    return tf.reshape(tensor=tensor, shape=new_shape)


def ctc_loss(
    labels: tf.Tensor,
    logits: tf.Tensor,
    label_length: tf.Tensor,
    logit_length: tf.Tensor,
    blank_index: Union[int, tf.Tensor],
    ctc_loss_data_cls: "Type[BaseCtcLossData]",
) -> tf.Tensor:
    """Computes a version of CTC loss from
    http://www.cs.toronto.edu/~graves/icml_2006.pdf.

    Args:
        labels:             tf.Tensor, shape = [batch, max_label_length],       dtype = tf.int32
        logits:             tf.Tensor, shape = [batch, max_length, mum_tokens], dtype = tf.float32
        label_length:       tf.Tensor, shape = [batch],                         dtype = tf.int32
        logit_length:       tf.Tensor, shape = [batch],                         dtype = tf.int32
        blank_index:        static integer >= 0
        ctc_loss_data_cls:  BaseCtcLossData class

    Returns:                tf.Tensor, shape = [batch, max_length, mum_tokens], dtype = tf.float32
    """
    log_probas = logit_to_logproba(logit=logits, axis=2)
    loss = ctc_loss_from_logproba(
        labels=labels,
        logprobas=log_probas,
        label_length=label_length,
        logit_length=logit_length,
        blank_index=blank_index,
        ctc_loss_data_cls=ctc_loss_data_cls,
    )
    return loss


def ctc_loss_from_logproba(
    labels: tf.Tensor,
    logprobas: tf.Tensor,
    label_length: tf.Tensor,
    logit_length: tf.Tensor,
    blank_index: Union[int, tf.Tensor],
    ctc_loss_data_cls: "Type[BaseCtcLossData]",
) -> tf.Tensor:
    """Computes a version of CTC loss from logarothmic probabilities considered as independent parameters.

    Args:
        labels:             tf.Tensor, shape = [batch, max_label_length],       dtype = tf.int32
        logprobas:          tf.Tensor, shape = [batch, max_length, mum_tokens], dtype = tf.float32
        label_length:       tf.Tensor, shape = [batch],                         dtype = tf.int32
        logit_length:       tf.Tensor, shape = [batch],                         dtype = tf.int32
        blank_index:        static integer >= 0
        ctc_loss_data_cls:  BaseCtcLossData class

    Returns:                tf.Tensor, shape = [batch, max_length, mum_tokens], dtype = tf.float32
    """
    loss_data = ctc_loss_data_cls(
        labels=labels,
        logprobas=tf.stop_gradient(logprobas),
        label_length=label_length,
        logit_length=logit_length,
        blank_index=blank_index,
    )

    return loss_data.forward_fn(logprobas)


class BaseCtcLossData(ABC):
    """Base class for CTC loss data."""

    def __init__(
        self,
        labels: tf.Tensor,
        logprobas: tf.Tensor,
        label_length: tf.Tensor,
        logit_length: tf.Tensor,
        blank_index: Union[int, tf.Tensor],
        swap_memory: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._logprobas = logprobas
        self._original_label = labels
        self._logit_length = logit_length
        self._original_label_length = label_length
        self.max_label_length_plus_one = tf.shape(labels)[1]
        self._verify_inputs()

        if isinstance(blank_index, (tf.Tensor, tf.Variable)):
            self._blank_index = blank_index
        else:
            self._blank_index = tf.constant(blank_index, dtype=tf.int32)

        self._swap_memory = swap_memory

    def _verify_inputs(self) -> None:
        assert len(self._logprobas.shape) == 3
        assert self._logprobas.dtype == tf.float32
        assert len(self._original_label.shape) == 2
        assert len(self._logit_length.shape) == 1
        assert len(self._original_label_length.shape) == 1

        assert self._logprobas.shape[0] == self._original_label.shape[0]
        assert self._logprobas.shape[0] == self._logit_length.shape[0]
        assert self._logprobas.shape[0] == self._original_label_length.shape[0]

    @tf.custom_gradient
    def forward_fn(self, unused_logprobas: tf.Tensor) -> tf.Tensor:
        def backprop(d_loss):
            return expand_many_dims(d_loss, axes=[1, 2]) * self.gradient_fn(unused_logprobas)

        return self.loss, backprop

    @tf.custom_gradient
    def gradient_fn(self, unused_logprobas: tf.Tensor) -> tf.Tensor:
        def backprop(d_gradient):
            output = tf.reduce_sum(input_tensor=expand_many_dims(d_gradient, axes=[1, 2]) * self.hessian_fn(unused_logprobas), axis=[3, 4])
            return output

        return self.gradient, backprop

    @tf.custom_gradient
    def hessian_fn(self, unused_logprobas: tf.Tensor) -> tf.Tensor:
        def backprop(d_hessian):
            raise NotImplementedError("Third order derivative over the ctc loss function is not implemented.")

        return self.hessian, backprop

    @cached_property
    def hessian(self) -> tf.Tensor:
        """Calculates Hessian of loss w.r.t. input logits.

        Returns: tf.Tensor, shape = [batch_size, max_logit_length, num_tokens, max_logit_length, num_tokens]
        """
        alpha_gamma_term = self.combine_transition_probabilities(a=self.alpha[:, :-1], b=self.gamma[:, 1:])
        # shape = [batch_size, max_logit_length, num_tokens, max_logit_length + 1, max_label_length + 1]
        alpha_gamma_beta_term = self.combine_transition_probabilities(a=alpha_gamma_term[:, :, :, :-1], b=self.beta[:, 1:])
        # shape = [batch_size, max_logit_length, num_tokens, max_logit_length, num_tokens]
        alpha_gamma_beta_loss_term = expand_many_dims(self.loss, axes=[1, 2, 3, 4]) + alpha_gamma_beta_term
        # shape = [batch_size, max_logit_length, num_tokens]
        logit_length_x_num_tokens = self.max_logit_length * self.num_tokens
        first_term = tf.reshape(
            tf.linalg.set_diag(
                input=tf.reshape(tensor=alpha_gamma_beta_loss_term, shape=[self.batch_size, logit_length_x_num_tokens, logit_length_x_num_tokens]),
                diagonal=tf.reshape(tensor=self.logarithmic_logproba_gradient, shape=[self.batch_size, logit_length_x_num_tokens]),
            ),
            shape=tf.shape(alpha_gamma_beta_term),
        )

        mask = expand_many_dims(input=tf.linalg.band_part(tf.ones(shape=[self.max_logit_length] * 2, dtype=tf.bool), 0, -1), axes=[0, 2, 4])
        symmetrized_first_term = tf.where(
            condition=mask,
            x=first_term,
            y=tf.transpose(first_term, [0, 3, 4, 1, 2]),
        )
        # shape = [batch_size, max_logit_length, num_tokens, max_logit_length, num_tokens]
        hessian = -tf.exp(symmetrized_first_term) + expand_many_dims(self.gradient, [3, 4]) * expand_many_dims(self.gradient, [1, 2])
        # shape = [batch_size, max_logit_length, num_tokens, max_logit_length, num_tokens]

        # Filter out samples with infinite loss
        hessian = tf.where(
            condition=expand_many_dims(self.loss == inf, [1, 2, 3, 4]),
            x=tf.zeros(shape=[1, 1, 1, 1, 1]),
            y=hessian,
        )
        # shape = [batch_size, max_logit_length, num_tokens, max_logit_length, num_tokens]

        # Filter out logits that beyond logits length
        hessian = tf.where(condition=expand_many_dims(self.logit_length_mask, axes=[2, 3, 4]), x=hessian, y=0.0)
        hessian = tf.where(condition=expand_many_dims(self.logit_length_mask, axes=[1, 2, 4]), x=hessian, y=0.0)

        return hessian

    @cached_property
    def gradient(self) -> tf.Tensor:
        # shape = [batch_size, max_logit_length, num_tokens]
        return -tf.exp(self.logarithmic_logproba_gradient)

    @cached_property
    def logarithmic_logproba_gradient(self) -> tf.Tensor:
        """Calculates logarithmic gradient of log loss w.r.t. input logarithmic probabilities.

        Returns: tf.Tensor, shape = [batch_size, max_logit_length, num_tokens]
        """
        logarithmic_logproba_gradient = tf.reshape(self.loss, [-1, 1, 1]) + self.combine_transition_probabilities(
            a=self.alpha[:, :-1], b=self.beta[:, 1:]
        )
        # shape = [batch_size, max_logit_length, num_tokens]

        # Filter out samples infinite loss
        logarithmic_logproba_gradient = tf.where(
            condition=expand_many_dims(self.loss == inf, [1, 2]),
            x=-inf,
            y=logarithmic_logproba_gradient,
        )
        # shape = [batch_size, max_logit_length, num_tokens]

        # Filter out logits that beyond logits length
        logarithmic_logproba_gradient = apply_logarithmic_mask(
            tensor=logarithmic_logproba_gradient,
            mask=tf.expand_dims(self.logit_length_mask, axis=2),
        )
        # shape = [batch_size, max_logit_length, num_tokens]

        return logarithmic_logproba_gradient

    @property
    def alpha(self) -> tf.Tensor:
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, ...]
        raise NotImplementedError()

    @property
    def beta(self) -> tf.Tensor:
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, ...]
        raise NotImplementedError()

    @property
    def gamma(self) -> tf.Tensor:
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, ...,
        #   max_logit_length + 1, max_label_length + 1, ...]
        raise NotImplementedError()

    @cached_property
    def expected_token_logproba(self) -> tf.Tensor:
        """Logarithmic probability to predict label token.

        Returns:shape = [batch_size, max_logit_length, max_label_length + 1]
        """
        label_logproba = tf.gather(
            params=self.logproba,
            indices=self.label,
            axis=2,
            batch_dims=1,
        )
        expected_token_logproba = apply_logarithmic_mask(label_logproba, tf.expand_dims(self.label_length_mask, axis=1))
        # shape = [batch_size, max_logit_length, max_label_length + 1]
        return expected_token_logproba

    @property
    @abstractmethod
    def loss(self) -> tf.Tensor:
        """Samplewise loss function value that is minus logarithmic probability to predict label sequence.

        Returns:    tf.Tensor, shape = [batch_size]
        """
        raise NotImplementedError()

    @cached_property
    def label_token_logproba(self) -> tf.Tensor:
        """shape = [batch_size, max_logit_length, max_label_length + 1]"""
        return tf.gather(
            params=self.logproba,
            indices=self.label,
            axis=2,
            batch_dims=1,
        )

    @cached_property
    def blank_logproba(self):
        """Calculates logarithmic probability to predict blank token for given logit.

        Returns:    tf.Tensor, shape = [batch_size, max_logit_length]
        """
        return self.logproba[:, :, self.blank_token_index]

    @cached_property
    def input_proba(self) -> tf.Tensor:
        """shape = [batch_size, input_logit_tensor_length, num_tokens], dtype = tf.float32"""
        return tf.exp(self.logproba)

    @cached_property
    def logproba(self) -> tf.Tensor:
        mask = tf.expand_dims(tf.sequence_mask(lengths=self._logit_length, maxlen=self.max_logit_length), 2)
        blank_logprobas = tf.reshape(tf.math.log(tf.one_hot(self.blank_token_index, self.num_tokens)), shape=[1, 1, -1])
        logprobas = tf.where(
            condition=mask,
            x=self._logprobas,
            y=blank_logprobas,
        )
        return logprobas

    '''
    def cleaned_label(self) -> tf.Tensor:
        """ shape = [batch, max_label_length + 1] """
        _ = self.max_label_length_plus_one
    '''

    @cached_property
    def cleaned_label(self):
        # Repair padding- apparently, TPU/ GPU jit cannot handle the padding here; I'm not sure why. Anyway, it does not seem necessary in our case.
        # labels = self._original_label[:, : self.max_label_length_plus_one]
        """
        labels = tf.cond(
            pred=tf.shape(self._original_label)[1] > self.max_label_length,
            true_fn=lambda: self._original_label[:, :self.max_label_length_plus_one],
            false_fn=lambda: pad_until(
                tensor=self._original_label,
                desired_size=self.max_label_length_plus_one,
                pad_value=self.pad_token_index,
                axis=1
            )
        )
        """
        # mask = tf.sequence_mask(lengths=self._original_label_length, maxlen=tf.shape(labels)[1])
        # blank_label = tf.ones_like(labels) * self.pad_token_index
        # cleaned_label = tf.where(
        #     condition=mask,
        #     x=labels,
        #     y=blank_label,
        # )
        # return cleaned_label
        cleaned_label = pad_until(
            tensor=self._original_label,
            desired_size=self.max_label_length_plus_one,
            pad_value=self.pad_token_index,
            axis=1,
        )
        cleaned_label = cleaned_label[:, : self.max_label_length_plus_one]
        return cleaned_label

    def select_from_act(self, act: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """Takes tensor of acts act_{b, a, t, u, ...} and labels label_{b,u},
        where b is the batch index, t is the logit index, and u is the label index,
        and returns for each token index k the tensor

            output_{b,a,t,k,...} = logsumexp_u act_{b,a,t,u_k,...} * kroneker_delta(u_k = label_{b,u})

        that is logarithmic sum of exponents of acts for all u_k = label_{b,u}, given b, t and k.

        Args:
            act:    tf.Tensor, shape = [batch_size, dim_a, max_logit_length, max_label_length + 1, ...]
            label:  tf.Tensor, shape = [batch_size, max_label_length + 1]

        Returns:    tf.Tensor, shape = [batch_size, max_label_length + 1, num_tokens, ...]
        """
        data = smart_transpose(a=act, perm=[0, 3, 2, 1])
        # shape = [batch_size, max_label_length + 1, max_logit_length, dim_a, ...]
        data = tf.squeeze(
            input=smart_reshape(tensor=data, shape=[1, self.batch_size * self.max_label_length_plus_one, self.max_logit_length]), axis=0
        )
        # shape = [batch_size * (max_label_length + 1), max_logit_length, dim_a, ...]

        segment_ids = tf.reshape(label + tf.expand_dims(tf.range(self.batch_size), 1) * self.num_tokens, shape=[-1])
        # shape = [batch_size * (max_label_length + 1)]
        num_segments = self.batch_size * self.num_tokens

        output = unsorted_segment_logsumexp(data=data, segment_ids=segment_ids, num_segments=num_segments)
        # shape = [batch_size * num_tokens, max_logit_length, dim_a, ...]
        output = smart_reshape(tf.expand_dims(output, 0), [self.batch_size, self.num_tokens, self.max_logit_length])
        # shape = [batch_size, num_tokens, max_logit_length, dim_a, ...]
        output = smart_transpose(output, [0, 3, 2, 1])
        # shape = [batch_size, dim_a, max_logit_length, num_tokens, ...]
        return output

    @cached_property
    def max_logit_length_plus_one(self) -> tf.Tensor:
        return self.max_logit_length + tf.constant(1, dtype=tf.int32)

    @cached_property
    def max_logit_length(self) -> tf.Tensor:
        return tf.shape(self._logprobas)[1]

    @cached_property
    def max_label_length_plus_one(self) -> tf.Tensor:
        return self.max_label_length + tf.constant(1, dtype=tf.int32)

    @cached_property
    def max_label_length(self) -> tf.Tensor:
        return reduce_max_with_default(self._original_label_length, default=tf.constant(0, dtype=tf.int32))

    @cached_property
    def pad_token_index(self) -> tf.Tensor:
        return self.blank_token_index

    @cached_property
    def num_tokens(self) -> tf.Tensor:
        return tf.shape(self._logprobas)[2]

    @cached_property
    def blank_token_index(self) -> tf.Tensor:
        return self._blank_index

    @cached_property
    def logit_length_mask(self) -> tf.Tensor:
        """shape = [batch_size, max_logit_length]"""
        return tf.sequence_mask(
            lengths=self._logit_length,
            maxlen=self.max_logit_length,
        )

    @cached_property
    def label_length_mask(self) -> tf.Tensor:
        """shape = [batch_size, max_label_length + 1], dtype = tf.bool"""
        return tf.sequence_mask(lengths=self.label_length, maxlen=self.max_label_length_plus_one)

    @property
    def label_length(self) -> tf.Tensor:
        return self._original_label_length

    @cached_property
    def preceded_label(self) -> tf.Tensor:
        """Preceded label. For example, for label "abc_" the sequence "_abc" is returned.

        Returns:    tf.Tensor, shape = [batch_size, max_label_length + 1]
        """
        return tf.roll(self.label, shift=1, axis=1)

    @cached_property
    def label(self) -> tf.Tensor:
        """shape = [batch, max_label_length + 1]"""
        return self.cleaned_label

    @cached_property
    def batch_size(self) -> tf.Tensor:
        return tf.shape(self._logprobas)[0]

    @abstractmethod
    def combine_transition_probabilities(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Given logarithmic probabilities a and b are merges like
        a, b -> log( exp a exp p exp b )
        """
        raise NotImplementedError()


def classic_ctc_loss(
    labels: tf.Tensor,
    logits: tf.Tensor,
    label_length: tf.Tensor,
    logit_length: tf.Tensor,
    blank_index: Union[int, tf.Tensor] = 0,
) -> tf.Tensor:
    """Computes CTC (Connectionist Temporal Classification) loss from
    http://www.cs.toronto.edu/~graves/icml_2006.pdf.

    Repeated non-blank labels will be merged.
    For example, predicted sequence
        a_bb_ccc_cc
    corresponds to label
        abcc
    where "_" is the blank token.

    If label length is longer then the logit length the output loss for the corresponding sample in the batch
    is +tf.inf and the gradient is 0. For example, for label "abb" at least 4 tokens are needed.
    It is because the output sequence must be at least "ab_b" in order to handle the repeated token.

    Args:
        labels:         tf.Tensor, shape = [batch, max_label_length],       dtype = tf.int32
        logits:         tf.Tensor, shape = [batch, max_length, mum_tokens], dtype = tf.float32
        label_length:   tf.Tensor, shape = [batch],                         dtype = tf.int32
        logit_length:   tf.Tensor, shape = [batch],                         dtype = tf.int32
        blank_index:    tf.Tensor or pythonic static integer between 0 <= blank_index < mum_tokens

    Returns:            tf.Tensor, shape = [batch, max_length, mum_tokens], dtype = tf.float32
    """
    return ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        blank_index=blank_index,
        ctc_loss_data_cls=ClassicCtcLossData,
    )


class ClassicCtcLossData(BaseCtcLossData):
    """Calculate loss data for CTC (Connectionist Temporal Classification) loss from
    http://www.cs.toronto.edu/~graves/icml_2006.pdf.

    This loss is actually the logarithmic likelihood for the classification task with multiple expected class.
    All predicated sequences consist of tokens (denoted like "a", "b", ... below) and the blank "_".
    The classic CTC decoding merges all repeated non-blank labels and removes the blank.
    For example, predicted sequence
        a_bb_ccc_c is decoded as "abcc".
    All predicated sequences that coincided with the label after the decoding are the expected classes
    in the logarithmic likelihood loss function.

    Implementation:

    We calculate alpha_{b,t,l,s} and beta_{b,t,l,s} that are the logarithmic probabilities similar to
    this the ones from the sited paper and defined precisely below.
    Here, b corresponds to batch, t to logit position, l to label index, and s=0,1 to state (see below for details).

    During the decoding procedure, after handling of a part of the logit sequence,
    we predict only a part of the target label tokens. We call this subsequence the in the target space as "state".
    For example, two decode label "abc" we have to decode "a" first then add "b" and move tot the state "ab" and
    then to the state "abc".

    In order to handle the token duplication swap in the classic CTC loss we extend the set of all possible labels.
    For each token sequence we define two sequences called "closed" and "open".
    For example, for label "abc" we consider its two states denoted "abc>" (closed) and "abc<" (open).
    The difference between them is in their behaviour with respect to the token appending. The rules are:
        "...a>" + "_" -> "...a>",
        "...a<" + "_" -> "...a>",
        "...a>" + "a" -> "...aa<",
        "...a<" + "a" -> "...a<",
        "...a>" + "b" -> "...ab<",
        "...a<" + "b" -> "...ab<",
    for any different tokens "a" and "b" and any token sequence denoted by "...".
    Namely, appending a token the is equal to the last one to an open state does not change this state.
    Appending a blank to a state always males this state closed.

    This is why alpha_{b,t,l,s} and beta_{b,t,l,s} in the code below are equipped with an additional index s=0,1.
    Closed states corresponds s=0 and open ones to s=1.

    In particular, the flowing identity is satisfied
        sum_s sum_l exp alpha_{b,t,l,s} * exp beta_{b,t,l,s} = loss_{b}, for any b and t
    """

    @cached_property
    def diagonal_non_blank_grad_term(self) -> tf.Tensor:
        """shape = [batch_size, max_logit_length, num_tokens]"""
        input_tensor = self.alpha[:, :-1] + self.any_to_open_diagonal_step_log_proba + tf.roll(self.beta[:, 1:, :, 1:], shift=-1, axis=2)
        # shape = [batch_size, max_logit_length, max_label_length + 1, states]
        act = tf.reduce_logsumexp(
            input_tensor=input_tensor,
            axis=3,
        )
        # shape = [batch_size, max_logit_length, max_label_length + 1]
        diagonal_non_blank_grad_term = self.select_from_act(act=act, label=self.label)
        # shape = [batch_size, max_logit_length, num_tokens]
        return diagonal_non_blank_grad_term

    @cached_property
    def horizontal_non_blank_grad_term(self) -> tf.Tensor:
        """Horizontal steps from repeated token: open alpha state to open beta state.

        Returns: shape = [batch_size, max_logit_length, num_tokens]
        """
        act = self.alpha[:, :-1, :, 1] + self.previous_label_token_log_proba + self.beta[:, 1:, :, 1]
        # shape = [batch_size, max_logit_length, max_label_length + 1]
        horizontal_non_blank_grad_term = self.select_from_act(act, self.preceded_label)
        return horizontal_non_blank_grad_term

    @cached_property
    def loss(self) -> tf.Tensor:
        """shape = [batch_size]"""
        params = tf.reduce_logsumexp(self.alpha[:, -1], -1)
        # shape = [batch_size, max_label_length + 1]
        loss = -tf.gather(
            params=params,  # shape = [batch_size, max_label_length + 1]
            indices=self.label_length,  # shape = [batch_size]
            batch_dims=1,
        )
        return loss

    @cached_property
    def gamma(self) -> tf.Tensor:
        """shape = [
            batch_size,
            max_logit_length + 1,
            max_label_length + 1,
            state,
            max_logit_length + 1,
            max_label_length + 1,
            state,
        ],
        """
        # This is to avoid InaccessibleTensorError in graph mode
        _, _, _ = self.horizontal_step_log_proba, self.any_to_open_diagonal_step_log_proba, self.diagonal_gamma

        gamma_forward_transposed = unfold(
            init_tensor=self.diagonal_gamma,
            # init_tensor=tf.tile(self.diagonal_gamma, [self.batch_size, self.max_logit_length_plus_one, 1, 1, 1, 1]),
            iterfunc=self.gamma_step,
            d_i=1,
            num_iters=self.max_logit_length,
            element_shape=tf.TensorShape([None, None, None, None, None, None]),
            name="gamma_1",
        )
        # shape = [max_logit_length + 1, batch_size, max_logit_length + 1, max_label_length + 1, state,
        #   max_label_length + 1, state]

        gamma_forward = tf.transpose(gamma_forward_transposed, [1, 2, 3, 4, 0, 5, 6])
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state,
        #   max_logit_length + 1, max_label_length + 1, state]

        mask = expand_many_dims(
            input=tf.linalg.band_part(tf.ones(shape=[self.max_logit_length_plus_one] * 2, dtype=tf.bool), 0, -1), axes=[0, 2, 3, 5, 6]
        )
        # shape = [1, max_logit_length + 1, 1, 1, max_logit_length + 1, 1, 1]
        gamma = apply_logarithmic_mask(gamma_forward, mask)
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state,
        #   max_logit_length + 1, max_label_length + 1, state]

        return gamma

    def gamma_step(
        self,
        previous_slice: tf.Tensor,
        i: tf.Tensor,
    ) -> tf.Tensor:
        """Args:
            previous_slice: tf.Tensor,
                            shape = [batch_size, max_logit_length + 1, max_label_length + 1, state,
                                max_label_length + 1, state]
            i:              tf.Tensor,
                            shape = [], 0 <= i < max_logit_length + 1

        Returns:            tf.Tensor,
                            shape = [batch_size, max_logit_length + 1, max_label_length + 1, state,
                                max_label_length + 1, state]
        """
        horizontal_step_states = expand_many_dims(self.horizontal_step_log_proba[:, i], axes=[1, 2, 3]) + tf.expand_dims(previous_slice, 5)
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state,
        #          max_label_length + 1, next_state, previous_state]
        horizontal_step = tf.reduce_logsumexp(horizontal_step_states, axis=6)
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state, max_label_length + 1, state]

        diagonal_step_log_proba = tf.reduce_logsumexp(
            expand_many_dims(self.any_to_open_diagonal_step_log_proba[:, i], axes=[1, 2, 3]) + previous_slice, axis=5
        )
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state, max_label_length + 1]

        # We move by one token because it is a diagonal step
        moved_diagonal_step_log_proba = tf.roll(diagonal_step_log_proba, shift=1, axis=4)
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state, max_label_length + 1]

        # Out state is always open:
        diagonal_step = tf.pad(
            tensor=tf.expand_dims(moved_diagonal_step_log_proba, 5),
            paddings=[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0]],
            constant_values=-np.inf,
        )
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state, max_label_length + 1, state]
        new_gamma_slice = logsumexp(
            x=horizontal_step,
            y=diagonal_step,
        )
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state, max_label_length + 1, state]

        condition = tf.reshape(tf.range(self.max_logit_length_plus_one) <= i, shape=[1, -1, 1, 1, 1, 1])
        # shape = [1, max_logit_length + 1, 1, 1, 1, 1, 1]
        output_slice = tf.where(
            condition=condition,
            x=new_gamma_slice,
            y=self.diagonal_gamma,
        )
        # shape = [batch_size, max_logit_length + 1, max_label_length + 1, state, max_label_length + 1, state]

        return output_slice

    @cached_property
    def diagonal_gamma(self) -> tf.Tensor:
        """shape = [batch_size, max_logit_length_plus_one, max_label_length + 1, state,
        max_label_length + 1, state]
        """
        diagonal_gamma = tf.math.log(
            tf.reshape(
                tensor=tf.eye(self.max_label_length_plus_one * 2, dtype=tf.float32),
                shape=[1, 1, self.max_label_length_plus_one, 2, self.max_label_length_plus_one, 2],
            )
        )
        diagonal_gamma = tf.tile(diagonal_gamma, [self.batch_size, self.max_logit_length_plus_one, 1, 1, 1, 1])
        return diagonal_gamma

    @cached_property
    def beta(self) -> tf.Tensor:
        """Calculates the beta_{b,t,l,s} that is logarithmic probability of sample 0 <= b < batch_size - 1 in the batch
        with logit subsequence from
            t, t + 1, ... max_logit_length - 2, max_logit_length - 1,
        for t < max_logit_length
        to predict the sequence of tokens
            w_max_label_length, w_{max_label_length + 1}, ... w_{max_label_length - 2}, w_{max_label_length - 1}
        for l < max_label_length
        that is either closed s=0 or open s=1.
        from label_b = [w_0, w_1, ... w_{max_label_length - 2}, w_{max_label_length - 1}].

        This logarithmic probability is calculated by iterations
            exp beta_{t-1,l} = p_horizontal_step_{t-1,l} * exp beta_{t,l} + p_diagonal_step_{t-1,l} * exp beta_{t,l+1},
        for 0 <= t < max_logit_length,
        where p_diagonal_step_{t,l} is the probability to predict label token w_l with logit l
        and p_horizontal_step_{t,l} is the probability to skip token w_l prediction with logit l, for example, with
        the blank prediction.

        Returns:    tf.Tensor,  shape = [batch_size, max_logit_length + 1, max_label_length + 1, state],
                    dtype = tf.float32
        """
        # This is to avoid InaccessibleTensorError in graph mode
        _, _ = self.horizontal_step_log_proba, self.any_to_open_diagonal_step_log_proba

        beta = unfold(
            init_tensor=self.last_beta_slice,
            iterfunc=self.beta_step,
            d_i=-1,
            num_iters=self.max_logit_length,
            element_shape=tf.TensorShape([None, None, 2]),
            name="beta",
        )
        # shape = [logit_length + 1, batch, label_length + 1, state]
        return tf.transpose(beta, [1, 0, 2, 3])

    def beta_step(self, previous_slice: tf.Tensor, i: tf.Tensor) -> tf.Tensor:
        """shape = [batch_size, max_label_length + 1, state]"""
        horizontal_step = tf.reduce_logsumexp(self.horizontal_step_log_proba[:, i] + tf.expand_dims(previous_slice, 3), 2)
        # shape = [batch_size, max_label_length + 1, state]
        diagonal_step = self.any_to_open_diagonal_step_log_proba[:, i] + tf.roll(previous_slice[:, :, 1:], shift=-1, axis=1)
        # shape = [batch_size, max_label_length + 1, state]
        new_beta_slice = logsumexp(
            x=horizontal_step,  # shape = [batch_size, max_label_length + 1, state]
            y=diagonal_step,  # shape = [batch_size, max_label_length + 1, state]
        )
        # shape = [batch_size, max_label_length + 1, state]
        return new_beta_slice

    @cached_property
    def last_beta_slice(self) -> tf.Tensor:
        """shape = [batch_size, max_label_length + 1, state]"""
        beta_last = tf.math.log(tf.one_hot(indices=self.label_length, depth=self.max_label_length_plus_one))
        beta_last = tf.tile(input=tf.expand_dims(beta_last, axis=2), multiples=[1, 1, 2])
        return beta_last

    @cached_property
    def alpha(self) -> tf.Tensor:
        """Calculates the alpha_{b,t,l,s} that is
        the logarithmic probability of sample 0 <= b < batch_size - 1 in the batch
        with logits subsequence from 0, 1, 2, ... t - 2, t - 1, for t < max_logit_length
        to predict the sequence of tokens w_0, w_1, w_2, ... w_{l-2}, w_{l-1} for l < max_label_length + 1
        that is either closed s=0 or open s=1.
        from label_b = [w_0, w_1, ... w_{max_label_length - 2}, w_{max_label_length - 1}].

        This logarithmic probability is calculated by iterations
            exp alpha_{t + 1,l} = p_horizontal_step_{t,l} * exp alpha_{t,l} + p_diagonal_step_{t,l} * exp alpha_{t,l-1},
        for 0 <= t < max_logit_length,
        where p_diagonal_step_{t,l} is the probability to predict label token w_l with logit l
        and p_horizontal_step_{t,l} is the probability to skip token w_l prediction with logit l, for example, with
        the blank prediction.

        Returns:    tf.Tensor,  shape = [batch_size, max_logit_length + 1, max_label_length + 1, state],
                    dtype = tf.float32
        """
        # This is to avoid InaccessibleTensorError in graph mode
        _, _ = self.horizontal_step_log_proba, self.any_to_open_diagonal_step_log_proba

        alpha = unfold(
            init_tensor=self.first_alpha_slice,
            iterfunc=self.alpha_step,
            d_i=1,
            num_iters=self.max_logit_length,
            element_shape=tf.TensorShape([None, None, 2]),
            name="alpha",
        )
        # shape = [logit_length + 1, batch_size, label_length + 1, state]
        return tf.transpose(alpha, [1, 0, 2, 3])

    def alpha_step(self, previous_slice: tf.Tensor, i: tf.Tensor) -> tf.Tensor:
        """Args:
            previous_slice: shape = [batch_size, max_label_length + 1, state]
            i:

        Returns:            shape = [batch_size, max_label_length + 1, state]
        """
        temp = self.horizontal_step_log_proba[:, i] + tf.expand_dims(previous_slice, 2)
        # shape = [batch_size, max_label_length + 1, next_state, previous_state]
        horizontal_step = tf.reduce_logsumexp(temp, 3)
        # shape = [batch_size, max_label_length + 1, state]
        diagonal_step_log_proba = tf.reduce_logsumexp(self.any_to_open_diagonal_step_log_proba[:, i] + previous_slice, 2)
        # shape = [batch_size, max_label_length + 1]

        # We move by one token because it is a diagonal step
        moved_diagonal_step_log_proba = tf.roll(diagonal_step_log_proba, shift=1, axis=1)
        # shape = [batch_size, max_label_length + 1]

        # Out state is always open:
        diagonal_step = tf.pad(tensor=tf.expand_dims(moved_diagonal_step_log_proba, 2), paddings=[[0, 0], [0, 0], [1, 0]], constant_values=-np.inf)
        # shape = [batch_size, max_label_length + 1, state]
        new_alpha_slice = logsumexp(
            x=horizontal_step,
            y=diagonal_step,
        )
        # shape = [batch_size, max_label_length + 1, state]
        return new_alpha_slice

    @cached_property
    def first_alpha_slice(self) -> tf.Tensor:
        """shape = [batch_size, max_label_length + 1, state]"""
        alpha_0 = tf.math.log(tf.one_hot(indices=0, depth=self.max_label_length_plus_one * 2))
        alpha_0 = tf.tile(input=tf.reshape(alpha_0, [1, -1, 2]), multiples=[self.batch_size, 1, 1])
        return alpha_0

    @cached_property
    def any_to_open_diagonal_step_log_proba(self) -> tf.Tensor:
        """Logarithmic probability to make a diagonal step from given state to an open state

        Returns:shape = [batch_size, max_logit_length, max_label_length + 1, state]
        """
        return tf.stack(values=[self.closed_to_open_diagonal_step_log_proba, self.open_to_open_diagonal_step_log_proba], axis=3)

    @cached_property
    def open_to_open_diagonal_step_log_proba(self) -> tf.Tensor:
        """Logarithmic probability to make a diagonal step from an open state to an open state
        with expected token prediction that is different from the previous one.

        Returns:shape = [batch_size, max_logit_length, max_label_length + 1]
        """
        # We check that the predicting token does not equal to previous one
        token_repetition_mask = self.label != tf.roll(self.label, shift=1, axis=1)
        # shape = [batch_size, max_label_length + 1]
        open_diagonal_step_log_proba = apply_logarithmic_mask(
            self.closed_to_open_diagonal_step_log_proba, tf.expand_dims(token_repetition_mask, axis=1)
        )
        return open_diagonal_step_log_proba

    @cached_property
    def closed_to_open_diagonal_step_log_proba(self) -> tf.Tensor:
        """Logarithmic probability to make a diagonal step from a closed state to an open state
        with expected token prediction.

        Returns:shape = [batch_size, max_logit_length, max_label_length + 1]
        """
        return self.expected_token_logproba

    @cached_property
    def horizontal_step_log_proba(self) -> tf.Tensor:
        """Calculates logarithmic probability of the horizontal step for given logit x label position.

        This is possible in two alternative cases:
        1. Blank
        2. Not blank token from previous label position.

        Returns: tf.Tensor, shape = [batch_size, max_logit_length, max_label_length + 1, next_state, previous_state]
        """
        # We map closed and open states to closed states
        blank_term = tf.tile(input=tf.expand_dims(tf.expand_dims(self.blank_logproba, 2), 3), multiples=[1, 1, self.max_label_length_plus_one, 2])
        # shape = [batch_size, max_logit_length, max_label_length + 1, 2]
        non_blank_term = tf.pad(
            tf.expand_dims(self.not_blank_horizontal_step_log_proba, 3),
            paddings=[[0, 0], [0, 0], [0, 0], [1, 0]],
            constant_values=tf.constant(-np.inf),
        )
        # shape = [batch_size, max_logit_length, max_label_length + 1, 2]
        horizontal_step_log_proba = tf.stack([blank_term, non_blank_term], axis=3)
        return horizontal_step_log_proba

    @cached_property
    def not_blank_horizontal_step_log_proba(self) -> tf.Tensor:
        """shape = [batch_size, max_logit_length, max_label_length + 1]"""
        mask = tf.reshape(1 - tf.one_hot(self.blank_token_index, depth=self.num_tokens), shape=[1, 1, -1])
        not_blank_log_proba = apply_logarithmic_mask(self.logproba, mask)
        not_blank_horizontal_step_log_proba = tf.gather(
            params=not_blank_log_proba,
            indices=tf.roll(self.label, shift=1, axis=1),
            axis=2,
            batch_dims=1,
        )
        # shape = [batch_size, max_logit_length, max_label_length + 1]
        return not_blank_horizontal_step_log_proba

    @cached_property
    def previous_label_token_log_proba(self) -> tf.Tensor:
        """Calculates the probability to predict token that preceded to label token.

        Returns:    tf.Tensor,  shape = [batch_size, max_logit_length, max_label_length + 1]
        """
        previous_label_token_log_proba = tf.gather(
            params=self.logproba,
            indices=self.preceded_label,
            axis=2,
            batch_dims=1,
        )
        # shape = [batch_size, max_logit_length, max_label_length + 1]
        return previous_label_token_log_proba

    @cached_property
    def blank_logproba(self) -> tf.Tensor:
        """shape = [batch_size, max_logit_length]"""
        return self.logproba[:, :, self.blank_token_index]

    def combine_transition_probabilities(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Transforms logarithmic transition probabilities a and b.

        Args:
            a:      shape = [batch, DIMS_A, max_logit_length, max_label_length + 1, state]
            b:      shape = [batch, max_logit_length, max_label_length + 1, state, DIMS_B]

        Returns:    shape = [batch, DIMS_A, max_logit_length, num_tokens, DIMS_B]
        """
        assert len(a.shape) >= 4
        assert len(b.shape) >= 4
        assert a.shape[-1] == 2
        assert b.shape[3] == 2

        dims_a = tf.shape(a)[1:-3]
        dims_b = tf.shape(b)[4:]
        a = tf.reshape(a, shape=[self.batch_size, -1, self.max_logit_length, self.max_label_length_plus_one, 2, 1])
        # shape = [batch_size, dims_a, max_logit_length, max_label_length + 1, state, 1]
        b = tf.reshape(b, shape=[self.batch_size, 1, self.max_logit_length, self.max_label_length_plus_one, 2, -1])
        # shape = [batch_size, 1, max_logit_length, max_label_length + 1, state, dims_b]

        # Either open or closed state from alpha and only closed state from beta
        ab_term = tf.reduce_logsumexp(a, 4) + b[:, :, :, :, 0]
        # shape = [batch_size, dims_a, max_logit_length, max_label_length + 1, dims_b]

        horizontal_blank_grad_term = expand_many_dims(self.blank_logproba, axes=[1, 3]) + tf.reduce_logsumexp(ab_term, axis=3)
        # shape = [batch_size, dims_a, max_logit_length, dims_b]

        act = a[:, :, :, :, 1] + expand_many_dims(self.previous_label_token_log_proba, axes=[1, 4]) + b[:, :, :, :, 1]
        # shape = [batch_size, dim_a, max_logit_length, max_label_length + 1, dim_b]

        horizontal_non_blank_grad_term = self.select_from_act(act, self.preceded_label)
        # shape = [batch_size, dim_a, max_logit_length, num_tokens, dim_b]

        input_tensor = a + expand_many_dims(self.any_to_open_diagonal_step_log_proba, axes=[1, 5]) + tf.roll(b[:, :, :, :, 1:], shift=-1, axis=3)
        # shape = [batch_size, dim_a, max_logit_length, max_label_length + 1, states, dim_b]

        act = tf.reduce_logsumexp(input_tensor=input_tensor, axis=4)
        # shape = [batch_size, dim_a, max_logit_length, max_label_length + 1, dim_b]

        diagonal_non_blank_grad_term = self.select_from_act(act=act, label=self.label)
        # shape = [batch_size, dim_a, max_logit_length, num_tokens, dim_b]

        non_blank_grad_term = logsumexp(horizontal_non_blank_grad_term, diagonal_non_blank_grad_term)
        # shape = [batch_size, dim_a, max_logit_length, num_tokens, dim_b]

        blank_mask = self.blank_token_index == tf.range(self.num_tokens)
        # shape = [num_tokens]

        output = tf.where(
            condition=expand_many_dims(blank_mask, axes=[0, 1, 2, 4]),
            x=tf.expand_dims(horizontal_blank_grad_term, 3),
            y=non_blank_grad_term,
        )
        # shape = [batch, dim_a, max_logit_length, num_tokens, dim_b]
        output_shape = tf.concat(
            [
                tf.expand_dims(self.batch_size, axis=0),
                dims_a,
                tf.expand_dims(self.max_logit_length, axis=0),
                tf.expand_dims(self.num_tokens, axis=0),
                dims_b,
            ],
            axis=0,
        )
        output_reshaped = tf.reshape(output, shape=output_shape)
        # shape = [batch, DIMS_A, max_logit_length, num_tokens, DIMS_B]

        return output_reshaped


def ctc_loss_tpu(
    labels: tf.Tensor,
    logits: tf.Tensor,
    label_length: tf.Tensor,
    logit_length: tf.Tensor,
    blank_index: Union[int, tf.Tensor] = 0,
) -> tf.Tensor:
    orig_dtype = logits.dtype
    if orig_dtype in (tf.float16, tf.bfloat16):
        logits = tf.cast(logits, tf.float32)
    loss = classic_ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        blank_index=blank_index,
    )
    if orig_dtype in (tf.float16, tf.bfloat16):
        loss = tf.cast(loss, orig_dtype)
    return loss
