from __future__ import absolute_import

import tensorflow as tf


class RowConv1D(tf.keras.layers.Conv1D):
    def __init__(self, filters, future_context, **kwargs):
        assert 0 <= future_context, "Future context must be positive"
        super().__init__(filters=filters, kernel_size=(future_context*2 + 1), **kwargs)
        self.future_context = future_context

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        kernel_shape = self.kernel_size + (input_channel, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        # Add mask to remove weights on half of the kernel to the left (only keep future
        # context)
        left_kernel_dims = (self.future_context, input_channel, self.filters)
        left_kernel = tf.fill(dims=left_kernel_dims, value=0)
        right_kernel_dims = (self.future_context + 1, input_channel, self.filters)
        right_kernel = tf.fill(dims=right_kernel_dims, value=1)
        mask_kernel = tf.cast(tf.concat([left_kernel, right_kernel], axis=0),
                              dtype=self.dtype)
        self.kernel = tf.multiply(self.kernel, mask_kernel)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = tf.keras.layers.InputSpec(ndim=self.rank + 2,
                                                    axes={channel_axis: input_channel})

        self._build_conv_op_input_shape = input_shape
        self._build_input_channel = input_channel
        self._padding_op = self._get_padding_op()
        self._conv_op_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)
        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=self.kernel.shape,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self._padding_op,
            data_format=self._conv_op_data_format)
        self.built = True
