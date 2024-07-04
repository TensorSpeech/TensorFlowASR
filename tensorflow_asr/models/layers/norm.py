import warnings

import keras
import tensorflow as tf


def _running_with_dtensor_strategy():
    """Check whether running with a `Strategy` that is backed by DTensor.

    In the DTensor based training, all the tensors are in global context, which
    is different from the local context. Some keras components need to
    behave differently, e.g. BatchNormalization and SyncBatchNormalization, as
    well as optimizers.

    This check will help those layer to branch the logic and keep the correct
    behavior between different context.
    """
    if not tf.distribute.has_strategy():
        return False
    strategy = tf.distribute.get_strategy()
    # TODO(scottzhu): Finalize the strategy API to check if a strategy is backed
    # by DTensor.
    return getattr(strategy, "_mesh", None) is not None


def _raise_for_non_sync_bn_with_renorm_and_dtensor_strategy(
    synchronized,
    training,
    renorm,
):
    if _running_with_dtensor_strategy() and not synchronized and training and renorm:
        raise NotImplementedError(
            "Renorm for BatchNormalization under DTensor based distribution "
            "strategy is not supported at the moment. Please file a feature "
            "request if this is blocking your adoption."
        )


class BatchNormalization(keras.layers.BatchNormalization):
    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, self.compute_dtype)
        training = self._get_training_value(training)
        # Determine a boolean value for `training`: could be True, False, or
        # None.
        _raise_for_non_sync_bn_with_renorm_and_dtensor_strategy(
            synchronized=self.synchronized,
            training=training,
            renorm=self.renorm,
        )

        if self.virtual_batch_size is not None:
            # Virtual batches (aka ghost batches) can be simulated by reshaping
            # the Tensor and reusing the existing batch norm implementation
            original_shape = tf.shape(inputs)
            original_shape = tf.concat([tf.constant([-1]), original_shape[1:]], axis=0)

            if tf.__internal__.tf2.enabled():
                expanded_shape = [self.virtual_batch_size, -1] if training else [-1, 1]
                expanded_shape = tf.concat(
                    [
                        tf.constant(expanded_shape),
                        original_shape[1:],
                    ],
                    axis=0,
                )
            else:
                # Preserve incorrect legacy behavior for backwards compatibility
                expanded_shape = tf.concat(
                    [
                        tf.constant([self.virtual_batch_size, -1]),
                        original_shape[1:],
                    ],
                    axis=0,
                )

            # Will cause errors if virtual_batch_size does not divide the batch
            # size
            inputs = tf.reshape(inputs, expanded_shape)

            def undo_virtual_batching(outputs):
                outputs = tf.reshape(outputs, original_shape)
                return outputs

        if self.fused:
            outputs = self._fused_batch_norm(inputs, mask=mask, training=training)
            if self.virtual_batch_size is not None:
                # Currently never reaches here since fused_batch_norm does not
                # support virtual batching
                outputs = undo_virtual_batching(outputs)
            return outputs

        inputs_dtype = inputs.dtype.base_dtype
        if inputs_dtype in (tf.float16, tf.bfloat16):
            # Do all math in float32 if given 16-bit inputs for numeric
            # stability.  In particular, it's very easy for variance to overflow
            # in float16 and for safety we also choose to cast bfloat16 to
            # float32.
            inputs = tf.cast(inputs, tf.float32)

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        if self.virtual_batch_size is not None:
            del reduction_axes[1]  # Do not reduce along virtual batch dim

        # Broadcasting only necessary for single-axis batch norm where the axis
        # is not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if v is not None and len(v.shape) != ndims and reduction_axes != list(range(ndims - 1)):
                return tf.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

        def _compose_transforms(scale, offset, then_scale, then_offset):
            if then_scale is not None:
                scale *= then_scale
                offset *= then_scale
            if then_offset is not None:
                offset += then_offset
            return (scale, offset)

        if not training:  # noqa: E712
            mean, variance = self.moving_mean, self.moving_variance
        else:
            # The following long block are handling mean/variance update during
            # the training stage in various of different settings.
            if self.adjustment:
                adj_scale, adj_bias = self.adjustment(tf.shape(inputs))
                scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

            # Some of the computations here are not necessary when
            # training==False but not a constant. However, this makes the code
            # simpler.
            keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
            mean, variance = self._moments(
                tf.cast(inputs, self._param_dtype),
                reduction_axes,
                keep_dims=keep_dims,
                mask=mask,
            )

            if self.virtual_batch_size is not None:
                # This isn't strictly correct since in ghost batch norm, you are
                # supposed to sequentially update the moving_mean and
                # moving_variance with each sub-batch. However, since the moving
                # statistics are only used during evaluation, it is more
                # efficient to just update in one step and should not make a
                # significant difference in the result.
                new_mean = tf.reduce_mean(mean, axis=1, keepdims=True)
                new_variance = tf.reduce_mean(variance, axis=1, keepdims=True)
            else:
                if _running_with_dtensor_strategy() and not self.synchronized:
                    new_mean = tf.math.reduce_mean(mean, axis=reduction_axes)
                    new_variance = tf.math.reduce_mean(variance, axis=reduction_axes)
                else:
                    new_mean, new_variance = mean, variance

            if self._support_zero_size_input():
                # Keras assumes that batch dimension is the first dimension for
                # Batch Normalization.
                input_batch_size = tf.shape(inputs)[0]
            else:
                input_batch_size = None

            if self.renorm:
                (
                    r,
                    d,
                    new_mean,
                    new_variance,
                ) = self._renorm_correction_and_moments(new_mean, new_variance, training, input_batch_size)
                # When training, the normalized values (say, x) will be
                # transformed as x * gamma + beta without renorm, and (x * r +
                # d) * gamma + beta = x * (r * gamma) + (d * gamma + beta) with
                # renorm.
                r = _broadcast(tf.stop_gradient(r, name="renorm_r"))
                d = _broadcast(tf.stop_gradient(d, name="renorm_d"))
                scale, offset = _compose_transforms(r, d, scale, offset)

            def _do_update(var, value):
                """Compute the updates for mean and variance."""
                return self._assign_moving_average(var, value, self.momentum, input_batch_size)

            def mean_update():
                if training:
                    return _do_update(self.moving_mean, new_mean)
                return self.moving_mean

            def variance_update():
                """Update the moving variance."""

                def true_branch_renorm():
                    # We apply epsilon as part of the moving_stddev to mirror
                    # the training code path.
                    moving_stddev = _do_update(self.moving_stddev, tf.sqrt(new_variance + self.epsilon))
                    return self._assign_new_value(
                        self.moving_variance,
                        # Apply relu in case floating point rounding causes it
                        # to go negative.
                        tf.nn.relu(moving_stddev * moving_stddev - self.epsilon),
                    )

                if not training:
                    return self.moving_variance

                if self.renorm:
                    return true_branch_renorm()

                return _do_update(self.moving_variance, new_variance)

            self.add_update(mean_update)
            self.add_update(variance_update)
            # End of handling mean/variance calculation and update.

        mean = tf.cast(mean, inputs.dtype)
        variance = tf.cast(variance, inputs.dtype)
        if offset is not None:
            offset = tf.cast(offset, inputs.dtype)
        if scale is not None:
            scale = tf.cast(scale, inputs.dtype)
        outputs = tf.nn.batch_normalization(
            inputs,
            _broadcast(mean),
            _broadcast(variance),
            offset,
            scale,
            self.epsilon,
        )
        if inputs_dtype in (tf.float16, tf.bfloat16):
            outputs = tf.cast(outputs, inputs_dtype)

        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        if self.virtual_batch_size is not None:
            outputs = undo_virtual_batching(outputs)
        return outputs

    def _fused_batch_norm(self, inputs, mask, training):
        """Returns the output of fused batch norm."""
        if mask is not None:
            warnings.warn(
                "Masking is not supported with `fused=True`. "
                "You should either turn off fusing "
                "(`fused=False`) or you should not pass a `mask` "
                "argument when calling the layer. "
                "For the moment `mask` will be ignored for the "
                "normalization."
            )
        if self.center:
            beta = self.beta
        else:
            beta = tf.constant(0.0, dtype=self._param_dtype, shape=self._param_shape)
        if self.scale:
            gamma = self.gamma
        else:
            gamma = tf.constant(1.0, dtype=self._param_dtype, shape=self._param_shape)

        input_batch_size = tf.shape(inputs)[0]
        use_fused_avg_updates = False
        exponential_avg_factor = None

        def _maybe_add_or_remove_bessels_correction(variance, remove=True):
            r"""Add or remove Bessel's correction."""
            # Removes Bessel's correction if remove == True, adds it otherwise.
            # This is to be consistent with non-fused batch norm. Note that the
            # variance computed by fused batch norm is with Bessel's correction.
            # This is only used in legacy V1 batch norm tests.
            if self._bessels_correction_test_only:
                return variance
            sample_size = tf.cast(tf.size(inputs) / tf.size(variance), variance.dtype)
            if remove:
                factor = (sample_size - tf.cast(1.0, variance.dtype)) / sample_size
            else:
                factor = sample_size / (sample_size - tf.cast(1.0, variance.dtype))
            return variance * factor

        def _fused_batch_norm_training():
            return tf.compat.v1.nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                mean=self.moving_mean,
                variance=_maybe_add_or_remove_bessels_correction(self.moving_variance, remove=False),
                epsilon=self.epsilon,
                is_training=True,
                data_format=self._data_format,
                exponential_avg_factor=exponential_avg_factor,
            )

        def _fused_batch_norm_inference():
            return tf.compat.v1.nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                mean=self.moving_mean,
                variance=self.moving_variance,
                epsilon=self.epsilon,
                is_training=False,
                data_format=self._data_format,
            )

        if training:
            output, mean, variance = _fused_batch_norm_training()
        else:
            output, mean, variance = _fused_batch_norm_inference()

        variance = _maybe_add_or_remove_bessels_correction(variance, remove=True)

        if training:
            momentum = tf.convert_to_tensor(self.momentum)

            def mean_update():
                """Update self.moving_mean with the most recent data point."""
                if use_fused_avg_updates:
                    return self._assign_new_value(self.moving_mean, mean)
                return self._assign_moving_average(self.moving_mean, mean, momentum, input_batch_size)

            def variance_update():
                """Update self.moving_variance with the most recent data
                point."""
                if use_fused_avg_updates:
                    return self._assign_new_value(self.moving_variance, variance)
                return self._assign_moving_average(self.moving_variance, variance, momentum, input_batch_size)

            self.add_update(mean_update)
            self.add_update(variance_update)

        return output

    def _renorm_correction_and_moments(self, mean, variance, training, inputs_size):
        """Returns the correction and update values for renorm."""
        stddev = tf.sqrt(variance + self.epsilon)
        # Compute the average mean and standard deviation, as if they were
        # initialized with this batch's moments.
        renorm_mean = self.renorm_mean
        # Avoid divide by zero early on in training.
        renorm_stddev = tf.maximum(self.renorm_stddev, tf.sqrt(self.epsilon))
        # Compute the corrections for batch renorm.
        r = stddev / renorm_stddev
        d = (mean - renorm_mean) / renorm_stddev
        # Ensure the corrections use pre-update moving averages.
        with tf.control_dependencies([r, d]):
            mean = tf.identity(mean)
            stddev = tf.identity(stddev)
        rmin, rmax, dmax = [self.renorm_clipping.get(key) for key in ["rmin", "rmax", "dmax"]]
        if rmin is not None:
            r = tf.maximum(r, rmin)
        if rmax is not None:
            r = tf.minimum(r, rmax)
        if dmax is not None:
            d = tf.maximum(d, -dmax)
            d = tf.minimum(d, dmax)
        # When not training, use r=1, d=0.
        if not training:
            r = tf.ones_like(r)
            d = tf.zeros_like(d)

        def _update_renorm_variable(var, value, inputs_size):
            """Updates a moving average and weight, returns the unbiased
            value."""
            value = tf.identity(value)

            def _do_update():
                """Updates the var, returns the updated value."""
                new_var = self._assign_moving_average(var, value, self.renorm_momentum, inputs_size)
                return new_var

            def _fake_update():
                return tf.identity(var)

            if training:
                return _do_update()

            return _fake_update()

        # TODO(yuefengz): colocate the operations
        update_new_mean = _update_renorm_variable(self.renorm_mean, mean, inputs_size)
        update_new_stddev = _update_renorm_variable(self.renorm_stddev, stddev, inputs_size)

        # Update the inference mode moving averages with the batch value.
        with tf.control_dependencies([update_new_mean, update_new_stddev]):
            out_mean = tf.identity(mean)
            out_variance = tf.identity(variance)

        return (r, d, out_mean, out_variance)
