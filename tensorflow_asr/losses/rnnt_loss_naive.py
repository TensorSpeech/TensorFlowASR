"""
(t,u) => t * U + u
(t,u,v) => (t * maxU + u) * alphabet_size + v
"""
import numpy as np
import tensorflow as tf

from tensorflow_asr.utils import shape_util

logger = tf.get_logger()


class RnntLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        blank,
        name=None,
    ):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
        self.blank = blank
        logger.info("Use Naive implementation for RNNT loss")

    def call(self, y_true, y_pred):
        return rnnt_loss(
            logits=y_pred["logits"],
            logit_length=y_pred["logits_length"],
            labels=y_true["labels"],
            label_length=y_true["labels_length"],
            blank=self.blank,
            name=self.name,
        )


def log_sum_exp(a, b):
    return tf.cond(
        tf.logical_and(tf.less(a, 0), tf.math.is_inf(a)),  # negative inf
        true_fn=lambda: b,
        false_fn=lambda: tf.cond(
            tf.logical_and(tf.less(b, 0), tf.math.is_inf(b)),  # negative inf
            true_fn=lambda: a,
            false_fn=lambda: tf.cond(
                tf.greater(a, b),  # a > b
                true_fn=lambda: tf.math.log1p(tf.math.exp(b - a)) + a,
                false_fn=lambda: tf.math.log1p(tf.math.exp(a - b)) + b,
            ),
        ),
    )


def getitem(tensor, index):
    return tf.gather_nd(tensor, [index])[0]


def setitem(tensor, index, value):
    return tf.tensor_scatter_nd_update(tensor, [index], [value])


def compute_alphas_naive(
    batch,  # []
    logprobs,  # [B, maxT, maxU + 1, V]
    labels,  # [B, maxU]
    T,  # []
    U,  # []
    alphas,  # [B, maxT, maxU + 1]
    llforward,  # [B]
    blank,  # []
):
    alphas = setitem(alphas, [batch, 0, 0], 0)

    def _update_alphas_case_1(_t, _u, _alphas):
        update = getitem(_alphas, [batch, _t - 1, _u]) + getitem(logprobs, [batch, _t - 1, _u, blank])
        _alphas = setitem(_alphas, [batch, _t, _u], update)
        return _alphas

    def _update_alphas_case_2(_t, _u, _alphas):
        update = getitem(_alphas, [batch, _t, _u - 1]) + getitem(logprobs, [batch, _t, _u - 1, getitem(labels, [batch, _u - 1])])
        _alphas = setitem(_alphas, [batch, _t, _u], update)
        return _alphas

    def _update_alphas_case_3(_t, _u, _alphas):
        no_emit = getitem(_alphas, [batch, _t - 1, _u]) + getitem(logprobs, [batch, _t - 1, _u, blank])
        emit = getitem(_alphas, [batch, _t, _u - 1]) + getitem(logprobs, [batch, _t, _u - 1, getitem(labels, [batch, _u - 1])])
        update = log_sum_exp(emit, no_emit)
        _alphas = setitem(_alphas, [batch, _t, _u], update)
        return _alphas

    t = tf.constant(0, tf.int32)

    def _t_cond(_t, _alphas):
        return tf.less(_t, T)

    def _t_body(_t, _alphas):
        u = tf.constant(0, tf.int32)

        def _u_cond(_u, _alphas):
            return tf.less(_u, U)

        def _u_body(_u, _alphas):
            _alphas = tf.cond(
                tf.logical_and(tf.equal(_u, 0), tf.greater(_t, 0)),
                true_fn=lambda: _update_alphas_case_1(_t, _u, _alphas),
                false_fn=lambda: _alphas,
            )
            _alphas = tf.cond(
                tf.logical_and(tf.equal(_t, 0), tf.greater(_u, 0)),
                true_fn=lambda: _update_alphas_case_2(_t, _u, _alphas),
                false_fn=lambda: _alphas,
            )
            _alphas = tf.cond(
                tf.logical_and(tf.greater(_t, 0), tf.greater(_u, 0)),
                true_fn=lambda: _update_alphas_case_3(_t, _u, _alphas),
                false_fn=lambda: _alphas,
            )
            return _u + 1, _alphas

        u, _alphas = tf.while_loop(_u_cond, _u_body, loop_vars=[u, _alphas])
        return _t + 1, _alphas

    t, alphas = tf.while_loop(_t_cond, _t_body, loop_vars=[t, alphas])

    loglike = getitem(alphas, [batch, T - 1, U - 1]) + getitem(logprobs, [batch, T - 1, U - 1, blank])
    llforward = setitem(llforward, [batch], loglike)

    return alphas, llforward


def compute_betas_naive(
    batch,  # []
    logprobs,  # [B, maxT, maxU + 1, V]
    labels,  # [B, maxU]
    T,  # []
    U,  # []
    betas,  # [B, maxT, maxU + 1]
    llbackward,  # [B]
    blank,  # []
):
    betas = setitem(betas, [batch, T - 1, U - 1], getitem(logprobs, [batch, T - 1, U - 1, blank]))

    def _update_betas_case_1(_t, _u, _betas):
        update = getitem(betas, [batch, _t + 1, U - 1]) + getitem(logprobs, [batch, _t, U - 1, blank])
        _betas = setitem(_betas, [batch, _t, U - 1], update)
        return _betas

    def _update_betas_case_2(_t, _u, _betas):
        update = getitem(betas, [batch, T - 1, _u + 1]) + getitem(logprobs, [batch, T - 1, _u, getitem(labels, [batch, _u])])
        _betas = setitem(_betas, [batch, T - 1, _u], update)
        return _betas

    def _update_betas_case_3(_t, _u, _betas):
        no_emit = getitem(betas, [batch, _t + 1, _u]) + getitem(logprobs, [batch, _t, _u, blank])
        emit = getitem(betas, [batch, _t, _u + 1]) + getitem(logprobs, [batch, _t, _u, getitem(labels, [batch, _u])])
        update = log_sum_exp(emit, no_emit)
        _betas = setitem(_betas, [batch, _t, _u], update)
        return _betas

    t = T - 1

    def _t_cond(_t, _betas):
        return tf.greater_equal(_t, 0)

    def _t_body(_t, _betas):
        u = U - 1

        def _u_cond(_u, _betas):
            return tf.greater_equal(_u, 0)

        def _u_body(_u, _betas):
            _betas = tf.cond(
                tf.logical_and(tf.equal(_u, U - 1), tf.less(_t, T - 1)),
                true_fn=lambda: _update_betas_case_1(_t, _u, _betas),
                false_fn=lambda: _betas,
            )
            _betas = tf.cond(
                tf.logical_and(tf.equal(_t, T - 1), tf.less(_u, U - 1)),
                true_fn=lambda: _update_betas_case_2(_t, _u, _betas),
                false_fn=lambda: _betas,
            )
            _betas = tf.cond(
                tf.logical_and(tf.less(_t, T - 1), tf.less(_u, U - 1)),
                true_fn=lambda: _update_betas_case_3(_t, _u, _betas),
                false_fn=lambda: _betas,
            )
            return _u - 1, _betas

        u, _betas = tf.while_loop(_u_cond, _u_body, loop_vars=[u, _betas])
        return _t - 1, _betas

    t, betas = tf.while_loop(_t_cond, _t_body, loop_vars=[t, betas])

    loglike = getitem(betas, [batch, 0, 0])
    llbackward = setitem(llbackward, [batch], loglike)

    return betas, llbackward


def compute_grads_naive(
    batch,  # []
    logprobs,  # [B, maxT, maxU + 1, V]
    labels,  # [B, maxU]
    T,  # []
    U,  # []
    llforward,  # [B]
    alphas,  # [B, maxT, maxU + 1]
    betas,  # [B, maxT, maxU + 1]
    grads,  # [B, maxT, maxU + 1, V]
    blank,  # []
):
    loglike = getitem(llforward, [batch])

    def _update_grads_case_1(_t, _u, _grads):
        logp = getitem(logprobs, [batch, _t, _u, blank])
        g = -tf.exp(getitem(alphas, [batch, _t, _u]) + getitem(betas, [batch, _t + 1, _u]) + logp - loglike)
        _grads = setitem(_grads, [batch, _t, _u, blank], g)
        return _grads

    def _update_grads_case_2(_t, _u, _grads):
        logp = getitem(logprobs, [batch, _t, _u, getitem(labels, [batch, _u])])
        g = -tf.exp(getitem(alphas, [batch, _t, _u]) + getitem(betas, [batch, _t, _u + 1]) + logp - loglike)
        _grads = setitem(_grads, [batch, _t, _u, getitem(labels, [batch, _u])], g)
        return _grads

    t = tf.constant(0, tf.int32)

    def _t_grads_cond(_t, _grads):
        return tf.less(_t, T)

    def _t_grads_body(_t, _grads):
        u = tf.constant(0, tf.int32)

        def _u_grads_cond(_u, _grads):
            return tf.less(_u, U)

        def _u_grads_body(_u, _grads):
            _grads = tf.cond(
                tf.less(_t, T - 1),
                true_fn=lambda: _update_grads_case_1(_t, _u, _grads),
                false_fn=lambda: _grads,
            )
            _grads = tf.cond(
                tf.less(_u, U - 1),
                true_fn=lambda: _update_grads_case_2(_t, _u, _grads),
                false_fn=lambda: _grads,
            )
            return _u + 1, _grads

        u, _grads = tf.while_loop(_u_grads_cond, _u_grads_body, loop_vars=[u, _grads])
        return _t + 1, _grads

    t, grads = tf.while_loop(_t_grads_cond, _t_grads_body, loop_vars=[t, grads])

    grads = setitem(
        grads,
        [batch, T - 1, U - 1, blank],
        -tf.exp(getitem(logprobs, [batch, T - 1, U - 1, blank]) + getitem(alphas, [batch, T - 1, U - 1]) - loglike),
    )
    return grads


def cost_and_grad_per_batch(
    batch,  # []
    logprobs,  # [B, maxT, maxU + 1, V]
    labels,  # [B, maxU]
    label_length,  # [B]
    logit_length,  # [B]
    alphas,  # [B, maxT, maxU + 1]
    llforward,  # [B]
    betas,  # [B, maxT, maxU + 1]
    llbackward,  # [B]
    grads,  # [B, maxT, maxU + 1, V]
    blank,  # []
):
    T = tf.gather(logit_length, batch)
    U = tf.gather(label_length, batch) + 1

    alphas, llforward = compute_alphas_naive(batch, logprobs, labels, T, U, alphas, llforward, blank)
    betas, llbackward = compute_betas_naive(batch, logprobs, labels, T, U, betas, llbackward, blank)
    grads = compute_grads_naive(batch, logprobs, labels, T, U, llforward, alphas, betas, grads, blank)

    return alphas, llforward, betas, llbackward, grads


def compute_rnnt_loss_and_grad_helper(
    logits,
    labels,
    label_length,
    logit_length,
    blank,
):
    batch_size, max_t, max_u_1, _ = shape_util.shape_list(logits)
    mask = tf.cast(
        tf.sequence_mask(logit_length, max_t)[:, :, None] & tf.sequence_mask(label_length, max_u_1)[:, None, :], dtype=tf.float32
    )  # [B, maxT, maxU + 1]

    logprobs = tf.nn.log_softmax(logits)

    alphas = tf.ones([batch_size, max_t, max_u_1], dtype=tf.float32) * -np.inf * mask
    llforward = tf.ones([batch_size], dtype=tf.float32) * -np.inf

    betas = tf.ones([batch_size, max_t, max_u_1], dtype=tf.float32) * -np.inf * mask
    llbackward = tf.ones([batch_size], dtype=tf.float32) * -np.inf

    grads = tf.zeros_like(logprobs)

    b = tf.constant(0, tf.int32)

    def _b_cond(_b, _alphas, _llforward, _betas, _llbackward, _grads):
        return tf.less(_b, batch_size)

    def _b_body(_b, _alphas, _llforward, _betas, _llbackward, _grads):
        (
            _alphas,
            _llforward,
            _betas,
            _llbackward,
            _grads,
        ) = cost_and_grad_per_batch(_b, logprobs, labels, label_length, logit_length, _alphas, _llforward, _betas, _llbackward, _grads, blank)
        return _b + 1, _alphas, _llforward, _betas, _llbackward, _grads

    b, alphas, llforward, betas, llbackward, grads = tf.while_loop(_b_cond, _b_body, loop_vars=[b, alphas, llforward, betas, llbackward, grads])

    return -llforward, grads


def rnnt_loss(
    logits,
    labels,
    label_length,
    logit_length,
    blank,
    name=None,
):
    name = "rnnt_loss" if name is None else name
    with tf.name_scope(name):
        logits = tf.convert_to_tensor(logits, name="logits")
        labels = tf.convert_to_tensor(labels, name="labels")
        label_length = tf.convert_to_tensor(label_length, name="label_length")
        logit_length = tf.convert_to_tensor(logit_length, name="logit_length")
        blank = tf.convert_to_tensor(blank, name="blank")

        args = [logits, labels, label_length, logit_length, blank]

        @tf.custom_gradient
        def compute_rnnt_loss_and_grad(logits_t, labels_t, label_length_t, logit_length_t, blank_t):
            """Compute RNN-T loss and gradients."""
            logits_t.set_shape(logits.shape)
            labels_t.set_shape(labels.shape)
            label_length_t.set_shape(label_length.shape)
            logit_length_t.set_shape(logit_length.shape)
            blank_t.set_shape(blank.shape)
            kwargs = dict(
                logits=logits_t,
                labels=labels_t,
                label_length=label_length_t,
                logit_length=logit_length_t,
                blank=blank_t,
            )
            result = compute_rnnt_loss_and_grad_helper(**kwargs)

            def grad(grad_loss):
                grads = [tf.reshape(grad_loss, [-1, 1, 1, 1]) * result[1]]
                grads += [None] * (len(args) - len(grads))
                return grads

            return result[0], grad

        return compute_rnnt_loss_and_grad(*args)
