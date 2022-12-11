import tensorflow as tf


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


def logp(
    denom,  # [B * maxT * maxU]
    flatten_acts,  # [B * maxT * maxU * V]
    maxT,  # []
    maxU,  # []
    vocab_size,  # []
    batch,  # []
    t,  # []
    u,  # []
    v,  # []
):
    col = (batch * maxT + t) * maxU + u
    return tf.gather(denom, col) + tf.gather(flatten_acts, col * vocab_size + v)


def compute_alphas_kernel_naive(
    batch,
    flatten_acts,  # [B * maxT * maxU * V]
    flatten_labels,  # [B * maxU]
    logit_length,  # [B]
    label_length,  # [B]
    maxT,  # []
    maxU,  # []
    denom,  # [B * maxT * maxU]
    alphas,  # [B * maxT * maxU]
    ll_forward,  # [B]
    vocab_size,  # []
    blank,  # []
):
    T = tf.gather(logit_length, batch)
    U = tf.gather(label_length, batch)

    label_offset = batch * maxU
    alpha_offset = batch * maxT * maxU
    alphas = tf.tensor_scatter_nd_update(alphas, [[alpha_offset]], [0])

    def _update_alphas_case_1(_t, _u, _alphas):
        update = tf.gather(_alphas, alpha_offset + (_t - 1) * maxU + _u) + logp(denom, flatten_acts, maxT, maxU, vocab_size, batch, _t - 1, 0, blank)
        _alphas = tf.tensor_scatter_nd_update(_alphas, [[alpha_offset + _t * maxU + _u]], [update])
        return _t, _u, _alphas

    def _update_alphas_case_2(_t, _u, _alphas):
        update = tf.gather(_alphas, alpha_offset + _u - 1)
        update += logp(denom, flatten_acts, maxT, maxU, vocab_size, batch, 0, _u - 1, tf.gather(flatten_labels, label_offset + _u - 1))
        _alphas = tf.tensor_scatter_nd_update(_alphas, [[alpha_offset + _u]], [update])
        return _alphas

    def _update_alphas_case_3(_t, _u, _alphas):
        no_emit = tf.gather(_alphas, alpha_offset + (_t - 1) * maxU + _u)
        no_emit += logp(denom, flatten_acts, maxT, maxU, vocab_size, batch, _t - 1, _u, blank)
        emit = tf.gather(_alphas, alpha_offset + _t * maxU + _u - 1)
        emit += logp(denom, flatten_acts, maxT, maxU, vocab_size, batch, _t, _u - 1, tf.gather(flatten_labels, label_offset + _u - 1))
        update = log_sum_exp(emit, no_emit)
        _alphas = tf.tensor_scatter_nd_update(_alphas, [[alpha_offset + _t * maxU + _u]], [update])
        return _alphas

    t = tf.constant(0, tf.int32)

    def _t_cond(_t, _T, U, _alphas):
        return tf.less(_t, _T)

    def _t_body(_t, _T, U, _alphas):
        u = tf.constant(0, tf.int32)

        def _u_cond(_t, _u, _U, _alphas):
            return tf.less(_u, _U)

        def _u_body(_t, _u, _U, _alphas):
            _t, _u, _alphas = tf.cond(
                tf.logical_and(tf.equal(_u, 0), tf.greater(_t, 0)),
                true_fn=_update_alphas_case_1,
                false_fn=lambda: (_t, _u, _alphas),
            )
            _t, _u, _alphas = tf.cond(
                tf.logical_and(tf.equal(_t, 0), tf.greater(_u, 0)),
                true_fn=_update_alphas_case_2,
                false_fn=lambda: (_t, _u, _alphas),
            )
            _t, _u, _alphas = tf.cond(
                tf.logical_and(tf.greater(_t, 0), tf.greater(_u, 0)),
                true_fn=_update_alphas_case_3,
                false_fn=lambda: (_t, _u, _alphas),
            )
            return _t, _u + 1, _U, _alphas

        _t, u, U, _alphas = tf.while_loop(_u_cond, _u_body, loop_vars=[_t, u, U, _alphas])
        return _t + 1, u, U, _alphas

    t, T, U, alphas = tf.while_loop(_t_cond, _t_body, loop_vars=[t, T, U, alphas])

    loglike = tf.gather(alphas, alpha_offset + (T - 1) * maxU + U - 1) + logp(denom, flatten_acts, maxT, maxU, vocab_size, batch, T - 1, U - 1, blank)
    ll_forward = tf.tensor_scatter_nd_update(ll_forward, [[batch]], [loglike])
    return alphas, ll_forward


def compute_betas_kernel_naive(
    batch,
    flatten_acts,  # [B * maxT * maxU * V]
    flatten_labels,  # [B * maxU]
    logit_length,  # [B]
    label_length,  # [B]
    maxT,  # []
    maxU,  # []
    denom,  # [B * maxT * maxU]
    betas,  # [B * maxT * maxU]
    ll_backward,  # [B]
    vocab_size,  # []
    blank,  # []
):
    T = tf.gather(logit_length, batch)
    U = tf.gather(label_length, batch)

    label_offset = batch * maxU
    beta_offset = batch * maxT * maxU
    betas = tf.tensor_scatter_nd_update(
        betas, [[beta_offset + (T - 1) * maxU + U - 1]], [logp(denom, flatten_acts, maxT, maxU, vocab_size, batch, T - 1, U - 1, blank)]
    )

    def _update_betas_case_1(_t, _u, _betas):
        update = tf.gather(betas, beta_offset + (_t + 1) * maxU + U - 1)
        update += logp(denom, flatten_acts, maxT, maxU, vocab_size, batch, _t, U - 1, blank)
        _betas = tf.tensor_scatter_nd_update(_betas, [[beta_offset + _t * maxU + U - 1]], [update])
        return _t, _u, _betas

    def _update_betas_case_2(_t, _u, _betas):
        update = tf.gather(betas, beta_offset + (T - 1) * maxU + _u + 1)
        update += logp(denom, flatten_acts, maxT, maxU, vocab_size, batch, T - 1, _u, tf.gather(flatten_labels, label_offset + _u))
        _betas = tf.tensor_scatter_nd_update(_betas, [[beta_offset + (T - 1) * maxU + _u]], [update])
        return _t, _u, _betas

    def _update_betas_case_3(_t, _u, _betas):
        no_emit = tf.gather(betas, beta_offset + (_t + 1) * maxU + _u)
        no_emit += logp(denom, flatten_acts, maxT, maxU, vocab_size, batch, _t, _u, blank)
        emit = tf.gather(betas, beta_offset + _t * maxU + _u + 1)
        emit += logp(denom, flatten_acts, maxT, maxU, vocab_size, batch, _t, _u, tf.gather(flatten_labels, label_offset + _u))
        update = log_sum_exp(emit, no_emit)
        _betas = tf.tensor_scatter_nd_update(_betas, [[beta_offset + _t * maxU + _u]], [update])
        return _t, _u, _betas

    t = tf.constant(T - 1, tf.int32)

    def _t_cond(_t, _T, _U, _betas):
        return tf.greater_equal(_t, 0)

    def _t_body(_t, _T, _U, _betas):
        u = tf.constant(_U - 1, tf.int32)

        def _u_cond(_t, _u, _T, _U, _betas):
            return tf.greater_equal(_u, 0)

        def _u_body(_t, _u, _T, _U, _betas):
            _t, _u, _betas = tf.cond(
                tf.logical_and(tf.equal(_u, _U - 1), tf.less(_t, _T - 1)),
                true_fn=_update_betas_case_1,
                false_fn=lambda: (_t, _u, _betas),
            )
            _t, _u, _betas = tf.cond(
                tf.logical_and(tf.equal(_t, _T - 1), tf.less(_u, _U - 1)),
                true_fn=_update_betas_case_2,
                false_fn=lambda: (_t, _u, _betas),
            )
            _t, _u, _betas = tf.cond(
                tf.logical_and(tf.less(_t, _T - 1), tf.less(_u, _U - 1)),
                true_fn=_update_betas_case_3,
                false_fn=lambda: (_t, _u, _betas),
            )
            return _t, _u - 1, _T, _U, _betas

        _t, u, _T, _U, _betas = tf.while_loop(_u_cond, _u_body, loop_vars=[_t, u, _T, _U, _betas])
        return _t - 1, _T, _U, _betas

    t, T, U, betas = tf.while_loop(_t_cond, _t_body, loop_vars=[t, T, U, betas])
    ll_backward = tf.tensor_scatter_nd_update(ll_backward, [[batch]], [tf.gather(betas, [beta_offset])])

    return betas, ll_backward


def compute_rnnt_loss_and_grad_helper(
    logits,
    labels,
    label_length,
    logit_length,
):
    return None


def rnnt_loss(
    logits,
    labels,
    label_length,
    logit_length,
    name=None,
):
    name = "rnnt_loss" if name is None else name
    with tf.name_scope(name):
        logits = tf.convert_to_tensor(logits, name="logits")
        labels = tf.convert_to_tensor(labels, name="labels")
        label_length = tf.convert_to_tensor(label_length, name="label_length")
        logit_length = tf.convert_to_tensor(logit_length, name="logit_length")

        args = [logits, labels, label_length, logit_length]

        @tf.custom_gradient
        def compute_rnnt_loss_and_grad(logits_t, labels_t, label_length_t, logit_length_t):
            """Compute RNN-T loss and gradients."""
            logits_t.set_shape(logits.shape)
            labels_t.set_shape(labels.shape)
            label_length_t.set_shape(label_length.shape)
            logit_length_t.set_shape(logit_length.shape)
            kwargs = dict(
                logits=logits_t,
                labels=labels_t,
                label_length=label_length_t,
                logit_length=logit_length_t,
            )
            result = compute_rnnt_loss_and_grad_helper(**kwargs)

            def grad(grad_loss):
                grads = [tf.reshape(grad_loss, [-1, 1, 1, 1]) * result[1]]
                grads += [None] * (len(args) - len(grads))
                return grads

            return result[0], grad

        return compute_rnnt_loss_and_grad(*args)
