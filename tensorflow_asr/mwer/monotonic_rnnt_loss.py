from functools import cached_property
import tensorflow as tf

logger = tf.get_logger()
LOG_0 = float("-inf")


class MonotonicRnntLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        blank=0,
        global_batch_size=None,
        name=None,
    ):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
        self.blank = blank
        self.global_batch_size = global_batch_size

    def call(self, y_true, y_pred):
        loss = monotonic_rnnt_loss(
            logits=y_pred["logits"],
            logit_length=y_pred["logits_length"],
            labels=y_true["labels"],
            label_length=y_true["labels_length"],
            blank=self.blank,
            name=self.name,
        )
        return tf.nn.compute_average_loss(loss, global_batch_size=self.global_batch_size)


@tf.function
def monotonic_rnnt_loss(
    logits,
    labels,
    label_length,
    logit_length,
    blank=0,
    name=None,
):
    name = "rnnt_loss" if name is None else name
    with tf.name_scope(name):
        logits = tf.convert_to_tensor(logits, name="logits")
        labels = tf.convert_to_tensor(labels, name="labels")
        label_length = tf.convert_to_tensor(label_length, name="label_length")
        logit_length = tf.convert_to_tensor(logit_length, name="logit_length")

        @tf.custom_gradient
        def compute_rnnt_loss_and_grad(logits_t: tf.Tensor):
            """Compute RNN-T loss and gradients."""
            loss_data = MonotonicRnntData(
                logits=logits_t,
                labels=labels,
                label_length=label_length,
                logit_length=logit_length,
            )

            return -loss_data.log_loss, loss_data.backprop

        output = compute_rnnt_loss_and_grad(logits)
        return output


class MonotonicRnntData:
    def __init__(
        self,
        logits: tf.Tensor,
        labels: tf.Tensor,
        logit_length: tf.Tensor,
        label_length: tf.Tensor,
    ):
        super().__init__()
        self._logits = logits
        self._labels = labels
        self._logit_length = logit_length
        self._label_length = label_length

    def backprop(self, loss: tf.Tensor) -> tf.Tensor:
        return tf.reshape(loss, shape=[-1, 1, 1, 1]) * self.grads

    @cached_property
    def grads(self) -> tf.Tensor:
        """Computes gradients w.r.t logits.

        Returns:    tf.Tensor shape = [batch_size, input_max_len, target_max_len + 1, vocab_size + 1], dtype = tf.float32
        """

        left_side = tf.exp(
            tf.expand_dims(self.alpha + self.beta - tf.reshape(self.log_loss, shape=[self.batch_size, 1, 1]), axis=3)
            + self.log_probs
        )

        right_side = tf.concat([self.grads_blank, self.grads_truth], axis=3)

        grads_logits = left_side - right_side

        return grads_logits

    @cached_property
    def grads_truth(self) -> tf.Tensor:
        """Computes part of the RHS corresponding to k = y_u+1

        Returns:    tf.Tensor shape = [batch_size, input_max_len, target_max_len + 1, vocab_size], dtype = tf.float32
        """
        grads_truth = tf.exp(
            (
                self.alpha[:, :-1, :-1]
                + self.beta[:, 1:, 1:]
                - tf.reshape(self.log_loss, shape=[self.batch_size, 1, 1])
                + self.truth_probs[:, :-1, :]
            )
        )

        grads_truth = tf.expand_dims(tf.pad(grads_truth, [[0, 0], [0, 1], [0, 1]], "CONSTANT"), axis=3)

        grads_truth = (
            tf.tile(grads_truth, multiples=[1, 1, 1, self.vocab_size - 1])
            * tf.pad(self.one_hot_labels, [[0, 0], [0, 0], [0, 1], [0, 0]], "CONSTANT")[:, :, :, 1:]
        )

        return grads_truth

    @cached_property
    def grads_blank(self) -> tf.Tensor:
        """Computes part of the RHS corresponding to k = blank

        Returns:    tf.Tensor shape = [batch_size, input_max_len, target_max_len + 1, 1], dtype = tf.float32
        """
        beta_expanded = tf.tensor_scatter_nd_update(
            tf.pad(self.beta, [[0, 0], [0, 1], [0, 0]], "CONSTANT", constant_values=LOG_0)[:, 1:, :],
            indices=tf.concat(
                [
                    tf.reshape(tf.range(self.batch_size, dtype=tf.int32), shape=[self.batch_size, 1]),
                    self.last_elem_indices,
                ],
                axis=1,
            ),
            updates=tf.zeros(shape=[self.batch_size], dtype=tf.float32),
        )

        grads_blank = tf.exp(
            (self.alpha + beta_expanded - tf.reshape(self.log_loss, shape=[self.batch_size, 1, 1]) + self.blank_probs)
        )

        return tf.expand_dims(grads_blank, axis=3)

    @cached_property
    def alpha(self) -> tf.Tensor:
        """Computes the forward alpha variable

        Returns:    tf.Tensor shape = [batch_size, input_max_len, target_max_len + 1], dtype = tf.float32
        """

        def next_state(last_output, trans_probs):
            blank_probs = trans_probs[0]
            truth_probs = trans_probs[1]

            alpha_b = last_output + blank_probs
            alpha_t = tf.concat(
                [LOG_0 * tf.ones(shape=[self.batch_size, 1]), last_output[:, :-1] + truth_probs], axis=1
            )

            alpha_next = tf.math.reduce_logsumexp(tf.stack([alpha_b, alpha_t], axis=0), axis=0)
            return alpha_next

        initial_alpha = tf.concat(
            [
                tf.zeros(shape=[self.batch_size, 1]),
                tf.ones(shape=[self.batch_size, self.target_max_len - 1]) * LOG_0,
            ],
            axis=1,
        )

        blank_probs_t = tf.transpose(self.blank_probs, perm=[1, 0, 2])
        truth_probs_t = tf.transpose(self.truth_probs, perm=[1, 0, 2])

        fwd = tf.scan(next_state, (blank_probs_t[:-1, :, :], truth_probs_t[:-1, :, :]), initializer=initial_alpha)

        alpha = tf.concat([tf.expand_dims(initial_alpha, axis=0), fwd], axis=0)
        alpha = tf.transpose(alpha, perm=[1, 0, 2])

        return alpha

    @cached_property
    def beta(self) -> tf.Tensor:
        """Computes the backward beta variable.

        Returns:    tf.Tensor shape = [batch_size, input_max_len, target_max_len + 1], dtype = tf.float32
        """

        def next_state(last_output, mask_and_trans_probs):

            mask_s, blank_probs, truth_probs = mask_and_trans_probs

            beta_b = last_output + blank_probs
            beta_t = tf.pad(last_output[:, 1:] + truth_probs, [[0, 0], [0, 1]], "CONSTANT", constant_values=LOG_0)

            beta_next = tf.math.reduce_logsumexp(tf.stack([beta_b, beta_t], axis=0), axis=0)

            masked_beta_next = self.nan_to_zero(beta_next * tf.expand_dims(mask_s, axis=1)) + self.nan_to_zero(
                last_output * tf.expand_dims((1.0 - mask_s), axis=1)
            )

            return tf.reshape(masked_beta_next, shape=tf.shape(last_output))

        beta_init_val = tf.gather_nd(self.blank_probs, self.last_elem_indices, batch_dims=1)
        # Initial beta for batches.
        initial_beta_mask = tf.one_hot(self._label_length, depth=self.target_max_len)
        initial_beta = tf.expand_dims(beta_init_val, axis=1) * initial_beta_mask + self.nan_to_zero(
            LOG_0 * (1.0 - initial_beta_mask)
        )

        beta_mask = tf.transpose(
            tf.sequence_mask(self._logit_length, maxlen=self.input_max_len, dtype=tf.float32), perm=[1, 0]
        )
        blank_probs_t = tf.transpose(self.blank_probs, perm=[1, 0, 2])
        truth_probs_t = tf.transpose(self.truth_probs, perm=[1, 0, 2])

        bwd = tf.scan(
            next_state,
            (beta_mask[1:, :], blank_probs_t[:-1, :, :], truth_probs_t[:-1, :, :]),
            initializer=initial_beta,
            reverse=True,
        )

        beta = tf.concat([bwd, tf.expand_dims(initial_beta, axis=0)], axis=0)
        beta = tf.transpose(beta, perm=[1, 0, 2])

        # remove beta entries that are beyond T and U of a given batch element
        beta = beta + tf.math.log(tf.cast(self.dp_mask, dtype=tf.float32))

        return beta

    @cached_property
    def log_loss(self) -> tf.Tensor:
        """Log loss defined by ln P(y*|x)."""
        return self.beta[:, 0, 0]

    @property
    def dp_mask(self) -> tf.Tensor:
        """Computes mask for each elem of the batch

        The mask indicates the region of interest for each batch element,
        that is the area bounded by label_length[i] and logit_length[i] where i
        is an index over the batch dimension.

        Returns:    tf.Tensor shape = [batch_size, input_max_len, target_max_len + 1, vocab_size + 1], dtype = tf.float32
        """
        label_mask = tf.expand_dims(
            tf.sequence_mask(self._label_length + 1, maxlen=self.target_max_len, dtype=tf.float32),
            axis=1,
        )

        input_mask = tf.expand_dims(
            tf.sequence_mask(self._logit_length, maxlen=self.input_max_len, dtype=tf.float32), axis=2
        )

        return label_mask * input_mask

    @cached_property
    def last_elem_indices(self) -> tf.Tensor:
        return tf.stack([self._logit_length - 1, self._label_length], axis=1)

    @cached_property
    def truth_probs(self) -> tf.Tensor:
        """Log probabilites of obtaining symbol y_u+1 at each encoder step t and decoder step u.

        Returns:    tf.Tensor shape = [batch_size, input_max_len, target_max_len], dtype = tf.float32
        """
        return tf.reduce_sum(tf.multiply(self.log_probs[:, :, :-1, :], self.one_hot_labels), axis=-1)

    @cached_property
    def blank_probs(self) -> tf.Tensor:
        """Log probabilites of obtaining a blank symbol at each encoder and decoder step.

        Returns:    tf.Tensor shape = [batch_size, input_max_len, target_max_len + 1], dtype = tf.float32
        """
        return self.log_probs[:, :, :, 0]

    @cached_property
    def log_probs(self) -> tf.Tensor:
        return tf.nn.log_softmax(self._logits)

    @cached_property
    def one_hot_labels(self) -> tf.Tensor:
        return tf.one_hot(
            tf.tile(tf.expand_dims(self._labels, axis=1), multiples=[1, self.input_max_len, 1]),
            depth=self.vocab_size,
        )

    @cached_property
    def batch_size(self) -> tf.Tensor:
        return tf.shape(self._logits)[0]

    @cached_property
    def input_max_len(self) -> tf.Tensor:
        return tf.shape(self._logits)[1]

    @cached_property
    def target_max_len(self) -> tf.Tensor:
        return tf.shape(self._logits)[2]

    @cached_property
    def vocab_size(self) -> tf.Tensor:
        return tf.shape(self._logits)[3]

    # TO-DO: remove need for this function
    def nan_to_zero(self, input_tensor: tf.Tensor) -> tf.Tensor:
        return tf.where(tf.math.is_nan(input_tensor), tf.zeros_like(input_tensor), input_tensor)
