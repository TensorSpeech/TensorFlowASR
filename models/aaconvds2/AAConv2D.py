from __future__ import absolute_import

import tensorflow as tf
from models.aaconvds2.Ops import relative_logits, shape_list, split_heads_2d, combine_heads_2d


@tf.function
def self_attention_2d(inputs, dk, dv, Nh, relative=True):
    """2d relative self−attention."""
    _, H, W, _ = shape_list(inputs)
    dkh = dk // Nh
    dvh = dv // Nh
    def flatten_hw(x, d): return tf.reshape(x, [-1, Nh, H * W, d])
    # Compute q, k, v
    # kqv = tf.nn.conv2d(inputs, 2 * dk + dv, 1, "valid")
    kqv = tf.keras.layers.Conv2D(filters=2 * dk + dv, kernel_size=1)(inputs)
    k, q, v = tf.split(kqv, [dk, dk, dv], axis=3)
    q *= dkh ** -0.5  # scaled dot−product
    # After splitting, shape is [B, Nh, H, W, dkh or dvh] q
    q = split_heads_2d(q, Nh)
    k = split_heads_2d(k, Nh)
    v = split_heads_2d(v, Nh)
    # [B, Nh, HW, HW]
    logits = tf.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh), transpose_b=True)
    if relative:
        rel_logits_h, rel_logits_w = relative_logits(H, W, Nh, dkh)(q)
        logits += rel_logits_h
        logits += rel_logits_w
    weights = tf.nn.softmax(logits)
    attn_out = tf.matmul(weights, flatten_hw(v, dvh))
    attn_out = tf.reshape(attn_out, [-1, Nh, H, W, dvh])
    attn_out = combine_heads_2d(attn_out)
    # Project heads
    # attn_out = tf.layers.conv2d(attn_out, dv, 1)
    attn_out = tf.keras.layers.Conv2D(filters=dv, kernel_size=1)(attn_out)
    return attn_out


class AAConv2D(tf.keras.layers.Layer):
    def __init__(self, fout, k, dk, dv, Nh, trainable=True, name="aaconv2d", relative=True, **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.fout = fout; self.k = k
        self.dk = int(dk); self.dv = int(dv); self.Nh = int(Nh)
        self.relative = relative

    def call(self, inputs, **kwargs):
        conv_out = tf.keras.layers.Conv2D(filters=self.fout - self.dv, kernel_size=self.k, padding="same")(inputs)
        attn_out = self_attention_2d(inputs, self.dk, self.dv, self.Nh, relative=self.relative)
        return tf.concat([conv_out, attn_out], axis=3)

    def get_config(self):
        config = super(AAConv2D, self).get_config()
        config.update(
            {
                "fout": self.fout,
                "k": self.k,
                "dk": self.dk,
                "dv": self.dv,
                "Nh": self.Nh,
                "relative": self.relative
            }
        )
        return config

    def from_config(self, config):
        return self(**config)
