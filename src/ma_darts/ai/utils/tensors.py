import tensorflow as tf


def get_grid_existence(
    t: tf.Tensor,  # (bs, s, s, 8, 3)
) -> tf.Tensor:  # (bs, s, s)
    shape = tf.shape(t)  # bs, s, s, 8, 3

    # Remove positions
    t = t[..., 2:, :]  # (bs, s, s, 6, 3)

    # Concat entries per cell
    t = tf.reduce_sum(t, axis=-1) / 3  # (bs, s, s, 6)

    # Find existences
    nothing = t[..., 0]  # (bs, s, s)
    something = tf.reduce_sum(t[..., 1:], axis=-1)  # (bs, s, s)

    denom = tf.maximum(something + nothing, 1e-6)
    xst_prob = something / denom

    return xst_prob  # (bs, s, s)
