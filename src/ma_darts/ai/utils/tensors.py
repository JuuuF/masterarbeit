import tensorflow as tf


def get_grid_existence_per_cell(
    t: tf.Tensor,  # (bs, s, s, 8, 3)
) -> tf.Tensor:  # (bs, s, s)
    shape = tf.shape(t)  # bs, s, s, 8, 3

    # Remove positions
    t = t[..., 2:, :]  # (bs, s, s, 6, 3)

    # Concat entries per cell
    t = tf.reduce_sum(t, axis=-1)  # (bs, s, s, 6)

    # Find existences
    nothing = t[..., 0]  # (bs, s, s)
    something = tf.reduce_sum(t[..., 1:], axis=-1)  # (bs, s, s)

    xst_prob = tf.math.divide_no_nan(something, something + nothing)

    return xst_prob  # (bs, s, s)


def get_grid_existences(
    t: tf.Tensor,  # (bs, s, s, 8, 3)
) -> tf.Tensor:  # (bs, s, s, 1, 3)
    shape = tf.shape(t)  # bs, s, s, 8, 3

    # Remove positions
    t = t[..., 2:, :]  # (bs, s, s, 6, 3)

    # Find existences
    nothing = t[..., 0, :]  # (bs, s, s, 3)
    something = tf.reduce_sum(t[..., 1:, :], axis=-2)  # (bs, s, s, 3)

    denom = tf.maximum(something + nothing, 1e-6)
    xst_prob = something / denom

    xst_prob = tf.expand_dims(xst_prob, axis=-1)  # (bs, s, s, 1, 3)

    return xst_prob  # (bs, s, s, 1, 3)
