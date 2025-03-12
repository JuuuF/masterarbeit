import tensorflow as tf
from ma_darts import img_size
from ma_darts.ai.utils import split_outputs_to_xst_pos_cls


class DIoULoss(tf.keras.losses.Loss):
    def __init__(
        self, square_size: float = 0.05, multiplier: float = 1, *args, **kwargs
    ):
        super(DIoULoss, self).__init__(*args, **kwargs)
        if square_size > 1:
            raise ValueError(
                "A square size greater than 1 is not supported. Keep in mind that values are relative!"
            )
        self._square_size = square_size
        self.square_size = tf.constant(square_size, tf.float32)
        self.base_union = tf.constant(square_size * square_size * 2, tf.float32)
        self.epsilon = tf.constant(1e-8, tf.float32)
        self.v_4_pi_squared = tf.constant(4 / (3.14159**2), tf.float32)
        self.multiplier = multiplier

    def get_config(self):
        config = super(DIoULoss, self).get_config()
        config.update(
            {
                "square_size": self._square_size,
                "multiplier": self.multiplier,
            }
        )
        return config

    # @tf.function
    def get_diou_losses_per_cell(
        self,
        pos_true,  # (bs, n, 2)
        pos_pred,
    ):
        diffs = tf.abs(pos_true - pos_pred)  # (bs, n, 2)
        side_lengths = tf.maximum(
            tf.constant(0, tf.float32), self.square_size - diffs
        )  # (bs, n, 2)

        # Compute IoU
        intersection_areas = side_lengths[..., 0] * side_lengths[..., 1]
        union_areas = self.base_union - intersection_areas
        ious = intersection_areas / (union_areas + 1e-8)

        # Get distances
        # 1. Squared middle point dists
        d2 = tf.square(diffs[..., 0]) + tf.square(diffs[..., 1])

        # 2. enclosing bounding box diagonal
        c2 = tf.square(diffs[..., 0] + self.square_size) + tf.square(
            diffs[..., 1] + self.square_size
        )
        distance_factors = d2 / (c2 + 1e-8)

        # IoU -> DIoU
        diou_losses = 1 - ious + distance_factors  # (bs, n)

        # NOTE: Conversion to CIoU is not necessary since the bboxes are
        # equal and that would just yield the same as DIoU does already.

        return diou_losses

    # @tf.function
    def call(
        self,
        y_true: tf.Tensor,  # (bs, s, s, 8, 3)
        y_pred: tf.Tensor,
    ):
        # Split into components
        # (bs, s, s, 1, 3), (bs, s, s, 2, 3)
        xst_true, pos_true, _ = split_outputs_to_xst_pos_cls(y_true)
        xst_pred, pos_pred, _ = split_outputs_to_xst_pos_cls(y_pred)

        # Create grid for absolute coordinates
        s = tf.cast(tf.shape(pos_true)[1], tf.float32)
        pos_range = tf.range(s, dtype=tf.float32) / s
        grid_pos = tf.stack(
            tf.meshgrid(pos_range, pos_range, indexing="ij"),
            axis=-1,
        )  # (s, s, 2)

        # Convert to absolute normalized coordinates
        pos_true = grid_pos[None, ..., None] + pos_true / s  # (bs, s, s, 2, 3) 0..1
        pos_pred = grid_pos[None, ..., None] + pos_pred / s

        # Flatten tensors
        batch_size = tf.shape(pos_true)[0]
        pos_true = tf.transpose(pos_true, (0, 1, 2, 4, 3))  # (bs, s, s, 3, 2)
        pos_pred = tf.transpose(pos_pred, (0, 1, 2, 4, 3))
        xst_true = tf.transpose(xst_true, (0, 1, 2, 4, 3))  # (bs, s, s, 3, 1)
        xst_pred = tf.transpose(xst_pred, (0, 1, 2, 4, 3))
        pos_true = tf.reshape(pos_true, (batch_size, -1, 2))  # (bs, n, 2)
        pos_pred = tf.reshape(pos_pred, (batch_size, -1, 2))
        xst_true = tf.reshape(xst_true, (batch_size, -1))  # (bs, n)
        xst_pred = tf.reshape(xst_pred, (batch_size, -1))

        # Get DIoU loss per cell
        diou_losses = self.get_diou_losses_per_cell(pos_true, pos_pred)  # (bs, n)

        # Apply true existence mask
        diou_losses = xst_true * diou_losses
        diou_loss = tf.reduce_sum(diou_losses, axis=1)  # (bs,)
        diou_loss = tf.reduce_mean(diou_loss)  # ()
        return diou_loss * tf.constant(self.multiplier, tf.float32)


if __name__ == "__main__":
    import pickle

    X, y_true = pickle.load(open("dump/sample.pkl", "rb"))
    y_pred = [tf.gather(y, [3, 2, 1, 0], axis=0) for y in y_true]
    y_pred = [
        tf.clip_by_value(y + tf.random.normal(y.shape, stddev=0.1, seed=0), 0, 1)
        for y in y_pred
    ]

    l = DIoULoss(square_size=0.05)
    for y_t, y_p in zip(y_true, y_pred):
        print("#" * 120)
        print(str(y_t.shape).center(120))
        print("#" * 120)

        losses = []
        import numpy as np

        n_steps = 25
        xs = np.arange(n_steps + 1) / n_steps
        for fac in xs:
            y_p_ = fac * y_p + (1 - fac) * y_t
            loss = l(y_t, y_p_).numpy()
            losses.append(loss)
            print(fac, loss, sep="\t")
        from matplotlib import pyplot as plt

        plt.plot(xs, losses)
        plt.show()
        exit()
