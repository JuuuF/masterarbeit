import tensorflow as tf
from ma_darts import img_size
from ma_darts.ai.utils import get_grid_existences


class PositionsLoss(tf.keras.losses.Loss):
    def __init__(self, square_size: float = 0.1, *args, **kwargs):
        super(PositionsLoss, self).__init__(*args, **kwargs)
        self._square_size = square_size
        self.square_size = tf.constant(square_size, tf.float32)
        self.base_union = tf.constant(square_size * square_size * 2, tf.float32)
        self.epsilon = tf.constant(1e-6, tf.float32)

    def get_config(self):
        config = super(PositionsLoss, self).get_config()
        config.update({"square_size": self._square_size})
        return config

    def diou_per_position(
        self,
        p: tf.Tensor,  # (2,)
        ps: tf.Tensor,  # (n, 2)
        xst_p: tf.Tensor,  # ()
        xst_ps: tf.Tensor,  # (n,)
    ) -> tf.Tensor:  # ()
        bother_computing = tf.cast(tf.greater(xst_p, 1e-6), tf.float32)

        # Get differences
        diffs = tf.abs(p[None, :] - ps)  # (n, 2)
        side_lengths = tf.maximum(
            tf.constant(0, tf.float32), self.square_size - diffs
        )  # (n, 2)
        intersections = side_lengths[:, 0] * side_lengths[:, 1]  # (n,)
        unions = self.base_union - intersections
        ious = intersections / (unions + self.epsilon)  # (n,)

        # Get distances for
        dists_squared = tf.square(diffs[:, 0]) + tf.square(diffs[:, 1])  # (n,)
        diags_squared = tf.square(diffs[:, 0] + self.square_size) + tf.square(
            diffs[:, 1] + self.square_size
        )  # (n,)
        dist_addition = dists_squared / (diags_squared + self.epsilon)

        # IoU -> DIoU
        dious = ious + dist_addition

        # Scale DIoUs by existences
        dious = dious * xst_ps * xst_p  # (n,)

        best_diou = tf.reduce_max(dious)  # ()
        return bother_computing * best_diou

    def batch_diou_(
        self,
        pos_true: tf.Tensor,  # (n, 2)
        pos_pred: tf.Tensor,
        xst_true: tf.Tensor,  # (n,)
        xst_pred: tf.Tensor,
    ) -> tf.Tensor:  # ()
        n_trues = tf.shape(pos_true)[0]
        dious_per_true_sample = tf.map_fn(
            lambda t: self.diou_per_position(
                pos_true[t], pos_pred, xst_true[t], xst_pred
            ),
            elems=tf.range(n_trues),
            fn_output_signature=tf.float32,
        )  # (n,)

        diou = tf.reduce_max(dious_per_true_sample)
        return diou

    def batch_diou(
        self,
        pos_true: tf.Tensor,  # (n, 2)
        pos_pred: tf.Tensor,
        xst_true: tf.Tensor,  # (n,)
        xst_pred: tf.Tensor,
    ) -> tf.Tensor:  # ()
        pos_true_expanded = tf.expand_dims(pos_true, 1)  # (n, 1, 2)
        pos_pred_expanded = tf.expand_dims(pos_pred, 0)  # (1, n, 2)
        xst_true_expanded = tf.expand_dims(xst_true, 1)  # (n, 1)
        xst_pred_expanded = tf.expand_dims(xst_pred, 0)  # (1, n)

        # Get diffs
        diffs = tf.abs(pos_true_expanded - pos_pred_expanded)  # (n, n, 2)

        side_lengths = tf.maximum(
            tf.constant(0, tf.float32), self.square_size - diffs
        )  # (n, n, 2)

        intersections = side_lengths[..., 0] * side_lengths[..., 1]  # (n, n)
        unions = self.base_union - intersections  # (n, n)
        ious = intersections / (unions + self.epsilon)  # (n, n)

        # Get distances for DIoU
        dists_squared = tf.square(diffs[..., 0]) + tf.square(diffs[..., 1])
        diags_squared = tf.square(diffs[..., 0] + self.square_size) + tf.square(
            diffs[..., 1] + self.square_size
        )  # (n, n)
        dists_addition = dists_squared / (diags_squared + self.epsilon)

        # IoU -> DIoU
        dious = ious + dists_addition  # (n, n)

        # Scale by existences
        dious = dious * xst_true * xst_pred  # (n, n)

        # Get best DIoU per true sample
        best_dious = tf.reduce_max(dious, axis=0)  # (n,)

        # Average over existing true samples
        n_trues = tf.maximum(
            tf.cast(tf.math.count_nonzero(xst_true), tf.float32),
            tf.constant(1, tf.float32),
        )
        mean_diou = tf.reduce_sum(best_dious) / n_trues  # ()
        return mean_diou

    def call(
        self,
        y_true: tf.Tensor,  # (bs, s, s, 8, 3)
        y_pred: tf.Tensor,
    ):
        # Remove classes
        pos_true = y_true[..., :2, :]  # (bs, s, s, 2, 3)
        pos_pred = y_pred[..., :2, :]

        # Create grid for absolute coordinates
        s = tf.cast(tf.shape(pos_true)[1], tf.float32)
        pos_range = tf.range(s, dtype=tf.float32) / s
        grid_pos = tf.stack(
            tf.meshgrid(pos_range, pos_range, indexing="ij"),
            axis=-1,
        )  # (s, s, 2)

        # Convert to absolute coordinates
        pos_true = grid_pos[None, ..., None] + pos_true / s  # (bs, s, s, 2, 3)
        pos_pred = grid_pos[None, ..., None] + pos_pred / s

        # Get existences
        xst_true = get_grid_existences(y_true)  # (bs, s, s, 1, 3)
        xst_pred = get_grid_existences(y_pred)

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

        dious_per_batch = tf.map_fn(
            lambda b: self.batch_diou(
                pos_true[b], pos_pred[b], xst_true[b], xst_pred[b]
            ),
            elems=tf.range(batch_size),
            fn_output_signature=tf.float32,
        )  # (bs,)
        diou = tf.reduce_mean(dious_per_batch)
        loss = 1 - diou
        return loss


if __name__ == "__main__":
    import pickle

    X, y_true = pickle.load(open("dump/sample.pkl", "rb"))
    y_pred = [tf.gather(y, [3, 2, 1, 0], axis=0) for y in y_true]
    y_pred = [
        tf.clip_by_value(y + tf.random.normal(y.shape, stddev=0.1, seed=0), 0, 1)
        for y in y_pred
    ]

    l = PositionsLoss(square_size=50)
    for y_t, y_p in zip(y_true, y_pred):
        print("#" * 120)
        print(str(y_t.shape).center(120))
        print("#" * 120)

        error = 0.1
        y_p = error * y_p + (1 - error) * y_t

        loss = l(y_t, y_p)
        print("loss =", loss.numpy())
        exit()
