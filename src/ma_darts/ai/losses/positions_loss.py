import tensorflow as tf
from ma_darts import img_size


class PositionsLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        square_size: int = 50,
    ):
        super().__init__()
        # self.square_size = tf.constant(square_size, tf.int32)
        self.img_size = tf.constant(img_size, tf.int32)

        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def get_existence(
        self,
        y: tf.Tensor,  # (bs, s, s, 8, 3)
    ) -> tf.Tensor:  # (bs, n, 1), bool

        # Remove positions
        y = y[..., 2:, :]  # (bs, s, s, 6, 3)

        # Flatten tensor
        y = tf.transpose(y, [0, 1, 2, 4, 3])  # (bs, y, x, 3, 6)
        shape = tf.shape(y)
        y = tf.reshape(y, (shape[0], -1, shape[-1]))  # (bs, n, 6)

        # Find existences
        nothing = y[..., 0]  # (bs, n)
        something = tf.reduce_sum(y[..., 1:], axis=-1)  # (bs, n)

        denom = tf.maximum(something + nothing, 1e-6)
        xst_prob = something / denom

        xst = tf.greater(xst_prob, 0.5)  # (bs, n), bool
        xst = tf.expand_dims(xst, -1)  # (bs, n, 1), bool
        return xst

    def call(
        self,
        y_true: tf.Tensor,  # (bs, s, s, 8, 3)
        y_pred: tf.Tensor,
    ):
        # Collect existences
        xst_true = self.get_existence(y_true)  # (bs, n, 1), bool
        xst_pred = self.get_existence(y_pred)

        # Remove classes
        pos_true = y_true[..., :2, :]  # (bs, s, s, 2, 3)
        pos_pred = y_pred[..., :2, :]  # (bs, s, s, 2, 3)

        # Create grid for absolute coordinates
        s = tf.shape(pos_true)[1]
        cell_size = tf.cast(tf.divide(self.img_size, s), tf.float32)
        grid_indices = tf.stack(
            tf.meshgrid(tf.range(s), tf.range(s), indexing="ij"),
            axis=-1,
        )  # (s, s, 2)
        global_grid_pos = tf.cast(grid_indices, tf.float32) * cell_size  # (s, s, 2)

        # Convert to absolute coordinates
        pos_true = (
            global_grid_pos[None, ..., None] + pos_true * cell_size
        )  # (bs, s, s, 2, 3)
        pos_pred = global_grid_pos[None, ..., None] + pos_pred * cell_size

        # Flatten positions
        pos_true = tf.transpose(pos_true, (0, 1, 2, 4, 3))  # (bs, s, s, 3, 2)
        pos_pred = tf.transpose(pos_pred, (0, 1, 2, 4, 3))
        shape = tf.shape(pos_true)
        pos_true = tf.reshape(pos_true, (shape[0], -1, shape[-1]))  # (bs, n, 2)
        pos_pred = tf.reshape(pos_pred, (shape[0], -1, shape[-1]))

        # Mask positions
        pos_true = pos_true * tf.cast(xst_true, tf.float32)  # (bs, n, 2)
        pos_pred = pos_pred * tf.cast(xst_pred, tf.float32)

        err = tf.square(y_true - y_pred)
        loss = tf.reduce_sum(err) * tf.cast(self.img_size / s, tf.float32) / 10
        # loss = self.loss_fn(pos_true, pos_pred) / cell_size**2

        return loss

        # TODO: IoU calculation using points as middle points with squares of size self.square_size as surrounding
        # The IoU of two points with a given square_size can be calculated using:
        # Intersection = max(0, (square_size - dy) * (square_size - dx))
        # Union = 2 * square_size**2 - Intersection
        # IoU = Intersection / Union

        return tf.reduce_mean(tf.square(pos_true - pos_pred))


if __name__ == "__main__":
    import pickle

    X, y_true = pickle.load(open("dump/sample.pkl", "rb"))
    y_pred = [tf.gather(y, [3, 2, 1, 0], axis=0) for y in y_true]
    y_pred = [
        tf.clip_by_value(y + tf.random.normal(y.shape, stddev=0.02), 0, 1)
        for y in y_pred
    ]

    l = PositionsLoss()
    for y_t, y_p in zip(y_true, y_pred):
        print("#" * 120)
        print(str(y_t.shape).center(120))
        print("#" * 120)

        error = 0.05
        y_p = error * y_p + (1 - error) * y_t

        loss = l(y_t, y_p)
        print("loss =", loss.numpy())
