import tensorflow as tf
from ma_darts import img_size
from ma_darts.ai.utils import get_grid_existences


class ExistenceLoss(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        super(ExistenceLoss, self).__init__(*args, **kwargs)

        self.kernel = tf.constant(self.get_kernel(3, 0.55), tf.float32)
        self.loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
            # apply_class_balancing=True,
            # alpha=0.25,
            # gamma=2.0,
        )

    def get_config(self):
        config = super(ExistenceLoss, self).get_config()
        return config

    # @tf.function
    def get_kernel(self, size: tf.Tensor, sigma: int = 2):  # ()
        x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        x = tf.exp(-tf.square(x) / (2 * sigma**2))
        kernel = tf.tensordot(x, x, axes=0)  # (size, size)
        kernel = kernel / tf.reduce_sum(kernel)

        # Reshape for big convolution
        kernel = tf.expand_dims(tf.expand_dims(kernel, -1), -1)  # (size, size, 1, 1)
        kernel = tf.tile(kernel, [1, 1, 3, 1])  # (size, size, 3, 1)
        return kernel

    # @tf.function
    def apply_filter(
        self,
        y: tf.Tensor,  # (bs, s, s, 3)
    ) -> tf.Tensor:  # (bs, s, s, 1)

        y_conv = tf.nn.depthwise_conv2d(
            y,
            self.kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        return y_conv

    def accentuate(
        self,
        y: tf.Tensor,  # (bs, s, s, 3)
    ):
        y2 = tf.square(y)
        return y2 / (y2 + tf.square(1 - y))

    # @tf.function
    def call(
        self,
        y_true,  # (bs, s, s, 8, 3)
        y_pred,
    ):

        # Get existences
        xst_true = get_grid_existences(y_true)[..., 0, :]  # (bs, s, s, 3)
        xst_pred = get_grid_existences(y_pred)[..., 0, :]

        # Filter images
        xst_true_f = self.apply_filter(xst_true)[..., 0]  # (bs, s, s)
        xst_pred_f = self.apply_filter(xst_pred)[..., 0]

        loss = self.loss_fn(xst_true_f, xst_pred_f)

        # Calculate weighing factor
        # factor ~1:      good guess -> little penalty
        # 1 < factor < 2: a bit off, -> a bit penalty
        # factor ~2:      far off    -> big penalty

        n_trues = tf.reduce_sum(xst_true)
        n_preds = tf.reduce_sum(self.accentuate(xst_pred))
        n_diff = tf.abs(n_preds - n_trues)
        factor = 2 * tf.math.sigmoid(n_diff / 10)  # scaled 1..2

        loss_adj = tf.pow(loss, 1 / factor)  # scale from x^0.5 to x^1
        return loss_adj

        # import numpy as np
        # import cv2

        # batch = 0
        # img = np.zeros((25, 25, 3), np.uint8)
        # img[..., 0] = np.uint8(xst_pred[batch, ..., 0] * 255)
        # img[..., 2] = np.uint8(self.accentuate(xst_pred)[batch, ..., 0] * 255)
        # # img[..., 0] = np.uint8(xst_pred_f[batch] * 255)
        # img = np.kron(img, np.ones((16, 16, 1), np.uint8))
        # cv2.imshow("", img)
        return loss_adj


class ExistenceLoss_(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.img_size = tf.cast(img_size, tf.float32)
        self.k = tf.constant(10, tf.float32)

    def get_existence(
        self,
        y: tf.Tensor,  # (bs, s, s, 8, 3)
    ) -> tf.Tensor:

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
        return xst_prob

    def call(
        self,
        y_true,  # (bs, s, s, 8, 3)
        y_pred,  # (bs, s, s, 8, 3)
    ):
        s = tf.shape(y_true)[1]
        xst_true = self.get_existence(y_true)  # (bs, n)
        xst_pred = self.get_existence(y_pred)  # (bs, n)
        loss = self.loss_fn(xst_true, xst_pred)
        # loss = tf.pow(loss, 0.8)
        # loss *= 10

        # err = self.loss_fn(xst_true, xst_pred)  # (bs,)

        # n_cells = tf.cast(s * s, tf.float32)
        # err_per_cell = tf.abs(xst_true - xst_pred)  # (bs, n) 0..1
        # squared_err_per_cell = tf.square(err_per_cell)  # (bs, n)
        # loss = tf.reduce_sum(err) * self.img_size / 10
        # loss = tf.reduce_mean(tf.square(xst_true - xst_pred))

        # Get error per cell
        # err = tf.abs(xst_true - xst_pred)  # (bs, n)
        # Apply a log scaling to error
        # log_err = tf.math.log(1 + self.k * err) / tf.math.log(1 + self.k)  # (bs, n)
        # loss = tf.reduce_mean(log_err)
        return loss


if __name__ == "__main__":
    import pickle

    X, y_true = pickle.load(open("dump/sample.pkl", "rb"))
    y_pred = [tf.gather(y, [3, 2, 1, 0], axis=0) for y in y_true]
    y_pred = [
        tf.clip_by_value(y + tf.random.normal(y.shape, stddev=0.1, seed=0), 0, 1)
        for y in y_pred
    ]

    l = ExistenceLoss()
    for y_t, y_p in zip(y_true, y_pred):
        print("#" * 120)
        print(str(y_t.shape).center(120))
        print("#" * 120)

        fac = 0.15
        y_p = fac * y_p + (1 - fac) * y_t

        loss = l(y_t, y_p)
        print("loss =", loss.numpy())
        import cv2

        cv2.waitKey()

        exit()
