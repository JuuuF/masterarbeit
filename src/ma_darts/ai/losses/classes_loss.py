import tensorflow as tf
from ma_darts import img_size


class ClassesLoss(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        super(ClassesLoss, self).__init__(*args, **kwargs)
        self.kernel = tf.constant(self.get_kernel(3, 0.55), tf.float32)
        self.loss_fn = tf.keras.losses.CategoricalFocalCrossentropy()

    def get_config(self):
        config = super(ClassesLoss, self).get_config()
        return config

    # @tf.function
    def get_kernel(
        self,
        size: tf.Tensor,  # ()
        sigma: int = 2,
    ):
        x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        x = tf.exp(-tf.square(x) / (2 * sigma**2))
        kernel = tf.tensordot(x, x, axes=0)  # (size, size)
        kernel = kernel / tf.reduce_sum(kernel)

        # Reshape for big convolution
        kernel = tf.expand_dims(tf.expand_dims(kernel, -1), -1)  # (size, size, 1, 1)
        kernel = tf.tile(kernel, [1, 1, 6, 1])  # (size, size, 6, 1)
        return kernel

    def accentuate(
        self,
        y: tf.Tensor,  # (bs, s, s, 3)
    ):
        y2 = tf.square(y)
        return y2 / (y2 + tf.square(1 - y))

    # @tf.function
    def apply_filter(
        self,
        y: tf.Tensor,  # (bs, s, s, 18)
    ) -> tf.Tensor:  # (bs, s, s, 18)

        y_conv = tf.nn.depthwise_conv2d(
            y,
            self.kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        return y_conv

    # @tf.function
    def call(
        self,
        y_true,  # (bs, s, s, 8, 3)
        y_pred,
    ):

        # Reshape for convolution
        cls_true = tf.reduce_sum(y_true[..., 2:, :], axis=-1)  # (bs, s, s, 6)
        cls_pred = tf.reduce_sum(y_pred[..., 2:, :], axis=-1)

        # Filter images
        cls_true_f = self.apply_filter(cls_true)  # (bs, s, s, 6)
        cls_pred_f = self.apply_filter(cls_pred)

        loss = self.loss_fn(cls_true_f, cls_pred_f)

        # Calculate weighing factor
        # factor ~1:      good guess -> little penalty
        # 1 < factor < 2: a bit off, -> a bit penalty
        # factor ~2:      far off    -> big penalty
        n_trues = tf.reduce_sum(cls_true, axis=[1, 2])  # (bs, 6)
        n_preds = tf.reduce_sum(self.accentuate(cls_pred), axis=[1, 2])  # (bs, 6)
        n_diff = tf.abs(n_trues - n_preds)
        factor = 2 * tf.math.sigmoid(n_diff / 10)
        factor = tf.reduce_mean(factor)  # ()

        loss_adj = tf.pow(loss, 1 / factor)
        return loss_adj


class ClassesLoss_(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.img_size = img_size

    def extract_classes(
        self,
        y: tf.Tensor,  # (bs, s, s, 8, 3)
    ):
        # Remove positions
        y = y[..., 2:, :]  # (bs, s, s, 6, 3)

        # Flatten
        y = tf.transpose(y, (0, 1, 2, 4, 3))  # (bs, s, s, 3, 6)
        shape = tf.shape(y)
        y = tf.reshape(y, (shape[0], -1, shape[-1]))  # (bs, n, 6)

        return y

    def call(
        self,
        y_true: tf.Tensor,  # (bs, s, s, 8, 3)
        y_pred: tf.Tensor,
    ):
        s = tf.shape(y_true)[1]
        cls_true = self.extract_classes(y_true)  # (bs, n, 6)
        cls_pred = self.extract_classes(y_pred)
        loss = self.loss_fn(cls_true, cls_pred) * 10
        # loss = tf.reduce_mean(tf.square(cls_true - cls_pred))

        # err = tf.square(cls_true - cls_pred)
        # err = self.loss_fn(cls_true, cls_pred)
        # loss = tf.reduce_sum(err) * self.img_size / 10
        return loss


if __name__ == "__main__":
    import pickle

    X, y_true = pickle.load(open("dump/sample.pkl", "rb"))
    y_pred = [tf.gather(y, [3, 2, 1, 0], axis=0) for y in y_true]
    y_pred = [
        tf.clip_by_value(y + tf.random.normal(y.shape, stddev=0.1), 0, 1)
        for y in y_pred
    ]

    l = ClassesLoss()
    for y_t, y_p in zip(y_true, y_pred):
        print("#" * 120)
        print(str(y_t.shape).center(120))
        print("#" * 120)

        error = 0.9
        y_p = error * y_p + (1 - error) * y_t

        loss = l(y_t, y_p)
        print("loss =", loss.numpy())
        exit()
