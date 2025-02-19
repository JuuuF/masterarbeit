import tensorflow as tf
from ma_darts import img_size


class ExistenceLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.img_size = tf.cast(img_size, tf.float32)

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
        # loss = self.loss_fn(xst_true, xst_pred)
        # loss = tf.pow(loss, 0.8)
        # loss *= 10
        err = tf.square(xst_true - xst_pred)
        loss = tf.reduce_sum(err) / tf.cast(s*s, tf.float32) * self.img_size / 2
        return loss


if __name__ == "__main__":
    import pickle

    X, y_true = pickle.load(open("dump/sample.pkl", "rb"))
    y_pred = [tf.gather(y, [3, 2, 1, 0], axis=0) for y in y_true]
    y_pred = [
        tf.clip_by_value(y + tf.random.normal(y.shape, stddev=0.02), 0, 1)
        for y in y_pred
    ]

    l = ExistenceLoss()
    for y_t, y_p in zip(y_true, y_pred):
        print("#" * 120)
        print(str(y_t.shape).center(120))
        print("#" * 120)

        fac = 0.1
        y_p = fac * y_p + (1 - fac) * y_t

        loss = l(y_t, y_p)
        print("loss =", loss.numpy())
