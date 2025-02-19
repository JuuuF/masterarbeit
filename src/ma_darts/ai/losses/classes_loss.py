import tensorflow as tf
from ma_darts import img_size


class ClassesLoss(tf.keras.losses.Loss):
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
        loss = self.loss_fn(cls_true, cls_pred)

        err = tf.square(cls_true - cls_pred)
        loss = tf.reduce_sum(err) / tf.cast(s, tf.float32) * 10
        return loss


if __name__ == "__main__":
    import pickle

    X, y_true = pickle.load(open("dump/sample.pkl", "rb"))
    y_pred = [tf.gather(y, [3, 2, 1, 0], axis=0) for y in y_true]
    y_pred = [
        tf.clip_by_value(y + tf.random.normal(y.shape, stddev=0.02), 0, 1)
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
