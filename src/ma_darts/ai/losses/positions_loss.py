import tensorflow as tf
from ma_darts import img_size
from ma_darts.ai.utils import get_grid_existences


class PositionsLoss(tf.keras.losses.Loss):
    def __init__(self, multiplier: float = 1, *args, **kwargs):
        super(PositionsLoss, self).__init__(*args, **kwargs)
        self.multiplier = multiplier

    def get_config(self):
        config = super(PositionsLoss, self).get_config()
        config.update(
            {
                "multiplier": self.multiplier,
            }
        )
        return config

    # @tf.function
    def call(
        self,
        y_true,  # (bs, s, s, 8, 3)
        y_pred,
    ):

        # Get existences
        xst_true = get_grid_existences(y_true)  # (bs, s, s, 1, 3)
        xst_true = tf.transpose(xst_true, (0, 1, 2, 4, 3))  # (bs, s, s, 3, 1)
        xst_true = tf.reshape(
            xst_true, (tf.shape(xst_true)[0], -1, 1)
        )  # (bs, s * s * 3, 1)

        # Get positions
        pos_true = y_true[..., :2, :]  # (bs, s, s, 2, 3)
        pos_pred = y_pred[..., :2, :]
        pos_true = tf.transpose(pos_true, (0, 1, 2, 4, 3))  # (bs, s, s, 3, 2)
        pos_pred = tf.transpose(pos_pred, (0, 1, 2, 4, 3))
        pos_true = tf.reshape(
            pos_true, (tf.shape(pos_true)[0], -1, 2)
        )  # (bs, s * s * 3, 2)
        pos_pred = tf.reshape(pos_pred, (tf.shape(pos_pred)[0], -1, 2))

        # Clear non-existing predictions
        pos_pred = pos_pred * xst_true
        pos_true = pos_true * xst_true

        # Get distances
        diffs = tf.abs(pos_true - pos_pred)  # (bs, s * s * 3, 2)
        total_dists = tf.reduce_sum(diffs, axis=[1, 2])  # (bs,)

        # Compensate with abount of existing points
        n_trues = tf.reduce_sum(xst_true, axis=[1, 2])  # (bs,)
        final_dists = total_dists / tf.maximum(n_trues, 1)

        # Convert to loss
        loss = tf.reduce_mean(final_dists)

        return loss * tf.constant(self.multiplier, tf.float32)


if __name__ == "__main__":
    import pickle

    X, y_true = pickle.load(open("dump/sample.pkl", "rb"))
    y_pred = [tf.gather(y, [3, 2, 1, 0], axis=0) for y in y_true]
    y_pred = [
        tf.clip_by_value(y + tf.random.normal(y.shape, stddev=0.1, seed=0), 0, 1)
        for y in y_pred
    ]

    l = PositionsLoss()
    for y_t, y_p in zip(y_true, y_pred):
        print("#" * 120)
        print(str(y_t.shape).center(120))
        print("#" * 120)

        losses = []
        import numpy as np

        xs = np.arange(101) / 100
        for fac in xs:
            y_p_ = fac * y_p + (1 - fac) * y_t
            loss = l(y_t, y_p_)
            losses.append(loss)
        from matplotlib import pyplot as plt

        plt.plot(xs, losses)
        plt.show()
        import cv2

        cv2.waitKey()

        exit()
