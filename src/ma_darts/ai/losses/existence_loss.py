import tensorflow as tf
from ma_darts import img_size
from ma_darts.ai.utils import split_outputs_to_xst_pos_cls


class ExistenceLoss(tf.keras.losses.Loss):
    def __init__(self, multiplier: float = 1, *args, **kwargs):
        super(ExistenceLoss, self).__init__(*args, **kwargs)

        self.loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=0.5,
            gamma=2.0,
            reduction=None,
        )
        self.multiplier = multiplier

    def get_config(self):
        config = super(ExistenceLoss, self).get_config()
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
        xst_true, _, _ = split_outputs_to_xst_pos_cls(y_true)  # (bs, s, s, 1, 3)
        xst_pred, _, _ = split_outputs_to_xst_pos_cls(y_pred)

        # Label smoothing
        xst_true = 0.998 * xst_true + 0.001

        # Flatten grid
        batch_size = tf.shape(y_true)[0]
        xst_true = tf.reshape(xst_true, (batch_size, -1))  # (bs, s * s * 3)
        xst_pred = tf.reshape(xst_pred, (batch_size, -1))

        loss = self.loss_fn(xst_true, xst_pred)  # (bs,)
        return loss * tf.constant(self.multiplier, tf.float32)  # (bs,)


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
