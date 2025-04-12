import numpy as np
import tensorflow as tf

from ma_darts import img_size
from ma_darts.ai.losses import (
    ExistenceLoss,
    ClassesLoss,
    PositionsLoss,
    xst_weight,
    cls_weight,
    pos_weight,
)


class YOLOv8Loss(tf.keras.losses.Loss):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(YOLOv8Loss, self).__init__(*args, **kwargs)

        self.xst_loss = ExistenceLoss(multiplier=xst_weight)

        # Classes Loss
        self.cls_loss = ClassesLoss(multiplier=cls_weight)

        # Positions Loss
        self.pos_loss = PositionsLoss(multiplier=pos_weight)

    # @tf.function
    def call(self, y_true, y_pred):

        # Compute XST loss
        xst_loss = self.xst_loss(y_true, y_pred)

        # Compute CLS loss
        cls_loss = self.cls_loss(y_true, y_pred)

        # Compute POS loss
        pos_loss = self.pos_loss(y_true, y_pred)

        # Combine all losses
        total_loss = xst_loss + cls_loss + pos_loss

        return total_loss

    def get_config(self):
        config = super(YOLOv8Loss, self).get_config()
        return config


if __name__ == "__main__":
    import pickle

    X, y_true = pickle.load(open("dump/sample.pkl", "rb"))
    y_pred = [tf.gather(y, [3, 2, 1, 0], axis=0) for y in y_true]
    y_pred = [
        tf.clip_by_value(y + tf.random.normal(y.shape, stddev=0.1, seed=0), 0, 1)
        for y in y_pred
    ]

    l = YOLOv8Loss()
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
