import numpy as np
import tensorflow as tf

from ma_darts import img_size
from ma_darts.ai.losses import ExistenceLoss, ClassesLoss, PositionsLoss, DIoULoss


class YOLOv8Loss(tf.keras.losses.Loss):
    def __init__(
        self,
        square_size: float = 0.05,
        *args,
        **kwargs,
    ):
        super(YOLOv8Loss, self).__init__(*args, **kwargs)
        self.square_size = square_size

        self.xst_loss = ExistenceLoss()
        self.xst_mult = tf.constant(400, tf.float32)  # 100

        # Classes Loss
        self.cls_loss = ClassesLoss()
        self.cls_mult = tf.constant(4000, tf.float32)  # 100

        # Positions Loss
        self.pos_loss = PositionsLoss()
        self.pos_mult = tf.constant(0.5, tf.float32)  # 1.5

        # DIoU Loss
        self.diou_loss = DIoULoss(self.square_size)
        self.diou_mult = tf.constant(0.02, tf.float32)

    # @tf.function
    def call(self, y_true, y_pred):

        # Compute XST loss
        raw_xst_loss = self.xst_loss(y_true, y_pred) * self.xst_mult

        # Compute CLS loss
        raw_cls_loss = self.cls_loss(y_true, y_pred) * self.cls_mult

        # Compute POS loss
        raw_pos_loss = self.pos_loss(y_true, y_pred) * self.pos_mult

        # Compute DIoU loss
        # raw_diou_loss = self.diou_loss(y_true, y_pred) * self.diou_mult

        # Combine all losses
        total_loss = raw_xst_loss + raw_cls_loss + raw_pos_loss  # + raw_diou_loss

        return total_loss

    def get_config(self):
        config = super(YOLOv8Loss, self).get_config()
        config.update(
            {
                "square_size": self.square_size,
            }
        )
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
