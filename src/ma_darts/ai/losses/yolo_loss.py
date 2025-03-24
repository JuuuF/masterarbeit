import numpy as np
import tensorflow as tf

from ma_darts import img_size
from ma_darts.ai.losses import ExistenceLoss, ClassesLoss, PositionsLoss, DIoULoss


class YOLOv8Loss(tf.keras.losses.Loss):
    def __init__(
        self,
        square_size: float = 0.05,
        cls_threshold: float = 0.003,
        cls_width: float = 0.002,
        pos_threshold: float = 0.004,
        pos_width: float = 0.002,
        diou_threshold: float = 0.003,
        diou_width: float = 0.001,
        *args,
        **kwargs,
    ):
        super(YOLOv8Loss, self).__init__(*args, **kwargs)
        self.square_size = square_size

        self.xst_loss = ExistenceLoss()
        self.xst_mult = tf.constant(400, tf.float32)  # 100

        # Classes Loss
        self.cls_loss = ClassesLoss()
        self.base_cls_loss = tf.constant(1, tf.float32)
        self.cls_mult = tf.constant(850, tf.float32)  # 100

        self._cls_width = cls_width
        self.cls_width = cls_width if cls_width is not None else cls_threshold
        self.cls_threshold = cls_threshold

        # Positions Loss
        self.pos_loss = PositionsLoss()
        self.base_pos_loss = tf.constant(1, tf.float32)
        self.pos_mult = tf.constant(0.5, tf.float32)  # 1.5

        self._pos_width = pos_width
        self.pos_width = pos_width if pos_width is not None else pos_threshold
        self.pos_threshold = pos_threshold

        # DIoU Loss
        self.diou_loss = DIoULoss(self.square_size)
        self.base_diou_loss = tf.constant(2, tf.float32)
        self.diou_mult = tf.constant(0.02, tf.float32)

        self._diou_width = diou_width
        self.diou_width = diou_width if diou_width is not None else diou_threshold
        self.diou_threshold = diou_threshold

    def get_activation(self, loss, threshold, width):
        full_sigmoid = tf.constant(3, tf.float32)
        return 1 - tf.sigmoid(full_sigmoid * (loss - threshold) / (width * 0.5))

    # @tf.function
    def call(self, y_true, y_pred):

        # Compute XST loss
        raw_xst_loss = self.xst_loss(y_true, y_pred) * self.xst_mult
        # xst_loss = raw_xst_loss * self.xst_mult

        # Compute CLS loss
        raw_cls_loss = self.cls_loss(y_true, y_pred) * self.cls_mult
        # cls_activation = self.get_activation(
        #     raw_xst_loss, self.cls_threshold, self.cls_width
        # )
        # # cls_activation = tf.constant(1, tf.float32)
        # cls_loss = (
        #     cls_activation * raw_cls_loss * self.cls_mult
        #     + (1 - cls_activation) * self.base_cls_loss
        # )

        # Compute POS loss
        raw_pos_loss = self.pos_loss(y_true, y_pred) * self.pos_mult
        # pos_activation = self.get_activation(
        #     raw_cls_loss, self.pos_threshold, self.pos_width
        # )
        # # pos_activation = tf.constant(1, tf.float32)
        # pos_loss = (
        #     pos_activation * raw_pos_loss * self.pos_mult
        #     + (1 - pos_activation) * self.base_pos_loss
        # )

        # Compute DIoU loss
        raw_diou_loss = self.diou_loss(y_true, y_pred) * self.diou_mult
        # diou_activation = self.get_activation(
        #     raw_pos_loss, self.diou_threshold, self.diou_width
        # )
        # # diou_activation = tf.constant(1, tf.float32)
        # diou_loss = (
        #     diou_activation * raw_diou_loss * self.diou_mult
        #     + (1 - diou_activation) * self.base_diou_loss
        # )

        # Combine all losses
        total_loss = raw_xst_loss + raw_cls_loss + raw_pos_loss + raw_diou_loss

        return total_loss

    def get_config(self):
        config = super(YOLOv8Loss, self).get_config()
        config.update(
            {
                "square_size": self.square_size,
                "cls_threshold": self.cls_threshold,
                "pos_threshold": self.pos_threshold,
                "diou_threshold": self.diou_threshold,
                "cls_width": self._cls_width,
                "pos_width": self._pos_width,
                "diou_width": self._diou_width,
                "transition_width": self.transition_width,
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
