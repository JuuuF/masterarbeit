import numpy as np
import tensorflow as tf

from ma_darts import img_size
from ma_darts.ai.losses import ExistenceLoss, ClassesLoss, PositionsLoss


class YOLOv8Loss(tf.keras.losses.Loss):
    def __init__(
        self,
        square_size: int = 50,
        class_introduction_threshold: float = np.inf,
        position_introduction_threshold: float = np.inf,
    ):
        super().__init__()
        self.xst_loss = ExistenceLoss()
        self.cls_loss = ClassesLoss()
        self.pos_loss = PositionsLoss()

        self.cls_introduction_end = tf.constant(
            class_introduction_threshold, tf.float32
        )
        self.cls_introduction_start = tf.constant(
            1.5 * class_introduction_threshold, tf.float32
        )
        self.base_cls_loss = tf.constant(10, tf.float32)

        self.pos_introduction_end = tf.constant(
            position_introduction_threshold, tf.float32
        )
        self.pos_introduction_start = tf.constant(
            1.5 * position_introduction_threshold, tf.float32
        )
        self.base_pos_loss = tf.constant(10, tf.float32)

    def get_factor(
        self,
        start,
        end,
        value,
    ):
        # Check if infinite value
        is_inf = tf.math.is_inf(start)

        # Compute factor
        diff = end - start
        fac = (value - start) / tf.where(is_inf, tf.ones_like(diff), diff)

        fac = tf.clip_by_value(fac, 0, 1)

        return tf.where(is_inf, tf.zeros_like(fac), fac)

    def get_cls_loss(
        self,
        y_true: tf.Tensor,  # (bs, s, s, 8, 3)
        y_pred: tf.Tensor,
        total_loss: tf.Tensor,  # ()
    ) -> tf.Tensor:  # ()

        cls_fac = self.get_factor(
            self.cls_introduction_start,
            self.cls_introduction_end,
            total_loss,
        )

        loss = tf.where(
            tf.equal(cls_fac, 0.0),
            self.base_cls_loss,
            cls_fac * self.cls_loss(y_true, y_pred)
            + (1 - cls_fac) * self.base_cls_loss,
        )

        return loss

    def get_pos_loss(
        self,
        y_true: tf.Tensor,  # (bs, s, s, 8, 3)
        y_pred: tf.Tensor,
        total_loss: tf.Tensor,  # ()
    ) -> tf.Tensor:  # ()

        pos_fac = self.get_factor(
            self.pos_introduction_start,
            self.pos_introduction_end,
            total_loss,
        )

        loss = tf.where(
            tf.equal(pos_fac, 0.0),
            self.base_pos_loss,
            pos_fac * self.pos_loss(y_true, y_pred)
            + (1 - pos_fac) * self.base_pos_loss,
        )
        return loss

    def call(self, y_true, y_pred):
        xst_loss = self.xst_loss(y_true, y_pred)
        total_loss = xst_loss
        # print("xst_loss", xst_loss.numpy())
        # print("total_loss", total_loss.numpy())
        # print()

        cls_loss = self.get_cls_loss(y_true, y_pred, total_loss)
        total_loss = total_loss + cls_loss
        # print("cls_loss", cls_loss.numpy())
        # print("total_loss", total_loss.numpy())
        # print()

        pos_loss = self.get_pos_loss(y_true, y_pred, total_loss)
        total_loss = total_loss + pos_loss
        # print("pos_loss", pos_loss.numpy())
        # print("total_loss", total_loss.numpy())
        # print()

        return total_loss


if __name__ == "__main__":
    import pickle

    X, y_true = pickle.load(open("dump/sample.pkl", "rb"))
    y_pred = [tf.gather(y, [3, 2, 1, 0], axis=0) for y in y_true]
    y_pred = [
        tf.clip_by_value(y + tf.random.normal(y.shape, stddev=0.02), 0, 1)
        for y in y_pred
    ]

    l = YOLOv8Loss(
        class_introduction_threshold=0.25,
        position_introduction_threshold=0.25,
    )
    for y_t, y_p in zip(y_true, y_pred):
        print("#" * 120)
        print(str(y_t.shape).center(120))
        print("#" * 120)

        error = 0.99
        y_p = error * y_p + (1 - error) * y_t

        loss = l(y_t, y_p)
        print("loss =", loss.numpy())
