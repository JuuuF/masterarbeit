import tensorflow as tf
from ma_darts import img_size, radii
from ma_darts.ai.utils import split_outputs_to_xst_pos_cls


class ClassesLoss(tf.keras.losses.Loss):
    def __init__(
        self, multiplier: float = 1, from_logits: bool = True, *args, **kwargs
    ):
        super(ClassesLoss, self).__init__(*args, **kwargs)

        self.loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
            reduction=None,
            from_logits=from_logits,
        )
        self.multiplier = multiplier
        self.from_logits = from_logits

    def get_config(self):
        config = super(ClassesLoss, self).get_config()
        config.update(
            {
                "multiplier": self.multiplier,
                "from_logits": self.from_logits,
            }
        )
        return config

    # @tf.function
    def call(
        self,
        y_true,  # (bs, s, s, 8, 3)
        y_pred,
    ):

        # (bs, s, s, 1, 3), (bs, s, s, 5, 3)
        xst_true, _, cls_true = split_outputs_to_xst_pos_cls(y_true)
        xst_pred, _, cls_pred = split_outputs_to_xst_pos_cls(y_pred)

        batch_size = tf.shape(cls_true)[0]
        n_classes = tf.shape(cls_true)[-2]
        cls_true = tf.transpose(cls_true, (0, 1, 2, 4, 3))  # (bs, s, s, 3, 5)
        cls_pred = tf.transpose(cls_pred, (0, 1, 2, 4, 3))
        xst_true = tf.transpose(xst_true, (0, 1, 2, 4, 3))  # (bs, s, s, 3, 1)
        cls_true = tf.reshape(cls_true, (batch_size, -1, n_classes))  # (bs, s*s*3, 5)
        cls_pred = tf.reshape(cls_pred, (batch_size, -1, n_classes))

        # Label Smoothing
        n_classes = 5
        epsilon = 0.01
        cls_true = (1 - epsilon * (n_classes + 1)) * cls_true + epsilon

        # Extract true class masks
        positive_mask = tf.cast(xst_true > 0.5, tf.float32)  # (bs, s, s, 3, 1)
        positive_mask = tf.reshape(positive_mask, (batch_size, -1))  # (bs, s*s*3)
        cls_loss = self.loss_fn(
            cls_true, cls_pred, sample_weight=positive_mask
        )  # (bs,)
        return cls_loss * tf.constant(self.multiplier, tf.float32)  # (bs,)


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
        exit()
