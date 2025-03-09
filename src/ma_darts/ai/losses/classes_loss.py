import tensorflow as tf
from ma_darts import img_size, radii


class ClassesLoss(tf.keras.losses.Loss):
    def __init__(self, multiplier: float = 1, *args, **kwargs):
        super(ClassesLoss, self).__init__(*args, **kwargs)

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.multiplier = multiplier

        # Get weighing factors based on dart field areas
        total_area = self.get_ring_area(0, radii["r_do"])
        inner_bull = self.get_ring_area(0, radii["r_bi"])
        outer_bull = self.get_ring_area(radii["r_bi"], radii["r_bo"])
        inner_singles = self.get_ring_area(radii["r_bo"], radii["r_ti"])
        triple_ring = self.get_ring_area(radii["r_ti"], radii["r_to"])
        outer_singles = self.get_ring_area(radii["r_to"], radii["r_di"])
        double_ring = self.get_ring_area(radii["r_di"], radii["r_do"])
        red_area = inner_bull + triple_ring / 2 + double_ring / 2
        green_area = outer_bull + triple_ring / 2 + double_ring / 2
        black_area = inner_singles / 2 + outer_singles / 2
        white_area = inner_singles / 2 + outer_singles / 2
        out_area = self.get_ring_area(radii["r_do"], 400)

        self.hidden_factor = tf.constant(1, tf.float32)
        self.black_factor = tf.constant(total_area / black_area, tf.float32)
        self.white_factor = tf.constant(total_area / white_area, tf.float32)
        self.red_factor = tf.constant(total_area / red_area, tf.float32)
        self.green_factor = tf.constant(total_area / green_area, tf.float32)
        self.out_factor = tf.constant(total_area / out_area, tf.float32)

    def get_config(self):
        config = super(ClassesLoss, self).get_config()
        config.update(
            {
                "multiplier": self.multiplier,
            }
        )
        return config

    def get_ring_area(self, r_inner, r_outer):
        a_inner = 3.14159 * r_inner**2
        a_outer = 3.14159 * r_outer**2
        return a_outer - a_inner

    # @tf.function
    def call(
        self,
        y_true,  # (bs, s, s, 8, 3)
        y_pred,
    ):

        # Classes
        cls_true = y_true[..., 2:, :]  # (bs, s, s, 6, 3)
        cls_pred = y_pred[..., 2:, :]
        cls_true = tf.transpose(cls_true, (0, 1, 2, 4, 3))  # (bs, s, s, 3, 6)
        cls_pred = tf.transpose(cls_pred, (0, 1, 2, 4, 3))
        batch_size = tf.shape(cls_true)[0]
        n_classes = tf.shape(cls_true)[-1]
        cls_true = tf.reshape(cls_true, (batch_size, -1, n_classes))
        cls_pred = tf.reshape(cls_pred, (batch_size, -1, n_classes))

        # Get class counts per batch
        n_hidden = 1875 - tf.reduce_sum(cls_true[..., 0], axis=1)  # (bs,)
        n_black = tf.reduce_sum(cls_true[..., 1], axis=1)
        n_white = tf.reduce_sum(cls_true[..., 2], axis=1)
        n_red = tf.reduce_sum(cls_true[..., 3], axis=1)
        n_green = tf.reduce_sum(cls_true[..., 4], axis=1)
        n_out = tf.reduce_sum(cls_true[..., 5], axis=1)
        # Get weights per batch
        weight_hidden = n_hidden * self.hidden_factor  # (bs,)
        weight_black = n_black * self.black_factor
        weight_white = n_white * self.white_factor
        weight_red = n_red * self.red_factor
        weight_green = n_green * self.green_factor
        weight_out = n_out * self.out_factor
        # Combine batch weights
        batch_weights = (
            weight_hidden
            + weight_black
            + weight_white
            + weight_red
            + weight_green
            + weight_out
        )  # (bs,)
        batch_weights = tf.expand_dims(batch_weights, -1)

        batch_weights = (
            tf.cast(batch_size, tf.float32)
            * batch_weights
            / tf.reduce_sum(batch_weights)
        )

        loss = self.loss_fn(cls_true, cls_pred, sample_weight=batch_weights)
        return loss * tf.constant(self.multiplier, tf.float32)


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
