import tensorflow as tf
from ma_darts import img_size
from ma_darts.ai.utils import get_grid_existences


class PositionsLoss(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        super(PositionsLoss, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super(PositionsLoss, self).get_config()
        return config

    # @tf.function
    def call(
        self,
        y_true,  # (bs, s, s, 8, 3)
        y_pred,
    ):

        # Get true existences
        xst_true = get_grid_existences(y_true)  # (bs, s, s, 1, 3)

        # Get positions
        pos_true = y_true[..., :2, :]  # (bs, s, s, 2, 3)
        pos_pred = y_pred[..., :2, :]

        # Filter by true existences
        pos_true_m = xst_true * pos_true  # (bs, s, s, 2, 3)
        pos_pred_m = xst_true * pos_pred

        n_trues = tf.reduce_sum(xst_true)
        sse = tf.reduce_sum(tf.square(pos_true_m - pos_pred_m))
        mean_sse = sse / tf.cast(tf.maximum(n_trues, 1), tf.float32)

        return mean_sse


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

        fac = 0.1
        y_p = fac * y_p + (1 - fac) * y_t

        loss = l(y_t, y_p)
        print("loss =", loss.numpy())
        import cv2

        cv2.waitKey()

        exit()
