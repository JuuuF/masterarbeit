import tensorflow as tf


class Augmentation:
    def __init__(self):
        pass

    def __call__(
        self,
        img: tf.Tensor,  # (800, 800, 3)
        pos: tf.Tensor,  # (2, 3)
        cls: tf.Tensor,  # (6, 3)
    ):
        return img, pos, cls
