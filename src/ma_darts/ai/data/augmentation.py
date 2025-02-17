import tensorflow as tf


class Augmentation:
    def __init__(self):
        pass

    def pixel_augmentation(
        self,
        img: tf.Tensor,  # (800, 800, 3)
    ):
        return img

    def transformation_augmentation(
        self,
        img: tf.Tensor,  # (800, 800, 3)
        pos: tf.Tensor,  # (2, 3)
    ):
        return img, pos

    def __call__(
        self,
        img: tf.Tensor,  # (800, 800, 3)
        pos: tf.Tensor,  # (2, 3)
        cls: tf.Tensor,  # (6, 3)
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        img = self.pixel_augmentation(img)

        img, pos = self.transformation_augmentation(img, pos)
        return img, pos, cls
