import tensorflow as tf
from ma_darts import img_size


class Augmentation:
    def __init__(self):
        self.img_size = tf.constant(img_size, tf.int32)

        # Image Augmentation Parameters
        self.brightness_adjust = 0.02
        self.contrast_adjust = 0.25
        self.noise_amount = 0.05
        self.saturation_amount = 0.15
        self.min_jpeg_quality = 25

    def pixel_augmentation(
        self,
        img: tf.Tensor,  # (800, 800, 3)
    ):
        seed = tf.random.uniform((2,), 0, 2**15, dtype=tf.int32)

        # Brightness
        img = tf.image.stateless_random_brightness(
            img,
            max_delta=self.brightness_adjust,
            seed=seed,
        )
        img = tf.clip_by_value(img, 0.0, 1.0)

        # Contrast
        img = tf.image.stateless_random_contrast(
            img,
            lower=1 - self.contrast_adjust,
            upper=1 + self.contrast_adjust,
            seed=seed,
        )
        img = tf.clip_by_value(img, 0.0, 1.0)

        # Noise
        noise = tf.random.normal(
            (self.img_size, self.img_size, 3),
            mean=0.0,
            stddev=self.noise_amount,
            dtype=tf.float32,
        )
        img = img + noise
        img = tf.clip_by_value(img, 0.0, 1.0)

        # Saturation
        img = tf.image.random_saturation(
            img,
            lower=1 - self.saturation_amount,
            upper=1 + self.saturation_amount,
        )

        # JPEG compression
        img = tf.image.random_jpeg_quality(
            img,
            min_jpeg_quality=self.min_jpeg_quality,
            max_jpeg_quality=100,
        )

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
