import tensorflow as tf
from ma_darts import img_size


class Augmentation:
    def __init__(self, img_size=img_size):
        self.img_size = tf.constant(img_size, tf.int32)

        # Image Augmentation Parameters
        self.brightness_adjust = 0.03
        self.contrast_adjust = 0.3
        self.noise_amount = 0.15
        self.saturation_amount = 0.2
        self.min_jpeg_quality = 20

    @tf.function
    def pixel_augmentation(
        self,
        img: tf.Tensor,  # (800, 800, 3)
    ):
        seed = tf.random.uniform((2,), 0, 2**15, dtype=tf.int32)

        # Adapt red and green channels
        channel_weights = tf.random.uniform((3,), 0.5, 1.0)  # * 0 + 0.5
        channel_b = img[..., 0:1] * channel_weights[0]
        channel_g = img[..., 1:2] * channel_weights[1]
        channel_r = img[..., 2:3] * channel_weights[2]
        img = tf.concat([channel_b, channel_g, channel_r], axis=-1)


        # Brightness
        img_brightness = tf.reduce_mean(img)
        brightness_lower = tf.maximum(0.1, img_brightness - self.brightness_adjust)
        brightness_upper = img_brightness + self.brightness_adjust
        brightness_target = tf.random.stateless_uniform(
            [],
            minval=brightness_lower,
            maxval=brightness_upper,
            seed=seed,
        )
        brightness_delta = brightness_target - img_brightness
        img = img + brightness_delta
        img = tf.clip_by_value(img, 0.0, 1.0)

        # Contrast
        img = tf.image.stateless_random_contrast(
            img,
            lower=1 - self.contrast_adjust,
            upper=1 + self.contrast_adjust,
            seed=seed + 1,
        )
        img = tf.clip_by_value(img, 0.0, 1.0)

        # Noise
        noise = tf.random.stateless_normal(
            shape=(self.img_size, self.img_size, 3),
            seed=seed + 2,
            mean=0.0,
            stddev=self.noise_amount,
            dtype=tf.float32,
        )
        img = img + noise
        img = tf.clip_by_value(img, 0.0, 1.0)

        # Saturation
        img = tf.image.stateless_random_saturation(
            img,
            lower=1 - self.saturation_amount * 3,
            upper=1 + self.saturation_amount,
            seed=seed + 3,
        )

        # JPEG compression
        img = tf.image.stateless_random_jpeg_quality(
            img,
            min_jpeg_quality=self.min_jpeg_quality,
            max_jpeg_quality=100,
            seed=seed + 3,
        )

        return img

    @tf.function
    def translation_matrix(self, dy=0, dx=0):
        return tf.cast(
            [
                [1, 0, -dx],
                [0, 1, -dy],
                [0, 0, 1],
            ],
            tf.float32,
        )

    @tf.function
    def rotation_matrix(
        self,
        angle: tf.Tensor,
    ):
        angle = tf.squeeze(angle)
        cos_a = tf.cos(angle)
        sin_a = tf.sin(angle)

        return tf.cast(
            [
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1],
            ],
            tf.float32,
        )

    @tf.function
    def apply_rotation_to_pos(
        self,
        pos: tf.Tensor,  # (2, 3)
        rotation: tf.Tensor,  # 0..2pi
    ):
        # Translate to origin
        pos = pos - 0.5

        # Get radii
        rs = tf.sqrt(tf.square(pos[0]) + tf.square(pos[1]))

        # Get angles
        ts = tf.atan2(pos[0], pos[1])

        # Add rotation to thetas
        ts_ = ts - rotation

        # Convert back to cartesian
        ys_ = rs * tf.sin(ts_)
        xs_ = rs * tf.cos(ts_)
        pos_ = tf.stack([ys_, xs_], axis=0)

        # Translate back from origin
        pos_ = pos_ + 0.5
        return pos_

    @tf.function
    def apply_translation_to_pos(
        self,
        pos: tf.Tensor,  # (2, 3)
        translation: tf.Tensor,  # (2,)
    ):
        translation = translation / tf.cast(self.img_size, tf.float32)
        pos = tf.stack(
            [
                pos[0] + translation[0],
                pos[1] + translation[1],
            ],
            axis=0,
        )
        return pos

    @tf.function
    def transformation_augmentation(
        self,
        img: tf.Tensor,  # (800, 800, 3)
        pos: tf.Tensor,  # (2, 3)
    ):
        # Flipping
        flips = tf.random.uniform((2,))
        if flips[0] > 0.5:
            img = tf.image.flip_left_right(img)
            pos = tf.stack([pos[0], 1 - pos[1]], axis=0)
        if flips[1] > 0.5:
            img = tf.image.flip_up_down(img)
            pos = tf.stack([1 - pos[0], pos[1]], axis=0)

        # Translate to origin
        M = tf.eye(3)
        M = tf.matmul(M, self.translation_matrix(-400, -400))

        # Rotation
        rotation_steps = tf.random.uniform((1,), 0, 20, tf.int32)  # 0..19
        rotation = tf.cast(rotation_steps, tf.float32) * tf.constant(
            0.31415, tf.float32
        )
        M = tf.matmul(M, self.rotation_matrix(rotation))
        pos = self.apply_rotation_to_pos(pos, rotation)

        # Translation
        translation = tf.random.normal((2,), mean=0.0, stddev=5.0, dtype=tf.float32)
        M = tf.matmul(M, self.translation_matrix(translation[0], translation[1]))
        pos = self.apply_translation_to_pos(pos, translation)

        # Translate back from origin
        M = tf.matmul(M, self.translation_matrix(400, 400))

        # Apply matrix to image
        M = tf.reshape(M, (9,))[:-1]
        img = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(img, 0),
            transforms=tf.expand_dims(M, 0),
            output_shape=(self.img_size, self.img_size),
            interpolation="BILINEAR",
            fill_value=0.0,
        )[0]

        # Clip ranges
        pos = tf.cast(tf.clip_by_value(pos, 0, 1), tf.float32)
        return img, pos

    @tf.function
    def __call__(
        self,
        img: tf.Tensor,  # (800, 800, 3)
        xst: tf.Tensor,  # (1, 3)
        pos: tf.Tensor,  # (2, 3)
        cls: tf.Tensor,  # (5, 3)
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        pixel_factor = tf.pow(tf.random.uniform((), 0, 1), 5)
        img = self.pixel_augmentation(img) * (1 - pixel_factor) + img * pixel_factor

        img, pos = self.transformation_augmentation(img, pos)
        return img, xst, pos, cls
