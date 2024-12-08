# execute from project root using
# python -m src.cv.dartsboard_finder
# to ensure relative imports work as expected

import os
import re
import cv2
import numpy as np
import pandas as pd
import pickle
import warnings
import tensorflow as tf
from rich import print
from tensorflow.keras import layers
import tensorflow_io as tfio
from scipy.ndimage import affine_transform

from ma_darts.cv.cv import extract_center

IMG_SIZE = 800


class Model:
    def get_model():
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

        def conv_block(x, n_filters):
            x = layers.Conv2D(
                n_filters,
                kernel_size=(3, 3),
                padding="same",
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D((2, 2))(x)
            return x

        filters = [16, 32, 64, 64, 64, 64, 64, 64]

        x = inputs
        for f in filters:
            x = conv_block(x, f)

        x = layers.Flatten()(x)

        dense_units = [64, 64, 16]
        for d in dense_units:
            x = layers.Dense(d, activation="relu")(x)

        output = layers.Dense(5, activation="linear")(x)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=[EllipseLoss(), tf.keras.metrics.MeanAbsoluteError()],
        )
        return model


class EllipseLoss(tf.keras.losses.Loss):
    def __init__(self, input_size: int = IMG_SIZE):
        super(EllipseLoss, self).__init__()
        self.input_size = input_size

    def call(self, y_true, y_pred):
        ellipse_true = self.generate_ellipse(y_true)
        ellipse_pred = self.generate_ellipse(y_pred)

        intersection = tf.reduce_sum(ellipse_true * ellipse_pred, axis=[1, 2])
        union = (
            tf.reduce_sum(ellipse_true, axis=[1, 2])
            + tf.reduce_sum(ellipse_pred, axis=[1, 2])
            - intersection
        )
        iou = tf.divide(intersection, union)
        print(f"{intersection=} {union=} {iou=}")

        return iou
        return 1 - iou

    def generate_ellipse(self, params):
        input_size = self.input_size

        # Create a grid for x and y coordinates
        y_grid, x_grid = tf.meshgrid(
            tf.range(input_size), tf.range(input_size), indexing="ij"
        )
        x_grid = tf.cast(x_grid, tf.float32)
        y_grid = tf.cast(y_grid, tf.float32)

        # Unpack the ellipse parameters
        cx, cy, a, b, angle = tf.split(params, 5, axis=-1)
        cx *= self.input_size
        cy *= self.input_size

        # Shift the grid to the ellipse's center
        x_shifted = x_grid[None, :, :] - cx[:, None, None]
        y_shifted = y_grid[None, :, :] - cy[:, None, None]

        # Convert angle from degrees to radians
        cos_angle = tf.cos(angle)
        sin_angle = tf.sin(angle)

        # Rotate the grid points
        x_rot = (
            x_shifted * cos_angle[:, None, None] + y_shifted * sin_angle[:, None, None]
        )
        y_rot = (
            -x_shifted * sin_angle[:, None, None] + y_shifted * cos_angle[:, None, None]
        )

        # Equation of ellipse (x_rot^2 / a^2) + (y_rot^2 / b^2) <= 1
        ellipse_equation = (x_rot / a[:, None, None]) ** 2 + (
            y_rot / b[:, None, None]
        ) ** 2
        mask = tf.cast(ellipse_equation <= 1, tf.float32)

        return mask


class Utils:
    @staticmethod
    def find_ellipse_params(img):
        if type(img) == str:
            img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        if len(img.shape) == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Threshold
        _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
        # Find contours
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contour = max(contours, key=cv2.contourArea)
        # Fit ellipse
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (a, b), theta = ellipse
        a /= 2
        b /= 2
        theta = np.deg2rad(theta)

        cx /= img.shape[1]
        cy /= img.shape[0]

        return cx, cy, a, b, theta

    def test_model(model, test_ds):
        l = EllipseLoss(input_size=IMG_SIZE)
        ress = []
        for i, (X, y_true) in enumerate(test_ds.unbatch()):
            print(i, end="\r")
            y_pred = model.predict(tf.expand_dims(X, 0), verbose=0)[0]
            mask_true = l.generate_ellipse(y_true)[0].numpy()
            mask_pred = l.generate_ellipse(y_pred)[0].numpy()

            res = X.numpy() / 10
            res[mask_pred != 0] *= 10
            ress.append(res)
        out = np.concatenate(ress, axis=0)
        out = (out * 255).astype(np.uint8)
        cv2.imwrite("dump/out.png", out)
        print("saved at dump/out.png")


class Data:
    img_size = IMG_SIZE

    class Augmentation:
        def __init__(self):
            pass

        def __call__(self, image, label):

            # Random brightness adjustment
            image = tf.image.random_brightness(image, max_delta=0.1)

            # Random contrast adjustment
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

            noise = tf.random.normal(
                shape=tf.shape(image), mean=0.0, stddev=0.1, dtype=tf.float32
            )
            image = tf.add(image, noise)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label

    def read_img(filepath):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize_with_pad(img, Data.img_size, Data.img_size)
        return img

    def preprocess_image(img):
        gray = tfio.experimental.color.rgb_to_grayscale(img)
        return gray

    @staticmethod
    def extract_line_params(
        sample_info: pd.Series,
    ) -> tuple[float, float, float, float]:
        h_line_r = sample_info["h_line_r"]
        h_line_theta = sample_info["h_line_theta"]
        v_line_r = sample_info["v_line_r"]
        v_line_theta = sample_info["v_line_theta"]
        return h_line_r, h_line_theta, v_line_r, v_line_theta

    @staticmethod
    def extract_ellipse_params(
        sample_info: pd.Series,
    ) -> tuple[float, float, float, float, float]:
        """
        Outputs:
            cy:     y position ratio 0..1
            cx:     x position ratio 0..1
            a:      ellipse width in pixels 0..w
            b:      ellipse height in pixels 0..h
            theta:  rotation in percent 0..1
        """
        img_w = sample_info["img_width"]
        img_h = sample_info["img_height"]
        cx = np.float32(sample_info["ellipse_cx"]) / img_w
        cy = np.float32(sample_info["ellipse_cy"]) / img_h
        w = np.float32(sample_info["ellipse_w"]) / img_w
        h = np.float32(sample_info["ellipse_h"]) / img_h
        theta = np.float32(sample_info["ellipse_theta"]) / 360
        return cy, cx, w, h, theta

    @staticmethod
    def load_sample(filepath: str | bytes):
        if type(filepath) == bytes:
            filepath = filepath.decode("utf-8")

        # Load image
        input_img = Data.read_img(filepath=os.path.join(filepath, "render.png"))

        # Load info
        with open(os.path.join(filepath, "info.pkl"), "rb") as f:
            sample_info = pickle.load(f)

        # Extract information
        line_params = Data.extract_line_params(sample_info)
        ellipse_params = Data.extract_ellipse_params(sample_info)
        outputs = line_params + ellipse_params

        return input_img, tf.convert_to_tensor(outputs, tf.float32)

    @staticmethod
    def get_dataset(
        data_dir: str = "data/generation/out/",
        augment: bool = True,
        shuffle: bool = True,
    ):
        sample_ids = [f for f in os.listdir(data_dir) if f.isnumeric()]
        if shuffle:
            np.random.shuffle(sample_ids)
        else:
            sample_ids = sorted(sample_ids, key=lambda x: int(x))

        image_paths = [os.path.join(data_dir, f) for f in sample_ids]
        ds = tf.data.Dataset.from_tensor_slices(image_paths)  # sample directory strings

        ds = ds.map(
            lambda filepath: tf.numpy_function(
                func=Data.load_sample,
                inp=[filepath],
                Tout=[tf.float32, tf.float32],
            ),
            num_parallel_calls=1,  # tf.data.AUTOTUNE,
        )  # (800, 800, 3), (9,)
        ds = ds.map(
            lambda img, params: (
                tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3]),
                tf.ensure_shape(params, [9]),
            )
        )
        ds = ds.map(lambda X, y: (Data.preprocess_image(X), y))
        # if augment:
        #     ds = ds.map(Data.Augmentation())
        # ds = ds.cache()
        # ds = ds.batch(8)
        # ds = ds.prefetch(8)
        return ds


ds = Data.get_dataset(shuffle=False)

model = Model.get_model()
model.summary(120)
exit()

X, y = next(iter(ds))
epochs = 1000
try:
    for epoch in range(epochs):
        model.fit(
            ds,
            epochs=epoch + 1,
            initial_epoch=epoch,
            verbose=1,
            validation_data=val_ds,
        )
except KeyboardInterrupt:
    print("\nTraining interrupted.")

Utils.test_model(model, val_ds.concatenate(ds.take(1)))

# ------------------------------------------------------------------------


class DidNotWorkButIDontWantToDeleteItYet:
    def get_bbox(img):
        rows = np.any(img, axis=1)
        y0 = np.argmax(rows)
        y1 = img.shape[0] - np.argmax(rows[::-1])
        h = y1 - y0

        cols = np.any(img, axis=0)
        x0 = np.argmax(cols)
        x1 = img.shape[1] - np.argmax(cols[::-1])
        w = x1 - x0
        return w, h

    def get_ellipse_mask(image_shape, cy, cx, minor_axis, major_axis, theta):
        """
        cy: y center 0..1
        cx: x center 0..1
        minor_axis: ellipse width 0..w
        major_axis: ellipse height 0..h
        theta: rotation in percent 0..1
        """
        img_w, img_h = image_shape
        # Convert to correct ranges
        cy = tf.cast(cy * img_h, tf.float32)
        cx = tf.cast(cx * img_w, tf.float32)
        h = tf.cast(major_axis * img_h, tf.float32)
        w = tf.cast(minor_axis * img_w, tf.float32)
        theta = tf.cast(np.radians(theta * 360), tf.float32)

        # Get minor and major axes
        cos_t = tf.cos(theta)
        sin_t = tf.sin(theta)
        a = w * cos_t + h * sin_t
        b = w * sin_t + h * cos_t
        print(a.numpy(), b.numpy())

        # Create grid of coordinates
        y = tf.range(img_h, dtype=tf.float32)
        x = tf.range(img_w, dtype=tf.float32)
        xx, yy = tf.meshgrid(x, y)

        # Translate to center
        xx_c = xx - cx
        yy_c = yy - cy

        # Rotate grid by -theta
        cos_t = tf.cos(theta)
        sin_t = tf.sin(theta)
        xx_rot = cos_t * xx_c + sin_t * yy_c
        yy_rot = -sin_t * xx_c + cos_t * yy_c

        # Normalize with w and h
        ellipse_eq = (xx_rot / (w / 2)) ** 2 + (yy_rot / (h / 2)) ** 2

        mask = tf.cast(ellipse_eq <= 1, tf.float32)
        return tf.cast(mask * 255, tf.uint8)

        # Translate to center
        y_centered = y - cy
        x_centered = x - cx

        # apply rotation
        cos_theta = tf.cos(theta)
        sin_theta = tf.sin(theta)
        y_rot = -sin_theta * x_centered + cos_theta * y_centered
        x_rot = cos_theta * x_centered + sin_theta * y_centered

        # Compute ellipse
        ellipse_mask = ((x_rot / w) ** 2 + (y_rot / h) ** 2) <= 1.0

        return tf.cast(ellipse_mask, tf.uint8)

    def draw_correct_ellipse(image_shape, cy, cx, bbox_width, bbox_height, theta):
        img_h, img_w = image_shape
        dst_ratio = bbox_width / bbox_height
        max_d = int(np.ceil(np.sqrt(bbox_width**2 + bbox_height**2)))
        c = max_d / 2

        # Angle
        theta_rad = theta * 2 * np.pi
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        def get_circle(d=max_d):
            r = d / 2
            img = np.zeros((d, d), np.uint8)
            y, x = np.ogrid[:d, :d]
            mask = (x - r) ** 2 + (y - r) ** 2 <= r**2
            img[mask] = 255
            return img

        def transform_img(img, s):

            # Scale along x
            M_scale = np.array(
                [
                    [1, 0, 0],
                    [0, 1 / s, 0],
                    [0, 0, 1],
                ]
            )

            M_trans2 = np.array(
                [
                    [1, 0, -c],
                    [0, 1, -c],
                    [0, 0, 1],
                ]
            )

            M_rot = np.array(
                [
                    [cos_t, -sin_t, 0],
                    [sin_t, cos_t, 0],
                    [0, 0, 1],
                ]
            )

            M_trans1 = np.array(
                [
                    [1, 0, c],
                    [0, 1, c],
                    [0, 0, 1],
                ]
            )
            out = affine_transform(img, M_trans1 @ M_scale @ M_rot @ M_trans2)
            return out

        def find_ellipse_squish():
            circle = get_circle()
            lower = 0
            upper = 1
            epsilon = 1e-3
            max_runs = 32

            diffs = []
            run = 0
            while run < max_runs:
                squish = (lower + upper) / 2
                assert squish > 1e-2, "Could not figure out ellipse shape"

                ellipse = transform_img(circle, squish)
                ellipse_w, ellipse_h = Utils.get_bbox(ellipse)
                ratio = ellipse_w / ellipse_h
                diff = abs(ratio - dst_ratio)
                diffs.append(diff)
                if len(diffs) > 5 and sorted(diffs) == diffs:
                    # if there is no improvement after 3 tries
                    # I assume we're going into the wrong direction
                    print("REVERSE")
                    diffs = []
                    lower = 1
                    upper = 0
                    run = 0
                    continue

                print(f"dst_ratio={dst_ratio:.02f}, ratio={ratio:.02f}")
                print(f"{squish=}, {diff=}", end="\n")
                cv2.imshow("", ellipse)
                cv2.waitKey()

                if diff < epsilon:
                    break

                if ratio < dst_ratio:
                    lower = squish
                else:
                    upper = squish
                run += 1

            scale = (ellipse_w / bbox_width + ellipse_h / bbox_height) / 2
            return squish, scale

        circle = get_circle()
        squish, scale = find_ellipse_squish()

        M_trans1 = np.array(
            [
                [1, 0, c],
                [0, 1, c],
                [0, 0, 1],
            ]
        )
        M_squish = np.array(
            [
                [1, 0, 0],
                [0, 1 / squish, 0],
                [0, 0, 1],
            ]
        )
        M_rot = np.array(
            [
                [cos_t, -sin_t, 0],
                [sin_t, cos_t, 0],
                [0, 0, 1],
            ]
        )
        M_scale = np.array(
            [
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1],
            ]
        )
        M_trans2 = np.array(
            [
                [1, 0, -cy],
                [0, 1, -cx],
                [0, 0, 1],
            ]
        )
        circle = np.pad(
            circle,
            (
                (0, max(img_h - max_d, 0)),
                (0, max(img_w - max_d, 0)),
            ),
        )
        out = affine_transform(circle, M_trans1 @ M_squish @ M_rot @ M_scale @ M_trans2)

        w, h = Utils.get_bbox(out)
        error = 1 - (w * h) / (bbox_width * bbox_height)
        print(f"{error=:.04f}")

        out = out[:img_h, :img_w]

        return out


while True:

    img_w = 800
    img_h = 800
    cx = 400
    cy = 400
    a = np.random.randint(20, 350)
    b = np.random.randint(20, 350)
    a, b = min(a, b), max(a, b)
    angle = np.random.uniform(0, 180)

    # a = 77
    # b = 159
    # angle= 372
    print(a, b, angle)

    # Get ellipse
    ellipse = cv2.ellipse(
        np.zeros((img_h, img_w), np.uint8),
        center=(cx, cy),
        axes=(a, b),
        angle=angle,
        startAngle=0,
        endAngle=360,
        color=(255, 255, 255),
        thickness=-1,
    )

    # Get ellipse params
    ellipse = cv2.threshold(ellipse, 127, 255, cv2.THRESH_BINARY)[1]
    points = np.column_stack(np.where(ellipse.transpose() > 0))
    hull = cv2.convexHull(points)[:, 0]
    (cx, cy), (_w, _h), theta = cv2.fitEllipse(hull)
    w, h = DidNotWorkButIDontWantToDeleteItYet.get_bbox(ellipse)

    cv2.rectangle(
        ellipse,
        (int(cx - w / 2), int(cy - h / 2)),
        (int(cx + w / 2), int(cy + h / 2)),
        (255, 255, 255),
    )
    cv2.imshow("input", ellipse)

    res = DidNotWorkButIDontWantToDeleteItYet.draw_correct_ellipse(
        image_shape=(img_w, img_h),
        cy=cx,
        cx=cy,
        bbox_width=w,
        bbox_height=h,
        theta=angle / 360,
    )

    intersect = np.bitwise_xor(ellipse, res)
    cv2.imshow("in", intersect)
    k = cv2.waitKey()
    if k == ord("q"):
        exit()
