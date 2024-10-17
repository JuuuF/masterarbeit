import os
import cv2
import numpy as np
import tensorflow as tf
from rich import print
from tensorflow.keras import layers

IMG_SIZE = 512


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
        img = cv2.imread(filepath)

        # Make img square
        h, w = img.shape[:2]
        if h > w:
            padding = (h - w) // 2
            padded_img = np.pad(
                img, ((0, 0), (padding, padding), (0, 0)), mode="constant"
            )
        elif w > h:
            padding = (w - h) // 2
            padded_img = np.pad(
                img, ((padding, padding), (0, 0), (0, 0)), mode="constant"
            )
        else:
            padded_img = img

        # Resize img
        resized_img = cv2.resize(
            padded_img, (Data.img_size, Data.img_size), interpolation=cv2.INTER_LINEAR
        )

        return resized_img

    @staticmethod
    def load_image_and_mask(filepath):
        if type(filepath) == bytes:
            filepath = filepath.decode("utf-8")
        img = Data.read_img(filepath)
        img = img.astype(np.float32) / 255
        mask_path = filepath.replace(".png", "_mask.png")  # Assuming mask filenames
        mask = Data.read_img(mask_path)
        ellipse_params = Utils.find_ellipse_params(mask)
        return tf.cast(img, tf.float32), tf.cast(ellipse_params, tf.float32)

    @staticmethod
    def get_ds(data_dir="generated/", augment: bool = True):
        image_paths = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if not "mask" in f
        ]
        ds = tf.data.Dataset.from_tensor_slices(image_paths)

        ds = ds.map(
            lambda filepath: tf.numpy_function(
                func=Data.load_image_and_mask,
                inp=[filepath],
                Tout=[tf.float32, tf.float32],
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.map(
            lambda img, params: (
                tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3]),
                tf.ensure_shape(params, [5]),
            )
        )
        if augment:
            ds = ds.map(Data.Augmentation())
        ds = ds.cache()
        ds = ds.batch(8)
        ds = ds.prefetch(8)
        return ds


# print(tf.config.list_physical_devices("GPU"))
# exit()
ds = Data.get_ds()
val_ds = Data.get_ds("generated_2", augment=False)

model = Model.get_model()
model.summary(120)

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
