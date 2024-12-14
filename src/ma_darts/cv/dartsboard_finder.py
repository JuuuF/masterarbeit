import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import re
import cv2
import numpy as np
import pandas as pd
import pickle
import shutil
from rich import print
from datetime import datetime

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras import layers
from tensorflow.keras import callbacks as tf_callbacks

from ma_darts.cv.cv import extract_center
from ma_darts.ai import callbacks as ma_callbacks
from ma_darts.ai.training import train_loop

IMG_SIZE = 800
BATCH_SIZE = 32
model_input = None
# model_input = "dump/ellipse_model.keras"


class Model:
    def get_model(
        filepath: str | None = None,
        n_input_channels: int = 1,
        n_outputs: int = 5,
    ):
        if filepath is not None and os.path.exists(filepath):
            model = tf.keras.models.load_model(filepath, compile=False)
        else:
            inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, n_input_channels))

            def conv_block(x: tf.Tensor, n_filters: int):
                x = layers.Conv2D(
                    n_filters,
                    kernel_size=(3, 3),
                    padding="same",
                )(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
                x = layers.MaxPooling2D((2, 2))(x)
                x = layers.Dropout(0.1)(x)
                return x

            filters = [16, 32, 64, 128, 256]
            dense_units = [128, 64, 32]

            # Convolutional
            x = inputs
            for f in filters:
                x = conv_block(x, f)

            # Flatten
            # x = layers.Flatten()(x)
            x = layers.GlobalAveragePooling2D()(x)

            # Dense
            for d in dense_units:
                x = layers.Dense(d, activation="relu")(x)

            output = layers.Dense(n_outputs, activation="linear")(x)

            model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer="adam",
            # loss="mse",
            loss=EllipseParamLoss(),
            metrics=[
                # EllipseIoULoss(),
                tf.keras.metrics.MeanSquaredError(),
                tf.keras.metrics.RootMeanSquaredError(),
                # tf.keras.metrics.MeanAbsoluteError(),
            ],
        )
        return model

    def fit_model(model, ds, val_ds, callbacks, epochs=1000):
        # train_loop(
        #     model,
        #     train_data=ds,
        #     epochs=1000,
        #     val_data=val_ds,
        #     callbacks=callbacks,
        # )
        try:
            model.fit(
                ds,
                epochs=epochs,
                verbose=1,
                validation_data=val_ds,
                callbacks=callbacks,
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted.")

        # Load best model checkpoint weights
        cp = Utils.get_best_model_checkpoint()
        if cp:
            cp_model = tf.keras.models.load_model(cp, compile=False)
            cp_model.save_weights("temp.weights.h5")
            model.load_weights("temp.weights.h5")
            os.remove("temp.weights.h5")
            del cp_model
        return model

    def test_model(model, val_ds):
        def draw_ellipse(img, cy, cx, w, h, theta, color=(255, 255, 255)):
            cy = int(cy * X.shape[0])
            cx = int(cx * X.shape[1])
            d = np.sqrt(X.shape[0] ** 2 + X.shape[1] ** 2)
            w = int(w * d)
            h = int(h * d)
            theta = float(theta * 360)

            img = cv2.ellipse(
                img,
                (cx, cy),
                (max(w // 2, 1), max(h // 2, 1)),
                theta,
                0,
                360,
                color,
            )
            return img

        X_batch, y_batch = next(iter(val_ds.take(1)))
        y_batch_ = model.predict(X_batch)

        for X, y, y_ in zip(X_batch, y_batch, y_batch_):
            # Data Conversion
            X = X.numpy()
            y = y.numpy()
            X = np.uint8(X[:, :, 0] * 255)
            X = cv2.cvtColor(X, cv2.COLOR_GRAY2BGR)

            # Draw Ellipses
            X = draw_ellipse(X, *y, color=(255, 0, 0))
            X = draw_ellipse(X, *y_, color=(0, 255, 0))

            # Get Line
            cy = int(y[0] * Data.img_size)
            cx = int(y[1] * Data.img_size)
            cy_ = int(y_[0] * Data.img_size)
            cx_ = int(y_[1] * Data.img_size)
            cv2.line(X, (cx, cy), (cx_, cy_), (200, 200, 200), lineType=cv2.LINE_AA)

            # Show Image
            cv2.imshow("", X)
            if cv2.waitKey() == ord("q"):
                break


class EllipseParamLoss(tf.keras.losses.Loss):
    def __init__(self, input_size: int = IMG_SIZE):
        super(EllipseParamLoss, self).__init__()
        self.input_size = input_size
        self.d = np.sqrt(2 * (self.input_size**2))

    def call(self, y_true, y_pred):
        # Unpack the true and predicted values
        cy, cx, w, h, theta = tf.unstack(y_true, axis=-1)
        cy_, cx_, w_, h_, theta_ = tf.unstack(y_pred, axis=-1)

        # Compute the differences for the ellipse parameters
        delta_cy = (cy - cy_) * self.input_size
        delta_cx = (cx - cx_) * self.input_size
        delta_w = (w - w_) * self.d
        delta_h = (h - h_) * self.d
        # the angle is special since it is a continuumm
        delta_theta = (
            tf.minimum(tf.abs(theta - theta_), tf.abs(theta + 0.5 - theta_)) * 360
        )

        # Square individual errors
        # delta_cy = delta_cy ** 2
        # delta_cx = delta_cx ** 2
        # delta_w = delta_w ** 2
        # delta_h = delta_h ** 2
        # delta_theta = delta_theta ** 2

        # Compute the loss
        loss = tf.reduce_mean(
            abs(delta_cy)
            + abs(delta_cx)
            + abs(delta_w)
            + abs(delta_h)
            + delta_theta / 2  # half thetas
        )
        return loss


class EllipseIoULoss(tf.keras.losses.Loss):
    def __init__(self, input_size: int = IMG_SIZE):
        """
        Initializes the loss function.
        :param grid_size: Size of the grid used for approximating the IoU.
        """
        super(EllipseIoULoss, self).__init__()
        self.input_size = tf.cast(input_size, tf.int32)
        self.d = tf.sqrt(2.0 * (tf.cast(self.input_size, tf.float32) ** 2))

    def unnormalize(self, cy, cx, w, h, theta):
        cy *= tf.cast(self.input_size, tf.float32)
        cx *= tf.cast(self.input_size, tf.float32)
        w *= self.d
        h *= self.d
        theta *= 2 * 3.14159
        return cy, cx, w, h, theta

    def ellipse_mask(self, cy, cx, w, h, theta):
        """
        This whole function is really scuffed. But it works. Don't touch it!
        """
        y = tf.linspace(0, self.input_size, self.input_size)
        x = tf.linspace(0, self.input_size, self.input_size)
        yy, xx = tf.meshgrid(y, x)
        yy = tf.cast(yy, tf.float32)
        xx = tf.cast(xx, tf.float32)

        # Translate to midpoint
        y_shifted = yy - cx
        x_shifted = xx - cy

        # Rotate
        cos_t = tf.cos(theta)
        sin_t = tf.sin(theta)
        x_rot = -x_shifted * cos_t + y_shifted * sin_t
        y_rot = x_shifted * sin_t + y_shifted * cos_t

        # Inside ellipse condition
        mask = ((x_rot / (h / 2)) ** 2 + (y_rot / (w / 2)) ** 2) <= 1.0
        return tf.cast(mask, tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Unpack ground truth and predicted ellipses
        ellipse_true = tf.unstack(y_true, axis=-1)
        ellipse_pred = tf.unstack(y_pred, axis=-1)

        ellipse_true = self.unnormalize(*ellipse_true)
        ellipse_pred = self.unnormalize(*ellipse_pred)

        # Create binary masks for the ellipses
        mask_true = self.ellipse_mask(*ellipse_true)
        mask_pred = self.ellipse_mask(*ellipse_pred)

        # Compute intersection and union
        intersection = tf.reduce_sum(mask_true * mask_pred)
        union = tf.reduce_sum(mask_true) + tf.reduce_sum(mask_pred) - intersection

        # Compute IoU
        iou = intersection / (union + 1e-6)

        # Loss is 1 - IoU
        return 1 - iou


class EllipseBBIoULoss(tf.keras.losses.Loss):
    def __init__(self, input_size: int = IMG_SIZE):
        super(EllipseBBIoULoss, self).__init__()
        self.input_size = tf.cast(input_size, tf.int32)
        self.d = tf.sqrt(2.0 * (tf.cast(self.input_size, tf.float32) ** 2))

    def unnormalize(self, cy, cx, w, h, theta):
        cy *= tf.cast(self.input_size, tf.float32)
        cx *= tf.cast(self.input_size, tf.float32)
        w *= self.d
        h *= self.d
        theta *= 2 * 3.14159
        return cy, cx, w, h, theta

    def extract_corners(self, cy, cx, w, h, theta):
        # TODO
        pass

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Unpack ground truth and predicted ellipses
        ellipse_true = tf.unstack(y_true, axis=-1)
        ellipse_pred = tf.unstack(y_pred, axis=-1)

        # Un-Normalize parameters
        ellipse_true = self.unnormalize(*ellipse_true)
        ellipse_pred = self.unnormalize(*ellipse_pred)

        # Create binary masks for the ellipses
        corners_true = self.extract_corners(*ellipse_true)
        corners_pred = self.extract_corners(*ellipse_pred)

        # Compute intersection and union
        intersection = tf.reduce_sum(mask_true * mask_pred)
        union = tf.reduce_sum(mask_true) + tf.reduce_sum(mask_pred) - intersection

        # Compute IoU
        iou = intersection / (union + 1e-6)

        # Loss is 1 - IoU
        return 1 - iou


# l = EllipseBBIoULoss()

# ellipse_true = np.array((0.5, 0.5, 0.1, 0.1, 0.125), np.float32)
# ellipse_params = l.unnormalize(*ellipse_true)
# c = l.extract_corners(*ellipse_params)
# print(c)
# exit()


class Utils:
    model_checkpoint_filepath = "data/ai/checkpoints/ellipse/ellipse_epoch={epoch:05d}_val_loss={val_loss:04f}.keras"

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
        l = EllipseParamLoss(input_size=IMG_SIZE)
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

    def get_callbacks() -> list[tf.keras.callbacks.Callback]:
        callbacks = []

        # Model Checkpoint
        run_id = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        Utils.model_checkpoint_filepath = os.path.join(
            os.path.dirname(Utils.model_checkpoint_filepath),
            f"{run_id}_" + Utils.model_checkpoint_filepath.split("/")[-1],
        )

        os.makedirs(os.path.dirname(Utils.model_checkpoint_filepath), exist_ok=True)
        model_checkpoint = ma_callbacks.ModelCheckpoint(
            filepath=Utils.model_checkpoint_filepath,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            initial_value_threshold=300,
            max_saves=10,
        )
        callbacks.append(model_checkpoint)

        # History Plotter
        history_plotter_filepath = "data/ai/history.png"
        os.makedirs(os.path.dirname(history_plotter_filepath), exist_ok=True)
        history_plotter = ma_callbacks.HistoryPlotter(
            filepath=history_plotter_filepath,
            update_on_batches=False,
        )
        callbacks.append(history_plotter)

        return callbacks

    def get_best_model_checkpoint():
        checkpoint_dir = os.path.dirname(Utils.model_checkpoint_filepath)
        filenames = ""
        basename = Utils.model_checkpoint_filepath.split("/")[-1]
        while basename:
            char = basename[0]
            if char == "{":
                filenames += "([0-9]|\.)+"
                while basename[0] != "}":
                    basename = basename[1:]
                basename = basename[1:]
                continue

            if char == ".":
                filenames += "\."
                basename = basename[1:]
                continue

            filenames += basename[0]
            basename = basename[1:]

        files = [
            os.path.join(checkpoint_dir, f)
            for f in os.listdir(checkpoint_dir)
            if re.match(filenames, f)
        ]
        if not files:
            return None

        def extract_number(f):
            f = f.split("val_loss=")[-1]
            i = 0
            found_dot = False
            while True:
                char = f[i]
                if char.isnumeric():
                    i += 1
                    continue
                if char == ".":
                    if found_dot:
                        break
                    found_dot = True
                    i += 1
                    continue
                break
            f = float(f[:i])
            return f

        files = sorted(files, key=extract_number)
        best_file = files[0]
        return best_file


class Data:
    img_size = IMG_SIZE

    class Augmentation:
        def __init__(self):
            pass

        def __call__(self, image, outputs):

            # Random brightness adjustment
            image = tf.image.random_brightness(image, max_delta=0.1)

            # Random contrast adjustment
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

            # Add noise
            noise_amount = tf.random.uniform(shape=(1,), minval=0.001, maxval=0.1)
            noise = tf.random.normal(
                shape=tf.shape(image), mean=0.0, stddev=noise_amount, dtype=tf.float32
            )
            image = tf.add(image, noise)
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, outputs

    def read_img(filepath):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize_with_pad(img, Data.img_size, Data.img_size)
        img /= 255
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
        out_w = Data.img_size
        out_h = Data.img_size

        # Get ellipse values
        cx = np.float32(sample_info["ellipse_cx"])  # px orig
        cy = np.float32(sample_info["ellipse_cy"])  # px orig
        w = np.float32(sample_info["ellipse_w"])  # px orig
        h = np.float32(sample_info["ellipse_h"])  # px orig
        theta = np.float32(sample_info["ellipse_theta"])  # deg

        # Adjust axes
        downscale = min(out_h / img_h, out_w / img_w)
        h *= downscale  # px out
        w *= downscale  # px out

        # Adjust centers
        cy *= downscale  # px out
        cx *= downscale  # px out
        if img_w < img_h:
            # x padding
            scaled_w = downscale * img_w
            pad_w = (out_w - scaled_w) / 2
            cx += pad_w
        elif img_w > img_h:
            # y padding
            scaled_h = downscale * img_h
            pad_h = (out_h - scaled_h) / 2
            cy += pad_h

        # Normalize
        cy /= out_h  # rel pos out
        cx /= out_w  # rel pos out
        w /= np.sqrt(
            Data.img_size**2 + Data.img_size**2
        )  # normalized by image disgonal
        h /= np.sqrt(Data.img_size**2 + Data.img_size**2)
        theta /= 360  # 0..1

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
        # line_params = Data.extract_line_params(sample_info)
        ellipse_params = Data.extract_ellipse_params(
            sample_info,
        )  # (cy, cx, w, h, theta)
        # outputs = line_params + ellipse_params
        outputs = ellipse_params

        return input_img, tf.convert_to_tensor(outputs, tf.float32)

    @staticmethod
    def get_dataset(
        data_dir: str = "data/generation/out/",
        augment: bool = True,
        shuffle: bool = True,
        batch_size: int = BATCH_SIZE,
    ):
        # Collect Files
        sample_ids = [f for f in os.listdir(data_dir) if f.isnumeric()]
        # sample_ids = sample_ids[:1024]  # XXX
        if shuffle:
            np.random.shuffle(sample_ids)
        else:
            sample_ids = sorted(sample_ids, key=lambda x: int(x))
        image_paths = [os.path.join(data_dir, f) for f in sample_ids]
        ds = tf.data.Dataset.from_tensor_slices(image_paths)  # sample directory strings

        # Load Samples
        ds = ds.map(
            lambda filepath: tf.numpy_function(
                func=Data.load_sample,
                inp=[filepath],
                Tout=[tf.float32, tf.float32],
            ),
            num_parallel_calls=1,  # tf.data.AUTOTUNE,
        )  # (800, 800, 3), (5,)

        # Set Shapes
        ds = ds.map(
            lambda img, params: (
                tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3]),
                tf.ensure_shape(params, [5]),
            )
        )

        # Preprocess Input Image
        ds = ds.map(lambda X, y: (Data.preprocess_image(X), y))  # (800, 800, 1), (5,)

        # Cache Data
        cache_dir = "data/cache/datasets/" + data_dir.replace("/", "-")
        if cache_dir[-1] == "-":
            cache_dir = cache_dir[:-1]

        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        ds = ds.cache(cache_dir)

        # Shuffle Dataset
        if shuffle:
            ds = ds.shuffle(batch_size * 3)

        # Add Augmentation
        if augment:
            ds = ds.map(Data.Augmentation())

        # Batch + Prefetch
        ds = ds.batch(batch_size)
        ds = ds.prefetch(8)
        return ds


# Data y order: (cy, cx, w, h, theta)
ds = Data.get_dataset(
    data_dir="data/generation/out",
    shuffle=True,
    augment=True,
)
val_ds = Data.get_dataset(
    data_dir="data/generation/out_val/",
    shuffle=False,
    augment=False,
)


def extract_sample():
    sample_path = "data/generation/out/53/"

    def draw_ellipse(img, cy, cx, w, h, theta):
        ellipse = cv2.ellipse(
            img * 0,
            (int(cx), int(cy)),
            (int(w / 2), int(h / 2)),
            theta,
            0,
            360,
            (255, 255, 255),
            thickness=10,
            lineType=cv2.LINE_AA,
        )
        img = cv2.addWeighted(img, 0.7, ellipse, 0.5, 0)
        cv2.circle(img, (int(cx), int(cy)), 9, (255, 0, 0), -1)
        cv2.circle(img, (int(cx), int(cy)), 5, (255, 255, 255), -1)
        cv2.circle(img, (int(cx), int(cy)), 1, (255, 0, 0), -1)
        return img

    # Load DS data
    img = Data.read_img(sample_path + "render.png").numpy()
    sample_info = pickle.load(open(sample_path + "info.pkl", "rb"))
    cy, cx, w, h, theta = Data.extract_ellipse_params(sample_info)

    # Un-normalize DS data
    cy *= Data.img_size
    cx *= Data.img_size
    theta *= 360

    print(img.shape)
    print(cy, cx, w, h, theta)

    img_ds = draw_ellipse(img, cy, cx, w, h, theta)
    cv2.imshow("extracted", img_ds)

    # Load stored data
    cy, cx, w, h, theta = sample_info[
        ["ellipse_cy", "ellipse_cx", "ellipse_w", "ellipse_h", "ellipse_theta"]
    ]

    img = cv2.imread(sample_info["out_file_template"].format(filename="render.png"))
    img = draw_ellipse(img, cy, cx, w, h, theta)
    img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
    # cv2.waitKey()

    # exit()


def test_ds(ds):
    for i, (X, y) in enumerate(ds.unbatch()):
        # if i != 53:
        #     continue
        X = np.uint8(X.numpy()[:, :, 0] * 255)
        cy, cx, w, h, theta = y.numpy()
        cy = int(cy * Data.img_size)
        cx = int(cx * Data.img_size)
        w = int(w)
        h = int(h)
        theta *= 360
        print(X.shape)
        print(cy, cx, w, h, theta)
        X //= 3

        cv2.ellipse(
            X,
            (cx, cy),
            (w // 2, h // 2),
            theta,
            0,
            360,
            (255, 255, 255),
            thickness=1,
        )
        cv2.imshow("constructed", X)
        if cv2.waitKey() == ord("q"):
            return


model = Model.get_model(model_input)
model.summary(120)

callbacks = Utils.get_callbacks()

Model.fit_model(model, ds, val_ds, callbacks, epochs=1000)

# Save model
model_out_file = "data/ai/ellipse/model.keras"
os.makedirs(os.path.dirname(model_out_file), exist_ok=True)
model.save(model_out_file)

Model.test_model(model, val_ds)
exit()

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
            from scipy.ndimage import affine_transform

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
            from scipy.ndimage import affine_transform

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
