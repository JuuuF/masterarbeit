import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re
import cv2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from ma_darts.ai import callbacks as ma_callbacks
from ma_darts.ai.training import train_loop
from ma_darts.cv.utils import show_imgs, matrices

from shutil import rmtree
from datetime import datetime
from itertools import permutations
from tensorflow.keras import layers
from tensorflow.keras.applications import *

IMG_SIZE = 800
BATCH_SIZE = 32 if "GPU_SERVER" in os.environ.keys() else 4
clear_cache = True

train = True
predict = True


class CombinedLoss(tf.keras.Loss):

    def __init__(
        self, positions_loss=None, existence_loss=None, existence_threshold=0.5
    ):
        super().__init__()
        if positions_loss is None:
            positions_loss = tf.keras.losses.MeanSquaredError()
        self.positions_loss = positions_loss

        if existence_loss is None:
            existence_loss = tf.keras.losses.BinaryCrossentropy()
        self.existence_loss = existence_loss
        self.existence_threshold = existence_threshold

    def call(self, y_true, y_pred):

        permutation_list = list(permutations(range(3)))  # (6, 3)
        permutation_losses = []

        for perm in permutation_list:
            y_true_perm = tf.gather(y_true, perm, axis=1)  # (bs, 3, 3)
            y_pred_perm = tf.gather(y_pred, perm, axis=1)  # (bs, 3, 3)
            perm_loss = self.calculate_permutation_loss(
                y_true_perm, y_pred_perm
            )  # (1,)
            permutation_losses.append(perm_loss)

        losses = tf.stack(permutation_losses, axis=-1)  # (6,)
        min_losses = tf.reduce_min(losses, axis=-1)  # (6,)
        loss = tf.reduce_mean(min_losses)  # (1,)

        return loss

    def calculate_permutation_loss(self, y_true, y_pred):
        # Split inputs into positions and existence
        true_positions = y_true[..., :2]  # (bs, 3, 2)
        true_existence = y_true[..., -1:]  # (bs, 3, 1)

        pred_positions = y_pred[..., :2]  # (bs, 3, 2)
        pred_existence = y_pred[..., -1:]  # (bs, 3, 1)

        # Existence loss
        existence_loss = self.existence_loss(true_existence, pred_existence)  # (1,)

        # Mask out non-existing positions
        true_positions_masked = true_positions * true_existence  # (bs, 3, 2)
        pred_positions_masked = pred_positions * true_existence  # (bs, 3, 2)

        # Calculate masked positional loss
        positions_loss = self.positions_loss(
            true_positions_masked, pred_positions_masked
        )  # (1,)

        return existence_loss + positions_loss

    def compute_positions_loss_vectorized(self, y_true_pos, y_pred_pos, y_true_pres):
        batch_size = tf.shape(y_true_pos)[0]

        perms = tf.constant(list(permutations(range(3))), dtype=tf.int32)

        # Expand dimensions for pairwise comparison
        y_true_pos_exp = tf.gather(y_true_pos, perms, axis=1)  # (batch_size, 6, 3, 2)
        y_pred_pos_exp = tf.expand_dims(y_pred_pos, axis=1)  # (batch_size, 1, 3, 2)

        # Compute squared distances for all permutations
        pos_diff = y_true_pos_exp - y_pred_pos_exp  # (batch_size, 6, 3, 2)
        pos_loss = tf.reduce_sum(tf.square(pos_diff), axis=-1)  # (batch_size, 6, 3)

        # Mask position loss using presence scores
        y_true_pres_exp = tf.expand_dims(y_true_pres, axis=1)  # (batch_size, 1, 3)
        pos_loss = pos_loss * y_true_pres_exp  # (batch_size, 6, 3)

        # Sum losses across dart tips and find the minimum loss for each batch
        pos_loss = tf.reduce_sum(pos_loss, axis=-1)  # (batch_size, 6)
        min_pos_loss = tf.reduce_min(pos_loss, axis=-1)  # (batch_size,)

        # Return average position loss across the batch
        return tf.reduce_mean(min_pos_loss)


def get_model():

    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    # Input
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(scale=255, offset=0, name="input_rescaling")(inputs)

    # Backbone
    backbone = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    # backbone = MobileNetV3Large(
    #     include_top=False,
    #     weights="imagenet",
    #     input_shape=input_shape,
    #     dropout_rate=0.3,
    # )
    backbone.trainable = False
    x = backbone(x)

    # Flatten
    x = layers.GlobalAveragePooling2D()(x)

    # Head
    for i, n in enumerate([256, 128, 64]):
        x = layers.Dense(n, activation="relu", name=f"head_{i}_dense")(x)
        x = layers.Dropout(0.3, name=f"head_{i}_dropout")(x)

    # Output
    outputs = x
    outputs = layers.Dense(9, name="output_downsample")(outputs)
    outputs = layers.Reshape((3, 3))(outputs)

    # Output activations
    positions = layers.Activation("linear", name="output_positions")(outputs[:, :, :2])
    existences = layers.Activation("sigmoid", name="output_existences")(
        outputs[:, :, -1:]
    )
    outputs = layers.Concatenate(axis=-1, name="output_concatenation")(
        [positions, existences]
    )

    # Build model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


class Utils:
    model_checkpoint_filepath = (
        f"data/ai/checkpoints/darts/{datetime.now().strftime('%y_%m_%d-%H_%M')}/"
        + "epoch={epoch:05d}_val_loss={val_loss:06f}.weights.h5"
    )

    def get_callbacks():
        callbacks = []
        # History plotter
        callbacks.append(
            ma_callbacks.HistoryPlotter(
                filepath="dump/training_history.png",
                update_on="seconds",
                update_frequency=5,
                ease_curves=False,
                smooth_curves=True,
            )
        )

        # Model Checkpoint
        callbacks.append(
            ma_callbacks.ModelCheckpoint(
                filepath=Utils.model_checkpoint_filepath,
                monitor="val_loss",
                save_best_only=True,
                max_saves=10,
                save_weights_only=True,
            )
        )

        # TensorBoard
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir="data/ai/logs",
            )
        )
        return callbacks

    def get_best_model_checkpoint():
        checkpoint_dir = os.path.dirname(Utils.model_checkpoint_filepath)
        filenames = ""
        basename = Utils.model_checkpoint_filepath.split("/")[-1]

        # Convert basename to regex
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

        # Find files matching regex
        files = [
            os.path.join(checkpoint_dir, f)
            for f in os.listdir(checkpoint_dir)
            if re.match(filenames, f)
        ]
        if not files:
            return None

        # extract validation loss
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

    def read_sample_img(sample_dir: str):
        filepath = os.path.join(sample_dir, "undistort.png")
        img = tf.io.read_file(filepath)
        img = tf.image.decode_image(img, channels=3)
        assert img.shape == (
            IMG_SIZE,
            IMG_SIZE,
            3,
        ), f"Image shape must be (800, 800, 3). Received {img.shape} for image {filepath}"
        img = tf.reverse(img, axis=[-1])
        img /= 255
        return img

    def read_sample_info(sample_dir: str):
        filepath = os.path.join(sample_dir, "info.pkl")
        with open(filepath, "rb") as f:
            sample_info = pickle.load(f)
        return sample_info

    def extract_darts_positions(sample_info: pd.Series):
        # Get darts positions
        dart_positions = np.array(sample_info["dart_positions_undistort"])

        # Add existence to all found darts
        dart_positions = np.pad(
            dart_positions, ((0, 0), (0, 1)), constant_values=1
        )  # (n, 3)

        # Add missing entries for non-existing darts
        if (missing := 3 - len(dart_positions)) > 0:
            dart_positions = np.pad(
                dart_positions, ((0, missing), (0, 0)), constant_values=0
            )  # (3, 3)

        return dart_positions  # (y, x, 0/1)

    def load_sample(sample_dir: bytes | str):
        if type(sample_dir) == bytes:
            sample_dir = sample_dir.decode("utf-8")

        img = Data.read_sample_img(sample_dir)
        sample_info = Data.read_sample_info(sample_dir)
        dart_positions = Data.extract_darts_positions(sample_info)

        """ assure correct position placement: """
        # img = (img.numpy() * 255).astype(np.uint8)
        # print(dart_positions * 800)
        # import cv2
        # from ma_darts.cv.utils import show_imgs
        # for y, x, _ in dart_positions * 800:
        #     cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), 2)
        # show_imgs(img)

        return img, tf.cast(dart_positions, tf.float32)

    class Augmentation:
        def __init__(
            self,
            brightness_adjust: float = 0.05,
            contrast_adjust: float = 0.01,
            max_noise: float = 0.1,
            max_rotation_angle: float = 4,
            max_translation_amount: float = 5,
            max_scaling: float = 0.1,
        ):
            self.brightness_adjust = brightness_adjust
            self.contrast_adjust = contrast_adjust
            self.max_noise = max_noise
            self.max_rotation_angle = np.deg2rad(max_rotation_angle)
            self.max_translation_amount = max_translation_amount
            self.max_scaling = max_scaling

        def pixel_augmentation(self, img: tf.Tensor) -> tf.Tensor:
            # Random brightness
            img = tf.image.random_brightness(
                img,
                max_delta=self.brightness_adjust,
            )

            # Random contrast
            img = tf.image.random_contrast(
                img,
                lower=1 - self.contrast_adjust,
                upper=1 + self.contrast_adjust,
            )

            # Random noise
            noise_amount = tf.random.uniform(
                shape=(1,), minval=min(0.001, self.max_noise), maxval=self.max_noise
            )
            noise = tf.random.normal(
                shape=tf.shape(img), mean=0.0, stddev=noise_amount, dtype=tf.float32
            )
            img = tf.add(img, noise)
            img = tf.clip_by_value(img, 0.0, 1.0)

            return img

        def transformation_augmentation(
            self, img: tf.Tensor, positions: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor]:
            ### TODO ###
            # This is not working yet!
            return img, positions

            # # Random rotation
            # M = np.eye(3)

            # # Translate center to origin
            # M = matrices.translation_matrix(-img.shape[1] // 2, -img.shape[0] // 2) @ M

            # # Translate
            # translation = np.random.normal(
            #     size=(2,),
            #     loc=0.0,
            #     scale=self.max_translation_amount / 3,
            # )
            # translation = np.clip(
            #     translation, -self.max_translation_amount, self.max_translation_amount
            # )
            # translation[0] = 0  # XXX
            # translation[1] = 100  # XXX
            # M = matrices.translation_matrix(*translation) @ M

            # # Rotate
            # rotation_amount = np.random.normal(
            #     loc=0.0,
            #     scale=self.max_rotation_angle / 3,
            # )
            # rotation_amount = np.clip(
            #     rotation_amount, -self.max_rotation_angle, self.max_rotation_angle
            # )
            # rotation_amount = np.deg2rad(45)  # XXX
            # M = matrices.rotation_matrix(-rotation_amount) @ M

            # # Re-transform to center
            # M = matrices.translation_matrix(img.shape[1] // 2, img.shape[0] // 2) @ M

            # # Apply transformation matrix
            # img_trans = self._apply_transformation_to_image(img, M)

            # positions_trans = self._apply_transformation_to_positions(positions, M)

            # return img_trans, positions_trans

        def _apply_transformation_to_image(self, img, M):
            M_affine = tf.reshape(M[:2, :], [-1])
            M_flat = tf.cast(M_affine, tf.float32)
            M_flat = tf.concat([M_flat, [0.0, 0.0]], axis=0)

            img_transformed = tf.keras.ops.image.affine_transform(
                img,
                M_flat,
                interpolation="bilinear",
                fill_mode="constant",
            )
            # img_transformed = tf.raw_ops.ImageProjectiveTransformV3(
            #     images=tf.expand_dims(img, 0),  # as batch
            #     transforms=tf.expand_dims(M_flat, 0),  # as batch
            #     output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
            #     interpolation="BILINEAR",
            #     fill_mode="CONSTANT",
            #     fill_value=0,
            # )[0]
            return img_transformed

        def _apply_transformation_to_positions(self, positions, M):
            M = tf.cast(M, tf.float32)

            # Extract positions
            positions_abs = positions[:, :2] * IMG_SIZE

            # Make positions x, y instead of y, x
            positions_abs = positions_abs[:, ::-1]
            # print("positions switched")
            # print(positions_abs)

            # Make positions homogenous
            positions_abs_hom = tf.concat(
                [positions_abs, tf.ones((positions_abs.shape[0], 1), dtype=tf.float32)],
                axis=1,
            )
            # print("positions homogenous")
            # print(positions_abs_hom)

            # Apply transformation
            trans_positions_abs_hom = tf.transpose(
                tf.matmul(M, tf.transpose(positions_abs_hom))
            )
            # print("positions homogenous transformed")
            # print(trans_positions_abs_hom)

            trans_positions_abs = (
                trans_positions_abs_hom[:, :2] / trans_positions_abs_hom[:, 2:3]
            )  # convert to cartesian
            # print("positions transformed")
            # print(trans_positions_abs)

            # return to y, x
            trans_positions_abs = trans_positions_abs[:, ::-1]
            # print("transformed y, x")
            # print(trans_positions_abs)

            # Un-Normalize
            out_pos = trans_positions_abs / IMG_SIZE
            # print("un-normalized")
            # print(out_pos)

            # Add existence
            out_pos = tf.concat([out_pos, positions[:, -1:]], axis=1)
            # print("existence")
            # print(out_pos)

            return out_pos

        def __call__(
            self, img: tf.Tensor, positions: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor]:
            img = self.pixel_augmentation(img)

            img, positions = self.transformation_augmentation(img, positions)

            return img, positions

    def cache_dataset(ds, data_dir):
        # Get cache files
        cache_base = "data/cache/datasets"
        cache_id = data_dir.replace("/", "-")
        if cache_id.endswith("-"):
            cache_id = cache_id[:-1]
        cache_dir = os.path.join(cache_base, cache_id)

        # Remove existing cache files
        if clear_cache and os.path.exists(cache_dir):
            rmtree(cache_dir)
            rm_files = os.listdir(os.path.dirname(cache_dir))
            rm_files = [
                f
                for f in rm_files
                if f.startswith(cache_id + ".") or f.startswith(cache_id + "_0")
            ]
            for f in rm_files:
                os.remove(os.path.join(cache_base, f))

        # Create clean cache directory
        os.makedirs(cache_dir)

        # Cache to directory
        return ds.cache(cache_dir)

    def get_ds(
        data_dir: str,
        shuffle: bool = False,
        augment: bool = False,
        show: bool = False,
    ):
        # Load sample IDs
        sample_ids = [
            f for f in os.listdir(data_dir) if f.isnumeric() and int(f) < 1024
        ]
        if shuffle:
            np.random.shuffle(sample_ids)
        else:
            sample_ids = sorted(sample_ids, key=lambda x: int(x))

        # Get image paths
        image_paths = [os.path.join(data_dir, id) for id in sample_ids]

        # Convert to dataset
        ds = tf.data.Dataset.from_tensor_slices(image_paths)

        # Load Samples into Dataset
        ds = ds.map(
            lambda filepath: tf.numpy_function(
                func=Data.load_sample,
                inp=[filepath],
                Tout=[tf.float32, tf.float32],
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Set shapes
        ds = ds.map(
            lambda img, positions: (
                tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3]),
                tf.ensure_shape(positions, [3, 3]),
            )
        )

        # Cache data
        ds = Data.cache_dataset(ds, data_dir)

        # Shuffle
        if shuffle:
            ds = ds.shuffle(BATCH_SIZE * 3)

        # Augment
        if augment:
            ds = ds.map(Data.Augmentation())

        if show:
            Data.check_ds(ds)

        # Batch
        ds = ds.batch(BATCH_SIZE)

        # Prefetch
        ds = ds.prefetch(8)

        return ds

    def check_ds(ds: tf.data.Dataset) -> None:
        import cv2

        for img, positions in ds:
            img = (img.numpy() * 255).astype(np.uint8)
            positions = positions.numpy() * IMG_SIZE
            for y, x, existing in positions:
                if not existing:
                    continue
                cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), 2)

            show_imgs(img)

    def visualize_data_predictions(model, ds):
        preds = model.predict(ds)

        def matrix_to_string(matrix):
            # Scale positions and format as a string
            formatted_rows = [
                f"\t({round(row[1], 1):5.1f}, {round(row[0], 1):5.1f}) | Confidence: {round(100 * row[2]):3}%"
                for row in matrix
            ]

            # Join the rows into a single string with line breaks
            return "\n".join(formatted_rows)

        def plot_points(img, pts, color):
            for y, x, existing in pts:
                if existing < 0.5:
                    continue
                cv2.circle(img, (int(x), int(y)), 5, color, 2)
                cv2.circle(img, (int(x), int(y)), 2, color, -1)

        for i, (img, y_true) in enumerate(ds.unbatch()):

            img = np.uint8(img * 255)
            y_true = np.array(y_true, np.float32)
            y_pred = np.array(preds[i], np.float32)

            y_true[:, 0] *= img.shape[0]
            y_true[:, 1] *= img.shape[1]
            print()
            print("Target values:")
            print(matrix_to_string(y_true))
            print("Predicted values:")
            print(matrix_to_string(y_pred))

            plot_points(img, y_true, (0, 255, 0))
            plot_points(img, y_pred, (255, 0, 0))

            show_imgs(img)


# -----------------------------------------------
# Get Model

model = get_model()
# model = tf.keras.models.load_model("data/ai/darts_model.keras", compile=False)

# Compile model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=[CombinedLoss()],
)
model.summary(160)

# -----------------------------------------------
# Get Data

train_ds = Data.get_ds(
    "data/generation/out/",
    shuffle=True,
    augment=True,
    show=False,
)

val_ds = Data.get_ds(
    "data/generation/out_val/",
    shuffle=False,
    augment=False,
    show=False,
)

test_ds = Data.get_ds(
    "data/generation/out_test/",
    shuffle=False,
    augment=False,
    show=False,
)

# -----------------------------------------------
# Fit Model

if train:
    try:
        model.fit(
            train_ds,
            epochs=1000,
            validation_data=val_ds,
            callbacks=Utils.get_callbacks(),
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    # train_loop(
    #     model,
    #     epochs=1000,
    #     train_data=train_ds,
    #     val_data=val_ds,
    #     callbacks=Utils.get_callbacks(),
    # )

    if best_weights := Utils.get_best_model_checkpoint():
        model.load_weights(best_weights)

    model.save("data/ai/darts_model.keras")

if predict:
    Data.visualize_data_predictions(model, test_ds)
