import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

import re
import cv2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from ma_darts.ai import callbacks as ma_callbacks
from ma_darts.ai.training import train_loop
from ma_darts.ai.utils import get_dart_scores, get_absolute_score_error
from ma_darts.cv.utils import show_imgs, matrices

from tqdm import tqdm
from shutil import rmtree
from argparse import ArgumentParser
from datetime import datetime
from itertools import permutations
from tensorflow.keras import layers
from tensorflow.keras.applications import *

IMG_SIZE = 800
BATCH_SIZE = 128 if "GPU_SERVER" in os.environ.keys() else 4
from ma_darts.ai.models import yolo_v8_model, YOLOv8Loss


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
                update_frequency=10,
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

        # Check if directory exists
        if not os.path.exists(checkpoint_dir):
            return

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

    def get_args():
        parser = ArgumentParser()

        parser.add_argument(
            "--train",
            action="store_true",
            help="Train model.",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=1000,
            help="Training epochs.",
        )
        parser.add_argument(
            "--limit_data",
            type=int,
            default=-1,
            help="Dataset size limitation.",
        )
        parser.add_argument(
            "--clear_cache",
            action="store_true",
            help="Clear dataset cache files.",
        )
        parser.add_argument(
            "--predict",
            action="store_true",
            help="Predict test dataset using model.",
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default=None,
            help="Model path (optional).",
        )
        parser.add_argument(
            "--model_type",
            type=str,
            default="n",
            help="Model architecture size. Avaulable: n, s, m, l, x",
        )

        args = parser.parse_args()
        return args


class Data:

    def read_sample_img(sample_dir: str):
        filepath = tf.strings.join([sample_dir, "/undistort.png"])
        img = tf.io.read_file(filepath)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
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

        img = Data.read_sample_img(sample_dir)  # (800, 800, 3)
        sample_info = Data.read_sample_info(sample_dir)
        dart_positions = Data.extract_darts_positions(
            sample_info
        )  # (3, 3): 3x (x, y, 0/1)

        # Convert to cell positions
        outputs = []
        for s in [25, 50, 100]:
            grid_size = 800 // s
            cell_idxs, cell_poss = np.divmod(dart_positions[:, :2] * 800, grid_size)
            cell_idxs = np.int32(cell_idxs)  # (2, 3)
            cell_poss /= grid_size  # (2, 3)

            scaled_output = np.zeros((s, s, 3, 3), np.float32)
            existing = dart_positions[:, -1] == 1
            for i in range(3):
                if not existing[i]:
                    continue
                grid_y = cell_idxs[i, 0]
                grid_x = cell_idxs[i, 1]
                cell = scaled_output[grid_y, grid_x]  # (3, 3)
                j = np.where(cell[:, -1] == 0)[0][0]
                cell[:2, j] = cell_poss[i]
                cell[-1, j] = 1

            outputs.append(scaled_output)

        return img, *[tf.cast(o, tf.float32) for o in outputs]

        """ assure correct position placement: """
        img = (img.numpy() * 255).astype(np.uint8)
        positions_s = Data.convert_to_img_positions(outputs[0])
        positions_m = Data.convert_to_img_positions(outputs[1])
        positions_l = Data.convert_to_img_positions(outputs[2])

        import cv2
        from ma_darts.cv.utils import show_imgs

        for y, x, _ in positions_s:
            cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), 2)
        show_imgs(img)
        return img, *[tf.cast(o, tf.float32) for o in outputs]

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
        if args.clear_cache and os.path.exists(cache_dir):
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
        os.makedirs(cache_dir, exist_ok=True)

        # Cache to directory
        return ds.cache(cache_dir)

    def get_ds(
        data_dir: str,
        shuffle: bool = False,
        augment: bool = False,
        show: bool = False,
    ):
        # Load sample IDs
        sample_ids = [f for f in os.listdir(data_dir) if f.isnumeric()]

        # Limit data
        if args.limit_data > 0:
            sample_ids = [f for f in sample_ids if int(f) < args.limit_data]

        # Shuffle or sort
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
        ds = ds.prefetch(tf.data.AUTOTUNE)

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

        preds = []

        try:
            print("Predicting...")
            for X, y_true in tqdm(ds):
                y_pred = model.predict(X, verbose=0)
                preds.extend([(x, y, y_) for x, y, y_ in zip(X, y_true, y_pred)])
        except KeyboardInterrupt:
            print("\nPrediction aborted.")

        def matrix_to_string(matrix):
            # Scale positions and format as a string
            formatted_rows = [
                f"\t({round(row[1], 1):5.1f}, {round(row[0], 1):5.1f}) | Confidence: {round(100 * row[2]):3}%"
                for row in matrix
            ]

            # Join the rows into a single string with line breaks
            return "\n".join(formatted_rows)

        def plot_points(img, pts, color, add_unsure: bool = False):
            for y, x, existing in pts:
                if existing < 0.5:
                    if add_unsure:
                        color = (0, 0, round(255 * existing * 2))
                        cv2.circle(img, (int(x), int(y)), 5, color, 2)
                        cv2.circle(img, (int(x), int(y)), 2, (255, 255, 255), -1)
                    continue
                color = tuple(round(c * (existing / 2 + 0.5)) for c in color)
                cv2.circle(img, (int(x), int(y)), 5, color, 2)
                cv2.circle(img, (int(x), int(y)), 2, (255, 255, 255), -1)

        for img, y_true, y_pred in preds:

            img = np.uint8(img * 255)
            y_true = np.array(y_true, np.float32)
            y_pred = np.array(y_pred, np.float32)

            y_true[:, 0] *= img.shape[0]
            y_true[:, 1] *= img.shape[1]
            y_pred[:, 0] *= img.shape[0]
            y_pred[:, 1] *= img.shape[1]

            scoring_true = y_true[y_true[:, -1] > 0.5, :2]
            scoring_pred = y_pred[y_pred[:, -1] > 0.5, :2]
            scores_true = get_dart_scores(
                list(scoring_true), img_size=IMG_SIZE, margin=100
            )
            scores_pred = get_dart_scores(
                list(scoring_pred), img_size=IMG_SIZE, margin=100
            )
            ase = get_absolute_score_error(scores_true, scores_pred)
            print()
            print("Target values:", sorted(scores_true))
            print(matrix_to_string(y_true))
            print("Predicted values:", sorted(scores_pred))
            print(matrix_to_string(y_pred))
            print("Absolute Score Error:", ase)

            plot_points(img, y_true, (0, 255, 0))
            plot_points(img, y_pred, (255, 0, 0), add_unsure=True)

            show_imgs(img)



# -----------------------------------------------
# Command Line Arguments

args = Utils.get_args()

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

if args.train:
    # Warmup
    print("Warmup...")
    for _ in range(5):
        model.predict(
            np.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), np.float32), verbose=0
        )

    try:
        model.fit(
            train_ds,
            epochs=args.epochs,
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

    model_path = "data/ai/darts_model.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}.")

if args.predict:
    Data.visualize_data_predictions(model, test_ds)


class Unused:
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
            true_positions_masked = true_positions * true_existence * 8  # (bs, 3, 2)
            pred_positions_masked = pred_positions * true_existence * 8  # (bs, 3, 2)

            # Calculate distances
            dists_y = (
                true_positions_masked[..., 0] - pred_positions_masked[..., 0]
            )  # (bs, 3)
            dists_x = (
                true_positions_masked[..., 1] - pred_positions_masked[..., 1]
            )  # (bs, 3)

            # MSE over distances
            dists = dists_y**2 + dists_x**2  # (bs, 3)
            dists_loss = tf.reduce_mean(dists)

            # Calculate masked positional loss
            # positions_loss = self.positions_loss(
            #     true_positions_masked, pred_positions_masked
            # )  # (1,)

            return existence_loss + (1 + existence_loss) * (dists_loss)

    class Backbones:
        def bipolar_rescale() -> dict:
            """(-1, 1) rescaling"""
            return dict(scale=2, offset=-1)

        def unnormalized_rescale() -> dict:
            """(0, 255) rescaling"""
            return dict(scale=255, offset=0)

        def mobilenet_v3_large(
            input_tensor,
            input_shape: list[int | None],
            trainable: bool = False,
        ) -> tf.Tensor:
            # Rescale to (-1, 1)
            x = layers.Rescaling(**Backbones.bipolar_rescale(), name="input_rescaling")(
                input_tensor
            )

            # Get backbone
            backbone = MobileNetV3Large(
                include_top=False,
                weights="imagenet",
                input_shape=input_shape,
                include_preprocessing=False,
            )
            backbone.trainable = trainable

            # Apply backbone
            x = backbone(x)
            return x

        def mobilenet_v2(
            input_tensor,
            input_shape: list[int | None],
            trainable: bool = False,
        ) -> tf.Tensor:
            # Rescale to (-1, 1)
            x = layers.Rescaling(**Backbones.bipolar_rescale(), name="input_rescaling")(
                input_tensor
            )

            # Get backbone
            backbone = MobileNetV2(
                include_top=False,
                weights="imagenet",
                input_shape=input_shape,
            )
            backbone.trainable = trainable

            # Apply backbone
            x = backbone(x)
            return x

        def efficientnet_v2_b1(
            input_tensor,
            input_shape: list[int | None],
            trainable: bool = False,
        ) -> tf.Tensor:
            # Rescale to (-1, 1)
            x = layers.Rescaling(**Backbones.bipolar_rescale(), name="input_rescaling")(
                input_tensor
            )

            # Get backbone
            backbone = EfficientNetV2B1(
                include_top=False,
                weights="imagenet",
                input_shape=input_shape,
                include_preprocessing=False,
            )
            backbone.trainable = trainable

            # Apply backbone
            x = backbone(x)
            return x

        def convnext_tiny(
            input_tensor,
            input_shape: list[int | None],
            trainable: bool = False,
        ) -> tf.Tensor:
            # Rescale to (0, 255)
            x = layers.Rescaling(
                **Backbones.unnormalized_rescale(), name="input_rescaling"
            )(input_tensor)

            # Get backbone
            backbone = ConvNeXtTiny(
                include_top=False,
                weights="imagenet",
                input_shape=input_shape,
                include_preprocessing=False,
            )
            backbone.trainable = trainable

            # Apply backbone
            x = backbone(x)
            return x

        def yolo_v8(
            input_tensor,
            input_shape: list[int | None],
            trainable: bool = False,
        ) -> tf.Tensor:
            from keras_cv.models import YOLOV8Backbone

            # Get backbone
            backbone = YOLOV8Backbone(
                stackwise_channels=[64, 128, 256, 512],
                stackwise_depth=[1, 2, 8, 8],
                include_rescaling=False,
                input_shape=input_shape,
            )
            backbone.trainable = trainable

            # Apply backbone
            x = backbone(input_tensor)
            return x

    def get_model():

        input_shape = (IMG_SIZE, IMG_SIZE, 3)

        # Input
        inputs = layers.Input(shape=input_shape)

        # Backbone
        x = Backbones.mobilenet_v3_large(inputs, input_shape, trainable=True)

        # Convolve
        x = layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Flatten
        x = layers.GlobalAveragePooling2D()(x)

        # Head
        for i, n in enumerate([256, 128, 64]):
            x = layers.Dense(n, activation="mish", name=f"head_{i}_dense")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3, name=f"head_{i}_dropout")(x)
        x = layers.BatchNormalization()(x)

        # Output
        outputs = x
        outputs = layers.Dense(9, activation="linear", name="output_downsample")(
            outputs
        )
        outputs = layers.Reshape((3, 3))(outputs)

        # Output activations
        positions = layers.Activation("hard_sigmoid", name="output_positions")(
            outputs[..., :2]
        )
        existences = layers.Activation("sigmoid", name="output_existences")(
            outputs[..., -1:]
        )
        outputs = layers.Concatenate(axis=-1, name="output_concatenation")(
            [positions, existences]
        )

        # Build model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model
