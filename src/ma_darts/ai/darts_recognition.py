import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_disable_constant_folding=true"
# os.environ["XLA_FLAGS"] = "--xla_dump_to=/masterarbeit/dump/logs"

import re
import cv2
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from ma_darts.ai import callbacks as ma_callbacks
from ma_darts.ai.training import train_loop
from ma_darts.ai.utils import get_dart_scores, get_absolute_score_error
from ma_darts.cv.utils import show_imgs, matrices
from ma_darts.ai.models import yolo_v8_model, YOLOv8Loss, score2class, YOLOv8
from ma_darts.ai.data.utils import finalize_base_ds
from ma_darts.ai.data import dataloader_paper, dataloader_ma

from tqdm import tqdm
from shutil import rmtree
from argparse import ArgumentParser
from datetime import datetime
from tensorflow.keras import layers

IMG_SIZE = 800
BATCH_SIZE = 4 if "GPU_SERVER" in os.environ.keys() else 4


class Utils:
    model_checkpoint_filepath = (
        f"data/ai/checkpoints/darts/{datetime.now().strftime('%y_%m_%d-%H_%M')}/"
        + "epoch={epoch:05d}_val_loss={val_loss:06f}.weights.h5"
    )

    def get_callbacks():
        callbacks = []
        # History plotter
        hp = ma_callbacks.HistoryPlotter(
            filepath="dump/training_history.png",
            update_on="seconds",
            update_frequency=60,
            ease_curves=False,
            smooth_curves=True,
        )
        callbacks.append(hp)

        # Model Checkpoint
        mc = ma_callbacks.ModelCheckpoint(
            filepath=Utils.model_checkpoint_filepath,
            monitor="val_loss",
            save_best_only=True,
            max_saves=10,
            save_weights_only=True,
        )
        callbacks.append(mc)

        # Prediction callback
        # X, y = next(iter(val_ds.take(1)))
        # pc = ma_callbacks.PredictionCallback(
        #     X=X,
        #     y=y,
        #     output_file="dump/pred.png",
        #     update_on="seconds",
        #     update_frequency=60,
        # )
        # callbacks.append(pc)

        # TensorBoard
        tb = tf.keras.callbacks.TensorBoard(
            log_dir="data/ai/logs",
            histogram_freq=1,
            profile_batch=(0, 500),
        )
        # callbacks.append(tb)

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
    dart_order = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    img_size = IMG_SIZE

    @staticmethod
    def get_class_table():
        keys = tf.constant(
            ["HIDDEN", "OUT", "DB", "DBull", "B", "Bull"]
            + [f"{x}" for x in Data.dart_order]
            + [f"D{x}" for x in Data.dart_order]
            + [f"T{x}" for x in Data.dart_order]
        )
        values = tf.constant(
            [0, 5, 3, 3, 4, 4]
            + [1 if i % 2 == 0 else 2 for i in range(len(Data.dart_order))]  # single
            + [3 if i % 2 == 0 else 4 for i in range(len(Data.dart_order))]  # double
            + [3 if i % 2 == 0 else 4 for i in range(len(Data.dart_order))]  # triple
        )
        lut = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=0,
        )
        return lut

    @staticmethod
    class Augmentation:
        def __init__(
            self,
            brightness_adjust: float = 0.05,
            contrast_adjust: float = 0.01,
            max_noise: float = 0.1,
            max_rotation_angle: float = 8,
            max_translation_amount: float = 5,
            max_scaling: float = 0.1,
        ):
            self.brightness_adjust = brightness_adjust
            self.contrast_adjust = contrast_adjust
            self.max_noise = max_noise
            self.max_rotation_angle = np.deg2rad(max_rotation_angle)
            self.max_translation_amount = tf.cast(max_translation_amount, tf.float32)
            self.max_scaling = max_scaling
            self.bad_practice = int(np.random.uniform(0, 100000))

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
            self,
            img: tf.Tensor,
            positions: tf.Tensor,  # (2, 3)
        ) -> tuple[tf.Tensor, tf.Tensor]:

            M_img = np.eye(3)
            M_pts = np.eye(3)

            # Translate center to origin
            c = 400
            M_img = matrices.translation_matrix(-c, -c) @ M_img
            M_pts = matrices.translation_matrix(-c, -c) @ M_pts

            # Translate
            # bound_y0 = np.reduce_min(positions[0])
            # bound_y1 = tf.reduce_min(1 - positions[0])
            # bound_y = tf.minimum(bound_y0, bound_y1) * 800
            # bound_y = tf.minimum(self.max_translation_amount, bound_y)

            # bound_x0 = tf.reduce_min(positions[1])
            # bound_x1 = tf.reduce_min(1 - positions[1])
            # bound_x = tf.minimum(bound_x0, bound_x1) * 800
            # bound_x = tf.minimum(self.max_translation_amount, bound_x)

            # translation = np.random.normal(
            #     size=(2,),
            #     loc=0.0,
            #     scale=self.max_translation_amount / 3,
            # )
            # translation[0] = np.clip(translation[0], -bound_y, bound_y)
            # translation[1] = np.clip(translation[1], -bound_x, bound_x)
            # M_img = matrices.translation_matrix(*translation) @ M_img

            # Rotate
            rotation_amount = np.random.uniform(0, np.pi / 2)

            # rotation_amount = np.clip(
            #     rotation_amount, -self.max_rotation_angle, self.max_rotation_angle
            # )
            M_img = matrices.rotation_matrix(rotation_amount) @ M_img
            M_pts = matrices.rotation_matrix(-rotation_amount) @ M_pts

            # Re-transform to center
            M_img = matrices.translation_matrix(c, c) @ M_img
            M_pts = matrices.translation_matrix(c, c) @ M_pts

            # Apply transformation matrix
            M_img = tf.cast(M_img, tf.float32)
            M_pts = tf.cast(M_pts, tf.float32)
            img_trans = self._apply_transformation_to_image(img, M_img)

            positions_trans = self._apply_transformation_to_positions(positions, M_pts)

            return img_trans, positions_trans

        def _apply_transformation_to_image(self, img, M):
            M_flat = tf.reshape(M, [-1])[:-1]
            img_transformed = tf.raw_ops.ImageProjectiveTransformV3(
                images=tf.expand_dims(img, 0),  # as batch
                transforms=tf.expand_dims(M_flat, 0),  # as batch
                output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
                interpolation="BILINEAR",
                fill_mode="CONSTANT",
                fill_value=0,
            )[0]
            return img_transformed

        def _apply_transformation_to_positions(
            self,
            positions,  # (2, 3)
            M,  # (3, 3)
        ):
            p = positions
            positions *= 800

            # Make positions xy + homogenous
            positions = positions[::-1]  # (2, 3)
            positions = tf.transpose(positions)  # (3, 2)
            positions = tf.concat([positions, tf.ones((3, 1))], axis=1)  # (3, 3)

            # Apply transformation matrix
            positions_transformed = tf.linalg.matmul(positions, M, transpose_b=True)
            positions_transformed = (
                positions_transformed[:, :2] / positions_transformed[:, -1:]
            )

            # switch to the superior yx format
            positions_transformed = tf.transpose(positions_transformed)
            positions_transformed = positions_transformed[::-1]  # (2, 3)

            positions_transformed /= 800

            return positions_transformed

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
            self,
            img: tf.Tensor,
            positions: tf.Tensor,
            classes: tf.Tensor,
        ) -> tuple[tf.Tensor, tf.Tensor]:

            img = self.pixel_augmentation(img)
            # img, positions = tf.numpy_function(
            #     func=self.transformation_augmentation,
            #     inp=[img, positions],
            #     Tout=[tf.float32, tf.float32],
            # )

            # TODO: positions MIGHT fall out of frame
            # -> filter them out by setting class to nothing

            # min_pos = tf.reduce_min(positions)
            # max_pos = tf.reduce_max(positions)
            # if min_pos < 0:
            #     idx = tf.where

            return img, positions, classes

    def check_ds(ds: tf.data.Dataset) -> None:
        import cv2

        classes = ["nothing", "black", "white", "red", "green", "out"]

        for img, (out_s, out_m, out_l) in ds:

            img = (img.numpy() * 255).astype(np.uint8)

            grid = out_s.numpy()
            for y, grid_row in enumerate(grid):
                if (grid_row[:, 2] == 1).all():
                    continue
                for x, grid_cell in enumerate(grid_row):  # (8, 3)
                    if (grid_cell[2] == 1).all():
                        continue

                    for i in range(3):
                        col = grid_cell[:, i]
                        if col[2] == 1:
                            continue
                        pos = col[:2]
                        cell_class = np.argmax(col[2:])
                        pos_y = y * 32 + round(pos[0] * 32)
                        pos_x = x * 32 + round(pos[1] * 32)
                        img[pos_y, pos_x] = (255, 255, 255)
                        for i, c in enumerate([0, 255]):
                            cv2.putText(
                                img,
                                classes[cell_class],
                                org=(pos_x + 10, pos_y),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=1,
                                color=(c, c, c),
                                thickness=2 - i,
                            )
                        # print(pos_y, pos_x)
                        cv2.circle(img, (pos_x, pos_y), 3, (255, 0, 0), 1)
                        cv2.circle(img, (pos_x, pos_y), 6, (255, 255, 255), 1)

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


# data_dir = "data/generation/out"
# sample_ids = [f for f in os.listdir(data_dir) if f.isnumeric()]
# sample_ids = sorted(sample_ids, key=int)

# for i in range(1000):
#     sample_id = sample_ids[i]
#     sample_info = pickle.load(open(os.path.join(data_dir, sample_id, "info.pkl"), "rb"))
#     score = sample_info.scores
#     classes = Data.extract_dart_classes(sample_info)
#     print(score, classes)
#     input()
# exit()


# -----------------------------------------------
# Command Line Arguments

args = Utils.get_args()

# -----------------------------------------------
# Get Model

if args.model_path is None:
    model = yolo_v8_model(
        input_size=800,
        classes=["nothing", "black", "white", "red", "green", "out"],
        variant=args.model_type,
    )
else:
    model = tf.keras.models.load_model(args.model_path, compile=False)

# Compile model
metrics = [
    tf.keras.metrics.MeanSquaredError(name="mse"),
    tf.keras.metrics.MeanAbsoluteError(name="mae"),
]
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=YOLOv8Loss(
        img_size=800,
        square_size=50,
        class_introduction_threshold=0.1,
        position_introduction_threshold=0.1,
    ),
    metrics=[metrics for _ in range(3)],
)
# model.summary(160)
# print(model.input_shape)
# print(model.output_shape)
# exit()

# -----------------------------------------------
# Get Data

train_ds = dataloader_ma(
    "data/generation/out/",
    batch_size=BATCH_SIZE,
    shuffle=True,
    augment=True,
    cache=True,
    clear_cache=args.clear_cache,
)

val_ds = dataloader_paper(
    base_dir="data/paper/",
    dataset="d2",
    split="val",
    img_size=IMG_SIZE,
    shuffle=False,
    augment=False,
    batch_size=BATCH_SIZE,
    cache=False,
    clear_cache=args.clear_cache,
)

# test_ds = Data.get_ds(
#     "data/generation/out_test/",
#     shuffle=False,
#     augment=False,
# )

# -----------------------------------------------
# Fit Model

if args.train:

    try:
        model.fit(
            train_ds,
            epochs=args.epochs,
            validation_data=val_ds,
            callbacks=Utils.get_callbacks(),
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    if best_weights := Utils.get_best_model_checkpoint():
        model.load_weights(best_weights)

    model_path = "data/ai/darts_model.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}.")

if args.predict:
    Data.visualize_data_predictions(model, test_ds)
