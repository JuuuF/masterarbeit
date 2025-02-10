import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

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
from ma_darts.ai.models import yolo_v8_model, YOLOv8Loss, score2class

from tqdm import tqdm
from shutil import rmtree
from argparse import ArgumentParser
from datetime import datetime
from itertools import permutations
from tensorflow.keras import layers
from tensorflow.keras.applications import *

IMG_SIZE = 800
BATCH_SIZE = 8 if "GPU_SERVER" in os.environ.keys() else 4


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
        # callbacks.append(
        #     tf.keras.callbacks.TensorBoard(
        #         log_dir="data/ai/logs",
        #         histogram_freq=1,
        #         profile_batch=(0, 500),
        #     )
        # )
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
    def read_sample_img(filepath: tf.Tensor):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_png(img, channels=3)
        img = tf.reverse(img, axis=[-1])  # BGR
        img = tf.cast(img, tf.float32) / 255  # 0..1

        # Ensure shape consistency
        img.set_shape((Data.img_size, Data.img_size, 3))
        return img

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
    def extract_dart_classes(scores: list[str]):
        class_table = Data.get_class_table()
        class_ids = class_table.lookup(scores)  # (3,)
        scores_onehot = tf.one_hot(class_ids, depth=6, dtype=tf.int32)  # (3, 6)
        return tf.transpose(scores_onehot)  # (6, 3)

        def get_class(s) -> int:
            if s == "HIDDEN":
                return 0  # nothing
            if s == "OUT":
                return 5  # out
            if s in ["DB", "DBull"]:
                return 3  # red
            if s in ["B", "Bull"]:
                return 4  # green
            if s.isnumeric():
                par = Data.dart_order.index(int(s)) % 2
                return 1 if par == 0 else 2  # 1: black, 2: white
            s = s[1:]
            par = Data.dart_order.index(int(s)) % 2
            return 3 if par == 0 else 4  # 3: red, 4: green

    @staticmethod
    def parse_positions_and_scores(json_str: tf.string):
        sample_info = json.loads(json_str.numpy().decode("utf-8"))
        positions = np.array(
            sample_info["dart_positions_undistort"], dtype=np.float32
        ).T  # (2, 3)
        scores = np.array([s[1] for s in sample_info["scores"]], dtype=str)

        # Sort by descending y value
        order = np.argsort(positions[0])[::-1]
        positions = positions[:, order]
        scores = scores[order]
        return positions, scores

    @staticmethod
    def read_positions_and_classes(filepath: tf.Tensor):
        sample_json = tf.io.read_file(filepath)

        # Parse position and score values
        positions, scores = tf.py_function(
            Data.parse_positions_and_scores,
            [sample_json],
            Tout=[tf.float32, tf.string],
        )

        # Process classes
        classes = tf.py_function(
            Data.extract_dart_classes,
            [scores],
            Tout=tf.int32,
        )  # (6, 3)

        positions.set_shape((2, 3))
        classes.set_shape((6, 3))

        return positions, classes

    @staticmethod
    def load_sample(img_path: tf.Tensor, info_path: tf.Tensor):

        # Load data
        img = Data.read_sample_img(img_path)  # (800, 800, 3)

        positions, classes = Data.read_positions_and_classes(
            info_path
        )  # (2, 3), (6, 3)

        return (
            tf.cast(img, tf.float32),
            tf.cast(positions, tf.float32),
            tf.cast(classes, tf.int32),
        )

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

    @staticmethod
    def get_out_grid(out_size, n_pos, n_cls):
        # Start with a blank cell
        cell_col = tf.zeros((n_pos + n_cls), tf.float32)  # (n,)
        # Add a nothing-entry
        cell_col = tf.tensor_scatter_nd_update(
            cell_col, [[2]], [1]  # Update index 2 with value 1
        )
        # Extend to cell
        cell_col = tf.expand_dims(cell_col, -1)  # (n, 1)
        cell = tf.repeat(cell_col, repeats=3, axis=-1)  # (n, 3)
        # Extend to row
        grid_row = tf.repeat(tf.expand_dims(cell, 0), out_size, axis=0)  # (x, n, 3)
        # Extend to grid
        grid = tf.repeat(tf.expand_dims(grid_row, 0), out_size, axis=0)  # (y, x, n, 3)
        return grid

    @staticmethod
    def scaled_out(
        pos: tf.Tensor,  # (2, 3)
        cls: tf.Tensor,  # (6, 3)
        img_size: int,
        out_size: int,
    ):
        out_grid = Data.get_out_grid(
            out_size,
            tf.shape(pos)[0],
            tf.shape(cls)[0],
        )  # (y, x, 8, 3)

        cell_size = img_size // out_size
        pos_abs = pos * img_size  # (2, 3)

        # Get cell positions
        # tf.print("pos abs:")
        # tf.print(pos_abs)
        grid_pos = tf.cast(pos_abs // cell_size, tf.int32)  # (2, 3)
        local_pos = (pos_abs % cell_size) / cell_size  # (2, 3)
        # tf.print("grid_pos:")
        # tf.print(grid_pos)
        # tf.print("local_pos:")
        # tf.print(local_pos)

        # --------------------------------------

        # grid_y, grid_x = grid_pos[0], grid_pos[1]  # (3,), (3,)
        # pos_update = tf.concat([local_pos, cls], axis=0)  # (8, 3)

        # indices = grid_pos
        # cell_cols = tf.argmax(
        #     tf.cast(out_grid[grid_y, grid_x, 2] == 1, tf.int32), axis=-1
        # )

        # indices = tf.stack(
        #     [grid_y, grid_x, tf.fill([3], 2)], axis=-1
        # )  # (3, 3): 3x (y, x, 2)

        # cell_cols = tf.argmax()  # these depend on each other

        updates = []
        indices = []
        # 0. ------------------------------------
        # Get target cell
        grid_y, grid_x = grid_pos[0, 0], grid_pos[1, 0]
        current_cell = out_grid[grid_y, grid_x]

        # Get next available cell column
        cell_col = tf.argmax(tf.cast(current_cell[2] == 1, tf.int32))
        cell_col = tf.cast(cell_col, tf.int32)

        # Update position and class data
        pos_update = tf.concat([local_pos[:, 0], cls[:, 0]], axis=0)

        # Get update indices
        indices.append([grid_y, grid_x, 0, cell_col])
        indices.append([grid_y, grid_x, 1, cell_col])
        indices.append([grid_y, grid_x, 2, cell_col])
        indices.append([grid_y, grid_x, 3, cell_col])
        indices.append([grid_y, grid_x, 4, cell_col])
        indices.append([grid_y, grid_x, 5, cell_col])
        indices.append([grid_y, grid_x, 6, cell_col])
        indices.append([grid_y, grid_x, 7, cell_col])
        updates.append(pos_update[0])
        updates.append(pos_update[1])
        updates.append(pos_update[2])
        updates.append(pos_update[3])
        updates.append(pos_update[4])
        updates.append(pos_update[5])
        updates.append(pos_update[6])
        updates.append(pos_update[7])

        # 1. ------------------------------------
        # Get target cell
        grid_y, grid_x = grid_pos[0, 1], grid_pos[1, 1]
        current_cell = out_grid[grid_y, grid_x]

        # Get next available cell column
        cell_col = tf.argmax(tf.cast(current_cell[2] == 1, tf.int32))
        cell_col = tf.cast(cell_col, tf.int32)
        # tf.print("cell_col:")
        # tf.print(cell_col)

        # Update position and class data
        pos_update = tf.concat([local_pos[:, 1], cls[:, 1]], axis=0)

        # Get update indices
        indices.append([grid_y, grid_x, 0, cell_col])
        indices.append([grid_y, grid_x, 1, cell_col])
        indices.append([grid_y, grid_x, 2, cell_col])
        indices.append([grid_y, grid_x, 3, cell_col])
        indices.append([grid_y, grid_x, 4, cell_col])
        indices.append([grid_y, grid_x, 5, cell_col])
        indices.append([grid_y, grid_x, 6, cell_col])
        indices.append([grid_y, grid_x, 7, cell_col])
        updates.append(pos_update[0])
        updates.append(pos_update[1])
        updates.append(pos_update[2])
        updates.append(pos_update[3])
        updates.append(pos_update[4])
        updates.append(pos_update[5])
        updates.append(pos_update[6])
        updates.append(pos_update[7])
        # for cell_row in tf.range(tf.shape(pos_update)[0], dtype=tf.int32):
        #     indices.append([grid_y, grid_x, cell_row, cell_col])
        #     updates.append(pos_update[cell_row])

        # 2. ------------------------------------
        # Get target cell
        grid_y, grid_x = grid_pos[0, 2], grid_pos[1, 2]
        current_cell = out_grid[grid_y, grid_x]

        # Get next available cell column
        cell_col = tf.argmax(tf.cast(current_cell[2] == 1, tf.int32))
        cell_col = tf.cast(cell_col, tf.int32)
        # tf.print("cell_col:")
        # tf.print(cell_col)

        # Update position and class data
        pos_update = tf.concat([local_pos[:, 2], cls[:, 2]], axis=0)

        # Get update indices
        indices.append([grid_y, grid_x, 0, cell_col])
        indices.append([grid_y, grid_x, 1, cell_col])
        indices.append([grid_y, grid_x, 2, cell_col])
        indices.append([grid_y, grid_x, 3, cell_col])
        indices.append([grid_y, grid_x, 4, cell_col])
        indices.append([grid_y, grid_x, 5, cell_col])
        indices.append([grid_y, grid_x, 6, cell_col])
        indices.append([grid_y, grid_x, 7, cell_col])
        updates.append(pos_update[0])
        updates.append(pos_update[1])
        updates.append(pos_update[2])
        updates.append(pos_update[3])
        updates.append(pos_update[4])
        updates.append(pos_update[5])
        updates.append(pos_update[6])
        updates.append(pos_update[7])
        # for cell_row in tf.range(tf.shape(pos_update)[0], dtype=tf.int32):
        #     indices.append([grid_y, grid_x, cell_row, cell_col])
        #     updates.append(pos_update[cell_row])

        # ---------------------------------------
        # Apply updates

        indices = tf.convert_to_tensor(indices, dtype=tf.int32)
        updates = tf.convert_to_tensor(updates, dtype=tf.float32)

        # Apply updates
        out_grid = tf.tensor_scatter_nd_update(out_grid, indices, updates)

        return out_grid

        updates = []
        indices = []

        for i in tf.range(3):
            # Get target cell
            grid_y, grid_x = grid_pos[0, i], grid_pos[1, i]
            current_cell = out_grid[grid_y, grid_x]

            # Get next available cell column
            cell_col = tf.argmax(tf.cast(current_cell[2] == 1, tf.int32))
            cell_col = tf.cast(cell_col, tf.int32)
            # tf.print("cell_col:")
            # tf.print(cell_col)

            # Update position and class data
            pos_update = tf.concat([local_pos[:, i], cls[:, i]], axis=0)

            # Get update indices
            for cell_row in tf.range(tf.shape(pos_update)[0], dtype=tf.int32):
                indices.append([grid_y, grid_x, cell_row, cell_col])
                updates.append(pos_update[cell_row])

        indices = tf.convert_to_tensor(indices, dtype=tf.int32)
        updates = tf.convert_to_tensor(updates, dtype=tf.float32)

        # Apply updates
        out_grid = tf.tensor_scatter_nd_update(out_grid, indices, updates)

        # tf.print("--- indices + updates")
        # for i, u in zip(indices, updates):
        #     tf.print(i, ":", u, "->", out_grid[i[0], i[1], i[2], i[3]])
        # tf.print("-"*50)

        return out_grid

    @staticmethod
    def positions_to_yolo(
        img: tf.Tensor,  # (800, 800, 3)
        pos: tf.Tensor,  # (2, 3)
        cls: tf.Tensor,  # (6, 3)
    ):
        cls = tf.cast(cls, tf.float32)
        out_s = Data.scaled_out(pos, cls, 800, 25)
        out_m = Data.scaled_out(pos, cls, 800, 50)
        out_l = Data.scaled_out(pos, cls, 800, 100)
        return img, out_s, out_m, out_l

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
        num_parallel_calls = tf.data.AUTOTUNE
        # num_parallel_calls = 1

        # Collect files
        img_paths = tf.data.Dataset.list_files(
            os.path.join(data_dir, "*", "undistort.png"),
            shuffle=False,
        )
        info_paths = tf.data.Dataset.list_files(
            os.path.join(data_dir, "*", "info.json"),
            shuffle=False,
        )
        ds = tf.data.Dataset.zip(img_paths, info_paths)  # (img_path, info_path)

        # Shuffle files
        if shuffle:
            ds = ds.shuffle(16 * BATCH_SIZE)

        # Load Samples into Dataset
        ds = ds.map(
            Data.load_sample, num_parallel_calls=num_parallel_calls
        )  # (800, 800, 3), (2, 3), (6, 3)

        # ds = ds.apply(tf.data.experimental.ignore_errors())

        # Set shapes
        ds = ds.map(
            lambda img, pos, cls: (
                tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3]),
                tf.ensure_shape(pos, [2, 3]),
                tf.ensure_shape(cls, [6, 3]),
            ),
            num_parallel_calls=num_parallel_calls,
        )

        # Cache data
        ds = Data.cache_dataset(ds, data_dir)

        # Shuffle
        if shuffle:
            ds = ds.shuffle(BATCH_SIZE * 3)

        # Augment
        if augment:
            ds = ds.map(
                Data.Augmentation(),
                num_parallel_calls=num_parallel_calls,
            )

        # Convert to yolo outputs
        ds = ds.map(
            Data.positions_to_yolo, num_parallel_calls=num_parallel_calls
        )  # (800, 800, 3), (25, 25, 8, 3), (50, 50, 8, 3), (100, 100, 8, 3)

        # Set shapes
        ds = ds.map(
            lambda img, out_s, out_m, out_l: (
                tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3]),
                tf.ensure_shape(out_s, [IMG_SIZE // 32, IMG_SIZE // 32, 2 + 6, 3]),
                tf.ensure_shape(out_m, [IMG_SIZE // 16, IMG_SIZE // 16, 2 + 6, 3]),
                tf.ensure_shape(out_l, [IMG_SIZE // 8, IMG_SIZE // 8, 2 + 6, 3]),
            ),
            num_parallel_calls=num_parallel_calls,
        )

        # Map to input and output
        ds = ds.map(lambda img, out_s, out_m, out_l: (img, (out_s, out_m, out_l)))

        if show:
            Data.check_ds(ds)

        # Batch
        ds = ds.batch(BATCH_SIZE)

        # Prefetch
        ds = ds.prefetch(num_parallel_calls)

        return ds

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
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=YOLOv8Loss(
        img_size=800,
        square_size=50,
        # class_introduction_threshold=0.5,
        # position_introduction_threshold=0.5,
    ),
    # metrics=[
    #     tf.keras.metrics.MeanSquaredError(name="mse"),
    #     tf.keras.metrics.MeanAbsoluteError(name="mae"),
    # ],
)
# model.summary(160)
# print(model.input_shape)
# print(model.output_shape)
# exit()

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

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#

#


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
