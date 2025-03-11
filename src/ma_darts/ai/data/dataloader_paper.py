import os
import cv2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from ma_darts.inference.paper_code import d1_val, d1_test, d2_val, d2_test, BOARD_DICT
from ma_darts.inference.paper_code import board_radii
from ma_darts.cv.utils import apply_matrix, show_imgs
from ma_darts.ai.data import finalize_base_ds


def load_labels(base_dir: str) -> pd.DataFrame:

    # Check for pickled labels
    filepath_pkl = os.path.join(base_dir, "labels.pkl")
    if os.path.exists(filepath_pkl):
        with open(filepath_pkl, "rb") as f:
            df = pickle.load(f)
            return df

    # Check for CSV file with labels
    filepath_csv = os.path.join(base_dir, "labels.csv")
    if os.path.exists(filepath_csv):
        df = pd.read_csv(filepath_csv)
        return df

    # Did not find a labels file
    raise ValueError(
        f"No data information found in {base_dir=}. "
        "Ensure there is a file calles 'labels.pkl' or 'labels.csv' in that directory."
    )


def get_data_split(df: pd.DataFrame, dataset: str, split: str) -> pd.DataFrame:
    # Get dataset splits
    if dataset == "d1":
        val_dirs, test_dirs = d1_val, d1_test
    elif dataset == "d2":
        val_dirs, test_dirs = d2_val, d2_test
    else:
        raise ValueError(
            f"Unknown dataset identifier: {dataset}. "
            "The value of 'dataset' has to be either 'd1' or 'd2'."
        )

    # Split data into datasets
    if split == "all":
        return df

    if split == "train":
        return df[np.logical_not(np.isin(df.img_folder, val_dirs + test_dirs))]

    if split == "val":
        return df[np.isin(df.img_folder, val_dirs)]

    if split == "test":
        return df[np.isin(df.img_folder, test_dirs)]

    raise ValueError(
        f"Unknown data split: {split}. "
        "Please make sure the split is 'all', 'train', 'val' or 'test'."
    )


def get_undistortion_homography(
    pos: pd.Series,  # (n, 4, 2): x, y; 0..1
) -> tuple[list[np.ndarray], list[float], list[float]]:

    # Make absolute
    pos *= 800

    # Get centers
    cs = pos.apply(lambda ps: np.mean(ps, axis=0))  # (n, 2): x, y
    cs = np.array(list(cs))

    sin_t = np.sin(np.deg2rad(9))
    cos_t = np.cos(np.deg2rad(9))

    # Get homographies
    r_ds = []
    Ms = []

    r_dst = 300
    c_dst = 400
    dst_pts = np.array(
        [
            [-sin_t * r_dst + c_dst, -cos_t * r_dst + c_dst],  # top
            [sin_t * r_dst + c_dst, cos_t * r_dst + c_dst],  # bottom
            [-cos_t * r_dst + c_dst, sin_t * r_dst + c_dst],  # left
            [cos_t * r_dst + c_dst, -sin_t * r_dst + c_dst],  # right
        ],
        dtype=np.float32,
    )

    for c, p in zip(cs, pos):
        r = np.linalg.norm(p - c, axis=-1).mean()

        src_pts = np.array(p, np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        r_ds.append(r / 800)
        Ms.append(M)

    return (
        Ms,  # (n, 3, 3)
        r_ds,  # (n,)
        list(cs / 800),  # (n, 2); x, y
    )


def undistort_dart_positions(pos: pd.Series, Ms: pd.Series) -> pd.Series:
    out_pos = []
    for p, M in zip(pos, Ms):
        # No positions
        if len(p) == 0:
            out_pos.append(np.zeros((3, 2)))
            continue

        # Make absolute
        p *= 800

        # Make homogenous
        xyz = np.float32(
            np.concatenate([p, np.ones((p.shape[0], 1))], axis=-1)
        )  # (n, 3)

        # Undistort
        xyz_undist = np.matmul(M, xyz.T).T
        xy_undistort = xyz_undist[:, :2] / xyz_undist[:, 2:]

        # Normalize
        xy_undistort /= 800

        # Fill with zeros
        xy_undistort = np.concatenate(
            [xy_undistort, np.zeros((3 - xy_undistort.shape[0], 2))]
        )

        # Flip it to get the correct translation
        out_pos.append(xy_undistort)

    return out_pos  # (n, 3, 2): x, y


def score_from_polar(
    angle: float,
    dist: float,
    r_t: float,
    r_ob: float,
    r_ib: float,
    w_dt: float,
    r_d: float,
):
    # Out
    if dist > r_d:
        return (0, "OUT")

    # Bulls Eye
    if dist <= r_ib:
        return (50, "DB")

    # Bull
    if dist <= r_ob:
        return (25, "B")

    score = int(BOARD_DICT[int((angle - 9) / 18)])

    # Double
    if dist <= r_d and dist > r_d - w_dt:
        return (2 * score, f"D{score}")

    # Triple
    if dist <= r_t and dist > r_t - w_dt:
        return (3 * score, f"T{score}")

    # Regular score
    return (score, str(score))


def get_scores(
    pos: pd.Series,  # (n, 3, 2): x, y
    r_ds: pd.Series,  # (n,) outer double radius
    centers: pd.Series,  # (n, 2): x, y
) -> list[tuple[int, str]]:
    all_scores = []
    for ps, r_d, c in zip(pos, r_ds, centers):
        scores = [(0, "HIDDEN") for _ in range(3)]
        (
            r_t,  # triple outer
            r_ob,  # outer bull
            r_ib,  # inner bull
            w_dt,  # multiplier width
        ) = board_radii(r_d)

        # Iterate over darts
        for i, p in enumerate(ps):
            if p.sum() == 0:
                continue
            p -= c
            angle = np.arctan2(-p[1], p[0]) / np.pi * 180  # -180..180
            angle %= 360  # 0..360

            dist = np.linalg.norm(p)

            score = score_from_polar(angle, dist, r_t, r_ob, r_ib, w_dt, r_d)
            scores[i] = score

        all_scores.append(scores)

    return all_scores  # (n, [int, str])


def scores_to_classes(all_scores: list[tuple[int, str]]) -> list[np.array]:
    # 0 nothing
    # 1 black
    # 2 white
    # 3 red
    # 4 green
    # 5 out
    classes = []
    for dart_scores in all_scores:
        out_classes = np.zeros((6, 3), np.int32)
        for i, (score_val, score_str) in enumerate(dart_scores):
            if score_str == "HIDDEN":
                out_classes[0, i] = 1
                continue
            if score_str in ["DB", "DBull"]:
                out_classes[3, i] = 1
                continue
            if score_str in ["B", "Bull"]:
                out_classes[4, i] = 1
                continue
            if score_str == "OUT":
                out_classes[5, i] = 1
                continue

            if score_str[0] in "DT":
                is_multiplier = True
                score = int(score_str[1:])
            else:
                is_multiplier = False
                score = int(score_str)

            is_black = score in [20, 18, 13, 10, 2, 3, 7, 8, 14, 12]
            if is_black:
                if is_multiplier:
                    out_classes[3, i] = 1
                else:
                    out_classes[1, i] = 1
            else:
                if is_multiplier:
                    out_classes[4, i] = 1
                else:
                    out_classes[2, i] = 1

        # Fill the rest with nothings
        out_classes[0, i + 1 :] = 1
        classes.append(out_classes)

    return classes  # (n, 6, 3)


def prepare_data_info(base_dir: str, df: pd.DataFrame) -> pd.DataFrame:
    # Trim down base dir
    if base_dir.endswith("/"):
        base_dir = base_dir[:-1]

    # Add filepath to data
    df["filepath"] = base_dir + "/imgs/" + df["img_folder"] + "/" + df["img_name"]

    # Split orientation and darts points
    df.xy.apply(np.array)
    df["orientation_points"] = df.xy.apply(lambda x: x[:4])
    df["orientation_points"] = df.orientation_points.apply(
        np.array
    )  # n * (4, 2) -> (x, y)

    df["darts_positions"] = df.xy.apply(lambda x: x[4:])  # normalized 0..1
    df["darts_positions"] = df.darts_positions.apply(np.array)  # n * (<= 3, 2): x, y

    # Get undistortion homography
    df["undistortion_homography"], df["r_do"], df["center"] = (
        get_undistortion_homography(df.orientation_points)
    )  # (n, 3, 3), (n,): outer double radius, (n, 2): x, y

    # Undistort points
    df["darts_positions_undistorted"] = undistort_dart_positions(
        df.darts_positions, df.undistortion_homography
    )  # (n, 3, 2): x, y; center-relative

    # Get scores
    df["scores"] = get_scores(
        df.darts_positions_undistorted, df.r_do, df.center
    )  # (n, [int, str])

    df["scores_classes"] = scores_to_classes(df.scores)  # (n, 6, 3)

    return df[
        [
            "filepath",  # (n,)
            "darts_positions_undistorted",  # (n, 3, 2): x, y; center-relative
            "scores_classes",  # (n, 6, 3)
            "undistortion_homography",  # (n, 3, 3)
        ]
    ]


def load_imgs(df: pd.DataFrame):
    imgs = []
    filepaths = df.loc[:, "filepath"]
    Ms = df.loc[:, "undistortion_homography"]
    for filepath, M in tqdm(zip(filepaths, Ms)):
        img = cv2.imread(filepath)
        img_undistort = apply_matrix(img, M)
        imgs.append(img_undistort)
    return imgs


def read_img(filepath: tf.Tensor):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.reverse(img, axis=[-1])  # BGR
    img = tf.cast(img, tf.float32) / 255  # 0..1

    img.set_shape((800, 800, 3))
    return img


def undistort_img(
    img: tf.Tensor,  # (800, 800, 3)
    M: tf.Tensor,  # (3, 3)
) -> tf.Tensor:
    img = np.uint8(img * 255)
    M = np.float32(M)
    img_undist = cv2.warpPerspective(img, M, (800, 800))
    img_undist = np.float32(img_undist) / 255
    return img_undist

    M_flat = tf.reshape(M, (-1,))[:8]
    M_flat = tf.cast(M_flat, tf.float32)
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(M_flat, 0),
        output_shape=tf.constant((800, 800), tf.int32),
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=0,
    )[0]

    return img


def paper_base_ds(
    base_dir: str, dataset: str = "d1", split: str = "all"
) -> tf.data.Dataset:

    # Get data info
    df = load_labels(base_dir=base_dir)
    df = get_data_split(df, dataset, split)

    # Prepare data
    df = prepare_data_info(base_dir, df)
    # i = list(df["filepath"]).index("data/paper/imgs/d2_03_03_2020/DSC_0005.JPG")  # 15544
    # df = df.iloc[15544:, :]

    filepaths = df["filepath"]  # (n,)
    classes = df["scores_classes"]  # (n, 6, 3)
    positions = df["darts_positions_undistorted"]  # (n, 3, 2): x, y; center-relative
    Ms = df["undistortion_homography"]  # (n, 3, 3)

    filepaths = list(filepaths)
    classes = np.array(list(classes.values))  # (n, 6, 3)
    existences = 1 - classes[:, :1]  # (n, 1, 3)
    classes = classes[:, 1:]  # (n, 5, 3)
    positions = np.array(list(positions.values))
    Ms = np.array(list(Ms.values))

    # Transfer positions to standard-form
    positions = positions[..., ::-1]  # xy -> yx
    positions = np.transpose(positions, (0, 2, 1))  # (n, 2, 3)
    positions = np.where(
        (positions[:, :1] != 0) & (positions[:, 1:] != 0), positions + 0.5, positions
    )
    positions = np.float32(positions)
    # positions += 0.5  # top-left normalized

    # Convert all to dataset slices
    ds_filepaths = tf.data.Dataset.from_tensor_slices(filepaths)
    ds_xst = tf.data.Dataset.from_tensor_slices(existences)
    ds_cls = tf.data.Dataset.from_tensor_slices(classes)
    ds_pos = tf.data.Dataset.from_tensor_slices(positions)
    ds_Ms = tf.data.Dataset.from_tensor_slices(Ms)

    # Read images
    ds_imgs = ds_filepaths.map(
        read_img,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Apply undistortion
    ds_imgs = tf.data.Dataset.zip(ds_imgs, ds_Ms).map(
        lambda img, M: tf.py_function(
            func=undistort_img,
            inp=[img, M],
            Tout=tf.float32,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = tf.data.Dataset.zip(ds_imgs, ds_xst, ds_pos, ds_cls)
    return ds


def dataloader_paper(
    base_dir: str,
    dataset: str = "d1",
    split: str = "all",
    img_size: int = 800,
    shuffle: bool = False,
    augment: bool = False,
    batch_size: int = 4,
    prefetch: int = tf.data.AUTOTUNE,
    cache: bool = True,
    clear_cache: bool = False,
) -> tf.data.Dataset:
    base_ds = paper_base_ds(base_dir, dataset, split)
    ds = finalize_base_ds(
        base_ds,
        data_dir=base_dir,
        img_size=img_size,
        shuffle=shuffle,
        augment=augment,
        batch_size=batch_size,
        prefetch=prefetch,
        cache=cache,
        clear_cache=clear_cache,
    )
    return ds


if __name__ == "__main__":
    ds = dataloader_paper(base_dir="data/paper", dataset="d1", split="all")
