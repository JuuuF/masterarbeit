import os
import json
import numpy as np
import tensorflow as tf

from ma_darts.ai.data import finalize_base_ds
from ma_darts import dart_order


def get_class_table():
    keys = tf.constant(
        ["HIDDEN", "OUT", "DB", "DBull", "B", "Bull"]
        + [f"{x}" for x in dart_order]
        + [f"D{x}" for x in dart_order]
        + [f"T{x}" for x in dart_order]
    )
    values = tf.constant(
        [0, 5, 3, 3, 4, 4]
        + [1 if i % 2 == 0 else 2 for i in range(len(dart_order))]  # single
        + [3 if i % 2 == 0 else 4 for i in range(len(dart_order))]  # double
        + [3 if i % 2 == 0 else 4 for i in range(len(dart_order))]  # triple
    )
    class_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=0,
    )
    return class_table


def _parse_json(json_str):
    # If json_str is a NumPy array (which happens when tf.py_function converts it),
    # extract the underlying Python object.
    if isinstance(json_str, np.ndarray):
        json_str = json_str.item()
    # If it's still a Tensor (e.g. in eager mode), convert it to a numpy array.
    if isinstance(json_str, tf.Tensor):
        json_str = json_str.numpy()
    # If the resulting object is bytes, decode it.
    if isinstance(json_str, bytes):
        json_str = json_str.decode("utf-8")
    # Decode the byte string and load JSON data
    sample_info = json.loads(json_str)

    # Process positions
    positions = tf.convert_to_tensor(
        sample_info.get("dart_positions_undistort", [(0, 0)]), dtype=tf.float32
    )

    positions = positions[:3]
    add_cols = tf.clip_by_value(tf.constant(3, tf.int32) - tf.shape(positions)[0], 0, 3)
    positions = tf.concat(
        [positions, tf.zeros((add_cols, 2), dtype=tf.float32)], axis=0
    )
    positions = tf.ensure_shape(positions, (3, 2))
    positions = tf.transpose(positions)  # (2, 3)

    # Process scores
    # (Assuming each score entry is a pair like [score, label])
    scores = tf.convert_to_tensor(
        [
            str(s[1])
            for s in sample_info.get("scores", [[0, "HIDDEN"] for _ in range(3)])
        ],
        dtype=tf.string,
    )
    scores = tf.ensure_shape(scores, (3,))

    # Sort by descending y value
    order = tf.argsort(positions[0], direction="DESCENDING")
    positions = tf.gather(positions, order, axis=1)
    scores = tf.gather(scores, order)

    return positions, scores


def parse_positions_and_scores(json_str: tf.string):
    positions, scores = tf.py_function(
        func=_parse_json, inp=[json_str], Tout=[tf.float32, tf.string]
    )
    return positions, scores

    sample_info = json.loads(json_str.numpy().decode("utf-8"))
    positions = tf.convert_to_tensor(
        sample_info.get("dart_positions_undistort", [(0, 0)]), dtype=tf.float32
    )  # (3, 2)
    positions = tf.ensure_shape(positions, (3, 2))

    # Make sure there are exactly 3 positions
    positions = positions[:3]
    add_cols = tf.clip_by_value(tf.constant(3, tf.int32) - tf.shape(positions)[0], 0, 3)
    tf.print("prev", tf.shape(positions))
    positions = tf.concat(
        [positions, tf.zeros((add_cols, 2), dtype=tf.float32)], axis=0
    )[:3]
    positions = tf.ensure_shape(positions, (3, 2))
    tf.print("post", tf.shape(positions))

    # Convert positions into correct shape
    positions = tf.transpose(positions)  # (2, 3)

    scores = tf.convert_to_tensor(
        [
            str(s[1])
            for s in sample_info.get("scores", [[0, "HIDDEN"] for _ in range(3)])
        ],
        dtype=tf.string,
    )  # (3,)
    scores = tf.ensure_shape(scores, (3,))

    # Sort by descending y value
    # tf.print(sample_info.get("scores", [0, "HIDDEN"]))
    # tf.print(positions[0])
    order = tf.argsort(positions[0], direction="DESCENDING")
    positions = tf.gather(positions, order, axis=1)
    scores = tf.gather(scores, order)
    return positions, scores


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
        par = dart_order.index(int(s)) % 2
        return 1 if par == 0 else 2  # 1: black, 2: white
    s = s[1:]
    par = dart_order.index(int(s)) % 2
    return 3 if par == 0 else 4  # 3: red, 4: green


def read_positions_and_classes(filepath: tf.Tensor):
    sample_json = tf.io.read_file(filepath)

    # Parse position and score values
    positions, scores = tf.py_function(
        parse_positions_and_scores,
        [sample_json],
        Tout=[tf.float32, tf.string],
    )

    # Process classes
    class_table = get_class_table()
    class_ids = class_table.lookup(scores)  # (3,)
    scores_onehot = tf.one_hot(class_ids, depth=6, dtype=tf.int32)  # (3, 6)
    classes = tf.transpose(scores_onehot)  # (6, 3)

    # Get existences
    existences = 1 - classes[:1]  # (1, 3)
    classes = classes[1:]  # (5, 3)

    existences.set_shape((1, 3))
    positions.set_shape((2, 3))
    classes.set_shape((5, 3))

    return existences, positions, classes


def read_sample_img(filepath: tf.Tensor, img_size: int = 800):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_png(img, channels=3)
    img = tf.reverse(img, axis=[-1])  # BGR
    img = tf.cast(img, tf.float32) / 255  # 0..1

    # Ensure shape consistency
    img.set_shape((img_size, img_size, 3))
    return img


def load_sample(img_path: tf.Tensor, info_path: tf.Tensor):

    # Load data
    img = read_sample_img(img_path)  # (800, 800, 3)

    existences, positions, classes = read_positions_and_classes(info_path)  # (1, 3), (2, 3), (5, 3)

    return img, existences, positions, classes

    """ assure correct position placement: """
    # img = (img.numpy() * 255).astype(np.uint8)
    # positions_s = convert_to_img_positions(outputs[0])
    # positions_m = convert_to_img_positions(outputs[1])
    # positions_l = convert_to_img_positions(outputs[2])

    # import cv2
    # from ma_darts.cv.utils import show_imgs

    # for y, x, _ in positions_s:
    #     cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), 2)
    # show_imgs(img)
    # return img, *[tf.cast(o, tf.float32) for o in outputs]


def dataloader_ma(
    data_dir: str,
    img_size: int = 800,
    shuffle: bool = False,
    augment: bool = False,
    batch_size: int = 4,
    prefetch: int = tf.data.AUTOTUNE,
    cache: bool = True,
    clear_cache: bool = False,
    sample_weight: float = 1.0,
):
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
        ds = ds.shuffle(16 * batch_size)

    # Load Samples into Dataset
    ds = ds.map(
        load_sample,
        num_parallel_calls=tf.data.AUTOTUNE,
    )  # (800, 800, 3), (1, 3), (2, 3), (5, 3)

    ds = finalize_base_ds(
        ds,
        data_dir=data_dir,
        img_size=img_size,
        shuffle=shuffle,
        augment=augment,
        batch_size=batch_size,
        cache=cache,
        clear_cache=clear_cache,
        sample_weight=sample_weight,
    )
    return ds
