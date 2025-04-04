import os
import tensorflow as tf
from ma_darts.ai.data import Augmentation


# @tf.function
def get_out_grid(out_size, n_cols):
    # Start with a blank cell
    cell_col = tf.zeros((n_cols), tf.float32)  # (n,)
    # Extend to cell
    cell_col = tf.expand_dims(cell_col, -1)  # (n, 1)
    cell = tf.repeat(cell_col, repeats=3, axis=-1)  # (n, 3)
    # Extend to row
    grid_row = tf.repeat(tf.expand_dims(cell, 0), out_size, axis=0)  # (x, n, 3)
    # Extend to grid
    grid = tf.repeat(tf.expand_dims(grid_row, 0), out_size, axis=0)  # (y, x, n, 3)
    return grid


# @tf.function
def update_block(i, grid, grid_pos, xst, local_pos, cls):
    # Get target cell
    grid_y = grid_pos[0, i]
    grid_x = grid_pos[1, i]
    current_cell = grid[grid_y, grid_x]  # (8, 3)

    # Get next available cell column
    cell_available = tf.cast(tf.equal(current_cell[0, :], 0), tf.int32)
    cell_col = tf.argmax(cell_available, output_type=tf.int32)

    # Setup updates and indices
    updates = tf.concat([xst[:, i], local_pos[:, i], cls[:, i]], axis=0)
    indices = tf.stack(
        [
            tf.fill([8], grid_y),
            tf.fill([8], grid_x),
            tf.range(8, dtype=tf.int32),
            tf.fill([8], cell_col),
        ],
        axis=1,
    )
    grid_res = tf.tensor_scatter_nd_update(grid, indices, updates)
    return grid_res


# @tf.function
def scaled_out(
    xst: tf.Tensor,  # (1, 3)
    pos: tf.Tensor,  # (2, 3)
    cls: tf.Tensor,  # (6, 3)
    img_size: int,
    out_size: int,
):
    cls = tf.cast(cls, tf.float32)
    out_grid = get_out_grid(
        out_size,
        tf.shape(xst)[0] + tf.shape(pos)[0] + tf.shape(cls)[0],
    )  # (y, x, 8, 3)

    cell_size = tf.constant(img_size // out_size, tf.float32)
    pos_abs = tf.cast(pos * img_size, tf.float32)  # (2, 3)
    pos_abs = tf.clip_by_value(pos_abs, 0.0, tf.cast(img_size - 1, tf.float32))

    grid_pos = tf.cast(pos_abs / cell_size, tf.int32)  # (2, 3)
    local_pos = tf.math.floormod(pos_abs, cell_size) / cell_size  # (2, 3)

    out_grid = update_block(0, out_grid, grid_pos, xst, local_pos, cls)
    out_grid = update_block(1, out_grid, grid_pos, xst, local_pos, cls)
    out_grid = update_block(2, out_grid, grid_pos, xst, local_pos, cls)
    return out_grid


# @tf.function
def positions_to_yolo(
    img: tf.Tensor,  # (800, 800, 3)
    xst: tf.Tensor,  # (1, 3)
    pos: tf.Tensor,  # (2, 3)
    cls: tf.Tensor,  # (6, 3)
):
    xst = tf.cast(xst, tf.float32)
    pos = tf.cast(pos, tf.float32)
    cls = tf.cast(cls, tf.float32)
    out_s = scaled_out(xst, pos, cls, 800, 25)
    # out_m = scaled_out(pos, cls, 800, 50)
    # out_l = scaled_out(pos, cls, 800, 100)
    return img, out_s  # , out_m, out_l


def cache_ds(
    ds: tf.data.Dataset,
    data_dir: str,
    clear_cache: bool = False,
) -> tf.data.Dataset:

    # Get cache files
    cache_base = "data/cache/datasets"
    cache_id = data_dir.replace("/", "-").rstrip("-")
    cache_file = os.path.join(cache_base, cache_id + ".tfdata")
    print("caching to", cache_file, flush=True)

    # Remove existing cache files
    if clear_cache and os.path.exists(cache_file):
        os.remove(cache_file)

    # Create clean cache directory
    if not os.path.exists(cache_base):
        os.makedirs(cache_base, exist_ok=True)

    # Cache to directory
    return ds.cache(cache_file)


def finalize_base_ds(
    ds: tf.data.Dataset,
    data_dir: str,
    img_size: int,
    shuffle: bool = False,
    augment: bool = False,
    batch_size: int = 4,
    prefetch: int = tf.data.AUTOTUNE,
    cache: bool = True,
    clear_cache: bool = False,
    sample_weight: float = 1.0,
) -> tf.data.Dataset:

    # Assure data types
    ds = ds.map(
        lambda img, xst, pos, cls: (
            tf.cast(img, tf.float32),
            tf.cast(xst, tf.float32),
            tf.cast(pos, tf.float32),
            tf.cast(cls, tf.float32),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # Set shapes
    ds = ds.map(
        lambda img, xst, pos, cls: (
            tf.ensure_shape(img, [img_size, img_size, 3]),
            tf.ensure_shape(xst, [1, 3]),
            tf.ensure_shape(pos, [2, 3]),
            tf.ensure_shape(cls, [5, 3]),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if cache:
        ds = cache_ds(ds, data_dir)

    if shuffle:
        ds = ds.shuffle(32 * batch_size)

    if augment:
        aug = Augmentation()
        ds = ds.map(
            aug,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Convert to yolo outputs
    ds = ds.map(
        positions_to_yolo,
        num_parallel_calls=tf.data.AUTOTUNE,
    )  # (800, 800, 3), (25, 25, 8, 3)

    # Set shapes
    ds = ds.map(
        lambda img, out_s: (
            tf.ensure_shape(img, [img_size, img_size, 3]),
            tf.ensure_shape(out_s, [img_size // 32, img_size // 32, 1 + 2 + 5, 3]),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Add weights
    ds = ds.map(
        lambda img, out: (img, out, tf.constant(sample_weight, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.padded_batch(batch_size)
    ds = ds.prefetch(prefetch)
    return ds


def dummy_ds(batch_size=4, n_samples=128):
    # Define a function that generates samples
    def generator():
        input_sample = tf.random.uniform((800, 800, 3), dtype=tf.float32)
        output_sample = (
            tf.random.uniform((25, 25, 8, 3), dtype=tf.float32),
            tf.random.uniform((50, 50, 8, 3), dtype=tf.float32),
            tf.random.uniform((100, 100, 8, 3), dtype=tf.float32),
        )

        for _ in range(n_samples):
            yield (
                input_sample,
                output_sample,
            )

    # Define output signature for TensorFlow
    output_signature = (
        tf.TensorSpec(shape=(800, 800, 3), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(25, 25, 8, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(50, 50, 8, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(100, 100, 8, 3), dtype=tf.float32),
        ),
    )

    # Create dataset
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    ds = ds.apply(tf.data.experimental.assert_cardinality(n_samples))

    # Batch the dataset
    ds = ds.batch(batch_size)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
