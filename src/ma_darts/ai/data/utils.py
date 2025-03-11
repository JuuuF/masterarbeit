import os
import tensorflow as tf
from ma_darts.ai.data import Augmentation


def get_out_grid(out_size, n_cols):
    # Start with a blank cell
    cell_col = tf.zeros((n_cols), tf.float32)  # (n,)
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


def scaled_out(
    xst: tf.Tensor,  # (1, 3)
    pos: tf.Tensor,  # (2, 3)
    cls: tf.Tensor,  # (6, 3)
    img_size: int,
    out_size: int,
):
    out_grid = get_out_grid(
        out_size,
        tf.shape(xst)[0] + tf.shape(pos)[0] + tf.shape(cls)[0],
    )  # (y, x, 8, 3)

    cell_size = img_size // out_size
    pos_abs = pos * img_size  # (2, 3)
    pos_abs = tf.clip_by_value(pos_abs, 0, img_size - 1)

    grid_pos = tf.cast(pos_abs // cell_size, tf.int32)  # (2, 3)
    local_pos = (pos_abs % cell_size) / cell_size  # (2, 3)
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

    # 0. ------------------------------------
    # Get target cell
    grid_y, grid_x = grid_pos[0, 0], grid_pos[1, 0]
    current_cell = out_grid[grid_y, grid_x]

    # Get next available cell column
    cell_col = tf.argmax(tf.cast(current_cell[2] == 1, tf.int32))
    cell_col = tf.cast(cell_col, tf.int32)

    # Update position and class data
    pos_update = tf.concat([xst[:, 0], local_pos[:, 0], cls[:, 0]], axis=0)

    # Get update indices
    indices = []
    indices.append([grid_y, grid_x, 0, cell_col])
    indices.append([grid_y, grid_x, 1, cell_col])
    indices.append([grid_y, grid_x, 2, cell_col])
    indices.append([grid_y, grid_x, 3, cell_col])
    indices.append([grid_y, grid_x, 4, cell_col])
    indices.append([grid_y, grid_x, 5, cell_col])
    indices.append([grid_y, grid_x, 6, cell_col])
    indices.append([grid_y, grid_x, 7, cell_col])
    updates = []
    updates.append(pos_update[0])
    updates.append(pos_update[1])
    updates.append(pos_update[2])
    updates.append(pos_update[3])
    updates.append(pos_update[4])
    updates.append(pos_update[5])
    updates.append(pos_update[6])
    updates.append(pos_update[7])

    # Apply updates
    indices = tf.convert_to_tensor(indices, dtype=tf.int32)
    updates = tf.convert_to_tensor(updates, dtype=tf.float32)
    out_grid = tf.tensor_scatter_nd_update(out_grid, indices, updates)

    # 1. ------------------------------------
    # Get target cell
    grid_y, grid_x = grid_pos[0, 1], grid_pos[1, 1]
    current_cell = out_grid[grid_y, grid_x]

    # Get next available cell column
    cell_col = tf.argmax(tf.cast(current_cell[2] == 1, tf.int32))
    cell_col = tf.cast(cell_col, tf.int32)

    # Update position and class data
    pos_update = tf.concat([xst[:, 1], local_pos[:, 1], cls[:, 1]], axis=0)

    # Get update indices
    indices = []
    indices.append([grid_y, grid_x, 0, cell_col])
    indices.append([grid_y, grid_x, 1, cell_col])
    indices.append([grid_y, grid_x, 2, cell_col])
    indices.append([grid_y, grid_x, 3, cell_col])
    indices.append([grid_y, grid_x, 4, cell_col])
    indices.append([grid_y, grid_x, 5, cell_col])
    indices.append([grid_y, grid_x, 6, cell_col])
    indices.append([grid_y, grid_x, 7, cell_col])
    updates = []
    updates.append(pos_update[0])
    updates.append(pos_update[1])
    updates.append(pos_update[2])
    updates.append(pos_update[3])
    updates.append(pos_update[4])
    updates.append(pos_update[5])
    updates.append(pos_update[6])
    updates.append(pos_update[7])

    # Apply updates
    indices = tf.convert_to_tensor(indices, dtype=tf.int32)
    updates = tf.convert_to_tensor(updates, dtype=tf.float32)
    out_grid = tf.tensor_scatter_nd_update(out_grid, indices, updates)

    # 2. ------------------------------------
    # Get target cell
    grid_y, grid_x = grid_pos[0, 2], grid_pos[1, 2]
    current_cell = out_grid[grid_y, grid_x]

    # Get next available cell column
    cell_col = tf.argmax(tf.cast(current_cell[2] == 1, tf.int32))
    cell_col = tf.cast(cell_col, tf.int32)

    # Update position and class data
    pos_update = tf.concat([xst[:, 2], local_pos[:, 2], cls[:, 2]], axis=0)

    # Get update indices
    indices = []
    indices.append([grid_y, grid_x, 0, cell_col])
    indices.append([grid_y, grid_x, 1, cell_col])
    indices.append([grid_y, grid_x, 2, cell_col])
    indices.append([grid_y, grid_x, 3, cell_col])
    indices.append([grid_y, grid_x, 4, cell_col])
    indices.append([grid_y, grid_x, 5, cell_col])
    indices.append([grid_y, grid_x, 6, cell_col])
    indices.append([grid_y, grid_x, 7, cell_col])
    updates = []
    updates.append(pos_update[0])
    updates.append(pos_update[1])
    updates.append(pos_update[2])
    updates.append(pos_update[3])
    updates.append(pos_update[4])
    updates.append(pos_update[5])
    updates.append(pos_update[6])
    updates.append(pos_update[7])

    # Apply updates
    indices = tf.convert_to_tensor(indices, dtype=tf.int32)
    updates = tf.convert_to_tensor(updates, dtype=tf.float32)
    out_grid = tf.tensor_scatter_nd_update(out_grid, indices, updates)

    return out_grid

    # updates = []
    # indices = []

    # for i in tf.range(3):
    #     # Get target cell
    #     grid_y, grid_x = grid_pos[0, i], grid_pos[1, i]
    #     current_cell = out_grid[grid_y, grid_x]

    #     # Get next available cell column
    #     cell_col = tf.argmax(tf.cast(current_cell[2] == 1, tf.int32))
    #     cell_col = tf.cast(cell_col, tf.int32)
    #     # tf.print("cell_col:")
    #     # tf.print(cell_col)

    #     # Update position and class data
    #     pos_update = tf.concat([local_pos[:, i], cls[:, i]], axis=0)

    #     # Get update indices
    #     for cell_row in tf.range(tf.shape(pos_update)[0], dtype=tf.int32):
    #         indices.append([grid_y, grid_x, cell_row, cell_col])
    #         updates.append(pos_update[cell_row])

    # indices = tf.convert_to_tensor(indices, dtype=tf.int32)
    # updates = tf.convert_to_tensor(updates, dtype=tf.float32)

    # # Apply updates
    # out_grid = tf.tensor_scatter_nd_update(out_grid, indices, updates)

    # # tf.print("--- indices + updates")
    # # for i, u in zip(indices, updates):
    # #     tf.print(i, ":", u, "->", out_grid[i[0], i[1], i[2], i[3]])
    # # tf.print("-"*50)

    # return out_grid


@tf.function
def positions_to_yolo(
    img: tf.Tensor,  # (800, 800, 3)
    xst: tf.Tensor,  # (1, 3)
    pos: tf.Tensor,  # (2, 3)
    cls: tf.Tensor,  # (6, 3)
):
    xst = tf.cast(xst, tf.float32)
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

    # Remove existing cache files
    if clear_cache and os.path.exists(cache_file):
        os.remove(cache_file)

    # Create clean cache directory
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
) -> tf.data.Dataset:

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
