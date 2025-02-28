import cv2
import numpy as np
import tensorflow as tf

from ma_darts import classes, dart_order
from ma_darts.ai.utils import calculate_scores_ma


def convert_to_absolute_coordinates(
    pos: np.ndarray,  # (bs, s, s, 2, 3)
):
    s = pos.shape[1]
    cell_size = 800 // s

    grid_indices = np.stack(
        np.meshgrid(np.arange(s), np.arange(s), indexing="ij"), axis=-1
    )  # (s, s, 2)
    global_grid_pos = grid_indices * cell_size  # (s, s, 2)
    pos_abs = global_grid_pos[None, ..., None] + pos * cell_size
    return pos_abs  # (bs, s, s, 2, 3)


def yolo_v8_predict(
    model: tf.keras.Model,
    imgs: np.array,  # (bs, 800, 800, 3)
    confidence_threshold: float = 0.5,
) -> list[
    list[tuple[tuple[float, float], tuple[int, str], float]]
]:  # [((y, x), (score_val, score_str), confidence)]

    # Add batch dimension
    added_batch = False
    if len(imgs.shape) == 3:
        added_batch = True
        imgs = np.expand_dims(imgs, 0)

    # Predict images
    pred = model.predict(imgs, verbose=0)  # (bs, s, s, 8, 3)

    # Convert to absolute coordinates
    pos_pred = convert_to_absolute_coordinates(pred[..., :2, :])  # (bs, s, s, 8, 3)

    # Extract best classes
    cls_pred = np.argmax(pred[..., 2:, :], axis=-2)  # (bs, s, s, 3)
    cnf_pred = np.max(pred[..., 2:, :], axis=-2)  # (bs, s, s, 3)

    # Flatten grid dimensions
    batch_size = imgs.shape[0]
    pos_pred = np.transpose(pos_pred, (0, 1, 2, 4, 3))  # (bs, s, s, 3, 2)
    pos_pred = np.reshape(pos_pred, (batch_size, -1, 2))  # (bs, s*s*3, 2)
    cls_pred = np.reshape(cls_pred, (batch_size, -1))  # (bs, s*s*3)
    cnf_pred = np.reshape(cnf_pred, (batch_size, -1))  # (bs, s*s*3)

    # Filter out unsure predictions
    # fmt: off
    out_ids = [np.where(cnf_batch > confidence_threshold)[0] for cnf_batch in cnf_pred]  # bs * (m*,)
    pos_pred = [pos_batch[ids] for pos_batch, ids in zip(pos_pred, out_ids)]  # bs * (m*, 2): (y, x)
    cls_pred = [cls_batch[ids] for cls_batch, ids in zip(cls_pred, out_ids)]  # bs * (m*,)
    cnf_pred = [cnf_batch[ids] for cnf_batch, ids in zip(cnf_pred, out_ids)]  # bs * (m*,)

    # Filter out nothing predictions
    out_ids = [np.where(cls_batch != 0)[0] for cls_batch in cls_pred]  # bs * (n*,)
    pos_pred = [pos_batch[ids] for pos_batch, ids in zip(pos_pred, out_ids)]  # bs * (n*, 2): (y, x)
    cls_pred = [cls_batch[ids] for cls_batch, ids in zip(cls_pred, out_ids)]  # bs * (n*,)
    cnf_pred = [cnf_batch[ids] for cnf_batch, ids in zip(cnf_pred, out_ids)]  # bs * (n*,)

    # Sort by confidences
    sort_ids = [np.argsort(cnf_batch)[::-1] for cnf_batch in cnf_pred]  # bs * (n*,)
    pos_pred = [pos_batch[sort_ids][0] for pos_batch in pos_pred]
    cls_pred = [cls_batch[sort_ids][0] for cls_batch in cls_pred]
    cnf_pred = [cnf_batch[sort_ids][0] for cnf_batch in cnf_pred]
    # fmt: on

    # Get scores
    scr_pred = [
        calculate_scores_ma(pos_batch, cls_batch)
        for pos_batch, cls_batch in zip(pos_pred, cls_pred)
    ]  # bs * (n*, 2): (int, str)

    # out = [
    #     list(zip(pos_batch, scores_batch, cnf_batch))
    #     for pos_batch, scores_batch, cnf_batch in zip(pos_pred, scr_pred, cnf_pred)
    # ]  # bs * n * [(y, x), (score_val, score_str), confidence]
    out = [
        {
            "dart_positions": pos_batch,
            "scores": scr_batch,
            "confidences": cnf_batch,
        }
        for pos_batch, scr_batch, cnf_batch in zip(pos_pred, scr_pred, cnf_pred)
    ]
    if added_batch:
        out = out[0]

    return out
