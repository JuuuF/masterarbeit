import cv2
import numpy as np
import tensorflow as tf

from ma_darts import classes, dart_order
from ma_darts.ai.utils import calculate_scores_ma, split_outputs_to_xst_pos_cls


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


def non_max_suppression(
    pos: np.ndarray,  # bs * (n*, 2)
    cls: np.ndarray,  # bs * (n*)
    cnf: np.ndarray,  # bs * (n*)
    distance_threshold: float = 10.0,
) -> dict[str, np.ndarray]:
    out_pos = []
    out_cls = []
    out_cnf = []
    for pos_batch, cls_batch, cnf_batch in zip(pos, cls, cnf):
        keep_indices = []
        suppressed = np.zeros(len(pos_batch), dtype=bool)

        # Iterate over confidences, starting with biggest
        for i in np.argsort(-cnf_batch):
            # Skip suppressed
            if suppressed[i]:
                continue
            keep_indices.append(i)

            # Compute distances
            dists = np.linalg.norm(pos_batch[i] - pos_batch, axis=1)

            # Suppress too close positions
            suppressed[dists < distance_threshold] = True

        # Apply filtering
        out_pos.append(pos_batch[keep_indices])
        out_cls.append(cls_batch[keep_indices])
        out_cnf.append(cnf_batch[keep_indices])

    return out_pos, out_cls, out_cnf


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

    # Split into components
    xst = pred[..., :1, :]  # (bs, s, s, 1, 3)
    pos = pred[..., 1:3, :]  # (bs, s, s, 2, 3)
    cls = pred[..., 3:, :]  # (bs, s, s, 5, 3)

    # Apply softmax activation
    cls = np.array(tf.keras.activations.softmax(cls, axis=-2))

    # Convert to absolute coordinates
    pos = convert_to_absolute_coordinates(pos)  # (bs, s, s, 2, 3)

    # Extract best classes
    cnf = np.max(cls, axis=-2) * xst[..., 0, :]  # (bs, s, s, 3)
    cls = np.argmax(cls, axis=-2)  # (bs, s, s, 3)

    # Flatten grid dimensions
    batch_size = imgs.shape[0]
    pos = np.transpose(pos, (0, 1, 2, 4, 3))  # (bs, s, s, 3, 2)
    pos = np.reshape(pos, (batch_size, -1, 2))  # (bs, s*s*3, 2)
    cls = np.reshape(cls, (batch_size, -1))  # (bs, s*s*3)
    cnf = np.reshape(cnf, (batch_size, -1))  # (bs, s*s*3)
    xst = np.reshape(xst, (batch_size, -1))  # (bs, s*s*3)

    # Filter out unsure predictions
    # fmt: off
    out_ids = [np.where(xst_batch > confidence_threshold)[0] for xst_batch in xst]  # bs * (m*,)
    pos = [pos_batch[ids] for pos_batch, ids in zip(pos, out_ids)]  # bs * (m*, 2): (y, x)
    cls = [cls_batch[ids] for cls_batch, ids in zip(cls, out_ids)]  # bs * (m*,)
    cnf = [cnf_batch[ids] for cnf_batch, ids in zip(cnf, out_ids)]  # bs * (m*,)
    # fmt: on

    # non-maximum suppression
    pos, cls, cnf = non_max_suppression(pos, cls, cnf)

    # Sort by confidences
    sort_ids = [np.argsort(cnf_batch)[::-1] for cnf_batch in cnf]  # bs * (n*,)
    pos = [pos_batch[sort_ids][0] for pos_batch in pos]
    cls = [cls_batch[sort_ids][0] for cls_batch in cls]
    cnf = [cnf_batch[sort_ids][0] for cnf_batch in cnf]

    # Get scores
    scr = [
        calculate_scores_ma(pos_batch, cls_batch + 1)
        for pos_batch, cls_batch in zip(pos, cls)
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
        for pos_batch, scr_batch, cnf_batch in zip(pos, scr, cnf)
    ]
    if added_batch:
        out = out[0]

    return out
