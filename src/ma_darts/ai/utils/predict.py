import cv2
import numpy as np
import tensorflow as tf

from ma_darts import classes, dart_order
from ma_darts.ai.utils import calculate_scores_ma

colors = [
    (50, 50, 50),  # nothing
    (0, 0, 0),  # black
    (127, 127, 127),  # white
    (0, 0, 255),  # red
    (0, 255, 0),  # green
    (127, 127, 127),  # out
]


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


def draw_prediction(
    img: np.ndarray,  # (800, 800, 3)
    pos: np.ndarray,  # (2,)
    cls_id: int,
    score: str,
    cnf: float,
):
    y, x = pos
    y, x = int(y), int(x)
    # Draw circle at position
    color = tuple(c * cnf for c in colors[cls_id])
    cv2.circle(img, (x, y), 2, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x, y), 3, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.circle(img, (x, y), 4, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    def add_text(txt, pos):
        txt_params = dict(
            img=img,
            text=txt,
            org=pos,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.25,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            **txt_params,
            thickness=2,
            color=(255, 255, 255),
        )
        cv2.putText(
            **txt_params,
            thickness=1,
            color=color,
        )

    add_text(str(score), pos=(x + 5, y - 2))
    add_text(f"({round(cnf * 100)}%)", pos=(x + 5, y + 8))

    # Draw text label

    return img


def yolo_v8_predict(
    model: tf.keras.Model,
    imgs: np.array,  # (bs, 800, 800, 3)
    confidence_threshold: float = 0.5,
    output_img: bool = False,
):

    # Add batch dimension
    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, 0)

    # Predict images
    pred = model.predict(imgs, verbose=0)

    # Convert to absolute coordinates
    pos_pred = convert_to_absolute_coordinates(pred[..., :2, :])

    # Extract best classes
    cls_pred = np.argmax(pred[..., 2:, :], axis=-2)  # (bs, s, s, 3)
    cnf_pred = np.max(pred[..., 2:, :], axis=-2)  # (bs, s, s, 3)

    # Flatten grid dimensions
    n_samples = imgs.shape[0]
    pos_pred = np.transpose(pos_pred, (0, 1, 2, 4, 3))  # (bs, s, s, 3, 2)
    pos_pred = np.reshape(pos_pred, (n_samples, -1, 2))  # (bs, s*s*3, 2)
    cls_pred = np.reshape(cls_pred, (n_samples, -1))  # (bs, s*s*3)
    cnf_pred = np.reshape(cnf_pred, (n_samples, -1))  # (bs, s*s*3)

    # Filter out unsure predictions
    out_ids = np.where(cnf_pred > confidence_threshold)[1]
    pos_pred = pos_pred[:, out_ids]
    cls_pred = cls_pred[:, out_ids]
    cnf_pred = cnf_pred[:, out_ids]

    # Filter out nothing predictions
    out_ids = np.where(cls_pred != 0)[1]
    pos_pred = pos_pred[:, out_ids]
    cls_pred = cls_pred[:, out_ids]
    cnf_pred = cnf_pred[:, out_ids]

    scores_pred = [
        calculate_scores_ma(pos, cls) for pos, cls in zip(pos_pred, cls_pred)
    ]

    ma_outputs = [
        {
            "dart_positions": pos,
            "scores": scores,
        }
        for scores, pos in zip(pos_pred, scores_pred)
    ]
    if not output_img:
        return ma_outputs

    # Start drawing
    imgs_out = np.uint8(imgs * 255)  # (bs, 800, 800, 3)

    for sample_idx, (poss, cnfs, clss, scores) in enumerate(
        zip(pos_pred, cnf_pred, cls_pred, scores_pred)
    ):
        overlay = imgs_out[sample_idx].copy()
        for pos, cnf, cls_id, (score_val, score_str) in zip(poss, cnfs, clss, scores):
            overlay = draw_prediction(overlay, pos, cls_id, score_str, cnf)
        blend = 0.6
        imgs_out[sample_idx] = cv2.addWeighted(
            imgs_out[sample_idx], (1 - blend), overlay, blend, 0.0
        )

    out_img = np.hstack(imgs_out)
    return ma_outputs, out_img

    from ma_darts.cv.utils import show_imgs

    show_imgs(*imgs_out)
    return imgs_out
