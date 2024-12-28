import cv2
import numpy as np


def rotation_matrix(theta: float, in_degrees: bool = False) -> np.ndarray:
    if in_degrees:
        theta = np.deg2rad(theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    M = np.array(
        [
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1],
        ]
    )
    return M


def translation_matrix(x: float = 0, y: float = 0) -> np.ndarray:
    M = np.array(
        [
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1],
        ]
    )
    return M


def shearing_matrix(x: float = 0, y: float = 0) -> np.ndarray:
    M = np.array(
        [
            [1, y, 0],
            [x, 1, 0],
            [0, 0, 1],
        ]
    )
    return M


def scaling_matrix(x: float = 1, y: float | None = None) -> np.ndarray:
    if y is None:
        y = x
    M = np.array(
        [
            [x, 0, 0],
            [0, y, 0],
            [0, 0, 1],
        ]
    )
    return M


def apply_matrix(
    img: np.ndarray,
    M: np.ndarray,
    adapt_frame: bool = False,
) -> np.ndarray:
    if adapt_frame:
        p0 = np.array([0, 0, 1])
        p1 = np.array([0, img.shape[0], 1])
        p2 = np.array([img.shape[1], 0, 1])
        p3 = np.array([img.shape[1], img.shape[0], 1])

        p0 = M @ p0
        p1 = M @ p1
        p2 = M @ p2
        p3 = M @ p3
        min_x = min(p0[0], p1[0], p2[0], p3[0])
        min_y = min(p0[1], p1[1], p2[1], p3[1])
        max_x = max(p0[0], p1[0], p2[0], p3[0])
        max_y = max(p0[1], p1[1], p2[1], p3[1])

        ty = -min_y
        tx = -min_x
        w = round(max_x - min_x)
        h = round(max_y - min_y)

        M_trans = translation_matrix(tx, ty)

        M = M_trans @ M
    else:
        w, h = img.shape[:2][::-1]

    img = cv2.warpPerspective(img, M, (w, h))
    return img
