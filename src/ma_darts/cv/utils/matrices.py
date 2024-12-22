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
