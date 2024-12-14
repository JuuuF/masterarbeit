import cv2
import numpy as np


def draw_polar_line(
    img,
    rho: float,
    theta: float,
    intensity: float = 1,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    line_type: int = cv2.LINE_8,
    inplace: bool = True,
):
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * rho
    y0 = b * rho

    diag = np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    pt1 = (int(x0 + diag * -b), int(y0 + diag * a))
    pt2 = (int(x0 - diag * -b), int(y0 - diag * a))

    color = tuple(int(c * intensity) for c in color)
    if inplace:
        cv2.line(img, pt1, pt2, color, thickness)
    else:
        img = cv2.line(img.copy(), pt1, pt2, color, thickness)
    return img
