import cv2
import numpy as np


def draw_polar_line(
    img: np.ndarray,
    rho: float,
    theta: float,
    intensity: float = 1,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    line_type: int = cv2.LINE_8,
    inplace: bool = True,
) -> np.ndarray:
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


def draw_polar_line_through_point(
    img: np.ndarray,
    pt: tuple[int, int],  # (y, x)
    theta: float,
    intensity: float = 1,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    line_type: int = cv2.LINE_8,
    inplace: bool = True,
):
    # Get variables
    a = np.cos(theta)
    b = np.sin(theta)
    y, x = pt

    # Calculate line points
    diag = np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    y0 = int(y + diag * a)
    x0 = int(x - diag * b)
    y1 = int(y - diag * a)
    x1 = int(x + diag * b)

    # Draw line
    color = tuple(int(c * intensity) for c in color)
    if inplace:
        cv2.line(img, (x0, y0), (x1, y1), color, thickness, line_type)
    else:
        img = cv2.line(img.copy(), (x0, y0), (x1, y1), color, thickness, line_type)

    return img
