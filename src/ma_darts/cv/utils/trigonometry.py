import numpy as np


def points_to_polar_line(p1, p2):
    y2, x2 = p2
    y1, x1 = p1

    if x2 >= x1:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    dy = y2 - y1
    dx = x2 - x1
    theta = np.arctan2(-dx, dy)

    rho = x1 * np.cos(theta) + y1 * np.sin(theta)
    theta %= np.pi

    return rho, theta


def point_point_dist(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def point_line_distance(y: int, x: int, rho: float, theta: float) -> float:
    """
    rho = x * cos(theta) + y * sin(theta)
    => x * cos(theta) + y * sin(theta) - rho = 0
    => a = cos(theta), b = sin(theta), c = -rho
    => ax + by + c = 0
    -> https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    # sqrt(a^2 + b^2) = sqrt(sin^2 + cos^2) = 1
    => dist = | cos(theta) * x0 + sin(theta) * y0 - rho |
    """
    dist = abs(np.cos(theta) * x + np.sin(theta) * y - rho)
    return dist


def point_theta_to_polar_line(pt: tuple[int, int], theta: float) -> tuple[float, float]:
    y, x = pt
    rho = x * np.cos(theta) + y * np.sin(theta)
    return rho, theta


def polar_line_intersection(rho_a, theta_a, rho_b, theta_b):
    sin_ta = np.sin(theta_a)
    sin_tb = np.sin(theta_b)
    cos_ta = np.cos(theta_a)
    cos_tb = np.cos(theta_b)

    det = cos_ta * sin_tb - sin_ta * cos_tb

    # No intersection
    if abs(det) < 1e-10:
        return (0, 0)

    y = (rho_b * cos_ta - rho_a * cos_tb) / det
    x = (rho_a * sin_tb - rho_b * sin_ta) / det

    return y, x
