import numpy as np

from ma_darts.cv.utils import polar_line_intersection


def center_point_from_lines(
    lines: list[tuple[float, float]],
) -> tuple[float, float]:
    ys = []
    xs = []
    for i, line_a in enumerate(lines):
        for line_b in lines[i + 1 :]:
            y, x = polar_line_intersection(*line_a, *line_b)
            ys.append(y)
            xs.append(x)
    cy = np.mean(ys)
    cx = np.mean(xs)
    return cy, cx
