import numpy as np

from ma_darts.cv.utils import point_line_distance

def filter_lines_by_center_dist(
    lines: list[tuple[float, float, float, float, float]],
    cy: int,
    cx: int,
    max_center_dist: float = 10,
) -> list[tuple[float, float, float, float, float]]:

    lines_filtered = []
    for line_idx, line in enumerate(lines):
        rho, theta = line[-2:]
        dist = point_line_distance(cy, cx, rho, theta)

        if dist > max_center_dist:
            continue
        lines_filtered.append(
            (
                line[0],  # p1
                line[1],  # p2
                line[2],  # length
                dist,  # distance
                line[3],  # rho
                line[4],  # theta
            )
        )

    return lines_filtered
