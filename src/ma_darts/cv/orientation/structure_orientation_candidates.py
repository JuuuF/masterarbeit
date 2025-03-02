import cv2
import numpy as np
from ma_darts import radii

from ma_darts.cv.utils import show_imgs


def structure_orientation_candidates(
    orientation_point_candidates: list[list[tuple[int, str]]],
    cy: int,
    cx: int,
    img_undistort: np.ndarray | None = None,
):

    # Find outer distances
    outer_dists = []
    for angle_positions in orientation_point_candidates:
        outer_dists += [abs(p[0]) for p in angle_positions if p[1] == "outer"]

    # Find "outer" values
    outer_dists_pos = []
    outer_dists_neg = []
    for point_bin in orientation_point_candidates:
        outers = [p[0] for p in point_bin if p[1] == "outer"]
        outer_pos = [p for p in outers if p > 0]
        outer_neg = [p for p in outers if p < 0]
        outer_dists_pos.append(outer_pos)
        outer_dists_neg.append(outer_neg)
    outer_dists = outer_dists_pos + outer_dists_neg

    # Start off with single values
    outer_measures = [abs(d[0]) if len(d) == 1 else -len(d) for d in outer_dists]

    # Resolve doubles
    mean_dist = np.mean([m for m in outer_measures if m > 0])
    for i, r in enumerate(outer_measures):
        # Identify doubles by negative values
        if r >= 0:
            continue
        # Use whatever measure is closest to mean
        dists = np.abs([abs(d) - mean_dist for d in outer_dists[i]])
        min_idx = np.argmin(dists)
        outer_measures[i] = abs(outer_dists[i][min_idx])

    # Interpolate missing
    outer_radii = outer_measures.copy()
    for i, r in enumerate(outer_measures):
        # Identify by zero values
        if r != 0:
            continue
        # Get previous
        j = (i - 1) % len(outer_measures)
        lower_steps = 1
        while (lower := outer_measures[j]) == 0:
            lower_steps += 1
            j = (j - 1) % len(outer_measures)
        # Get next
        j = (i + 1) % len(outer_measures)
        upper_steps = 1
        while (upper := outer_measures[j]) == 0:
            upper_steps += 1
            j = (j - 1) % len(outer_measures)
        # Interpolate values by distance
        total_steps = lower_steps + upper_steps
        lower_fraction = lower_steps / total_steps
        upper_fraction = upper_steps / total_steps
        interp = lower_fraction * lower + upper_fraction * upper
        outer_radii[i] = interp

    double_thresholds = [o * 1.2 for o in outer_radii]

    src = []
    dst = []

    prepare_show_img = (
        img_undistort is not None
    )  # or create_debug_img  # TODO: debug_img
    if prepare_show_img:
        img = img_undistort.copy() // 2

    for i, angle_positions in enumerate(orientation_point_candidates):
        theta = np.pi / 20 + i * np.pi / 10
        double_threshold = double_thresholds[i]
        for r, pos in angle_positions:
            src_y = cy - np.cos(theta) * r
            src_x = cx + np.sin(theta) * r
            if pos == "outer":
                # triple ring - outside
                dst_r = radii["r_to"] * (1 if r > 0 else -1)
            elif abs(r) > double_threshold:
                # double ring
                dst_r = radii["r_di"] * (1 if r > 0 else -1)
            else:
                # triple ring - inside
                dst_r = radii["r_ti"] * (1 if r > 0 else -1)
            dst_y = 400 - np.cos(theta) * dst_r
            dst_x = 400 + np.sin(theta) * dst_r
            src.append((src_x, src_y))
            dst.append((dst_x, dst_y))

            if prepare_show_img:
                if abs(abs(dst_r) - radii["r_to"]) < 1e-3:
                    c = (255, 160, 160)
                elif abs(abs(dst_r) - radii["r_di"]) < 1e-3:
                    c = (160, 255, 160)
                else:
                    c = (160, 160, 255)
                cv2.circle(
                    img,
                    (int(src_x), int(src_y)),
                    3,
                    c,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
                cv2.circle(
                    img,
                    (int(dst_x), int(dst_y)),
                    3,
                    [i//1.5 for i in c],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
                cv2.arrowedLine(
                    img,
                    (int(src_x), int(src_y)),
                    (int(dst_x), int(dst_y)),
                    (255, 255, 255),
                    line_type=cv2.LINE_AA,
                    tipLength=0.5,
                )

    if prepare_show_img:
        threshold_points = [
            (int(cx + r * np.sin(theta)), int(cy - r * np.cos(theta)))
            for r, theta in zip(
                double_thresholds, np.arange(0, 2 * np.pi, np.pi / 10) + np.pi / 20
            )
        ]
        for i, p in enumerate(threshold_points):
            p2 = threshold_points[(i + 1) % len(threshold_points)]
            cv2.line(
                img,
                p,
                p2,
                color=(200, 200, 200),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        # if show:
        show_imgs(projection_mapping=img, block=False)
        # show_imgs()
    # Utils.append_debug_img(img, "Orientation Point Projections")  # TODO: debug img

    return src, dst  # (x, y), (x, y)
