import numpy as np


def points_transformation(
    pts_yx: list[tuple[float, float]] | np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    pts_xy = np.array(pts_yx, np.float32)[:, ::-1]
    pts_xy_h = np.hstack([pts_xy, np.ones((len(pts_xy), 1))])

    pts_xy_h_dst = (M @ pts_xy_h.T).T
    pts_xy_dst = pts_xy_h_dst[:, :2] / pts_xy_h_dst[:, [-1]]
    pts_yx_dst = pts_xy_dst[:, ::-1]
    return pts_yx_dst


def homography_similarity(
    M_true: np.ndarray,  # (3, 3)
    M_pred: np.ndarray,  # (3, 3)
):
    M_true_ = np.linalg.inv(M_true)

    src_pts = [
        (100, 400),  # top
        (700, 400),  # bottom
        (400, 100),  # left
        (400, 700),  # right
        (400, 400),  # center
    ]

    # Translate points to original position and back using predicted transformation
    M_re = M_pred @ np.linalg.inv(M_true)
    dst_pts = points_transformation(src_pts, M_re)  # (4, 2)

    diffs = dst_pts - src_pts  # (4, 2)
    dists = np.sqrt(np.square(diffs[:, 0]) + np.square(diffs[:, 1]))
    mean_dist = np.mean(dists)

    return mean_dist
