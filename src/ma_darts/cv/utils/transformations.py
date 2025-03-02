import numpy as np

from ma_darts import radii

_src_pts = None


def get_src_pts():
    global _src_pts
    if _src_pts is not None:
        return _src_pts

    _src_pts = [(400, 400)]
    img = np.zeros((800, 800), np.uint8)
    for i in range(20):
        angle = np.deg2rad(9) + i * np.deg2rad(18)
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        for r in [radii["r_bo"], radii["r_to"], radii["r_do"]]:
            y = 400 - cos_t * r
            x = 400 + sin_t * r
            _src_pts.append((x, y))

    return _src_pts


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

    src_pts = get_src_pts()  # (n, 2)

    # Translate points to original position and back using predicted transformation
    M_re = M_pred @ np.linalg.inv(M_true)
    dst_pts = points_transformation(src_pts, M_re)  # (n, 2)

    diffs = dst_pts - src_pts  # (4, 2)
    dists = np.sqrt(np.square(diffs[:, 0]) + np.square(diffs[:, 1]))
    mean_dist = np.mean(dists)

    # import cv2
    # from ma_darts.cv.utils import show_imgs

    # img = np.zeros((800, 800, 3))
    # for y, x in src_pts:
    #     cv2.circle(img, (int(x), int(y)), 3, (255, 0, 0), 1)
    # for y, x in dst_pts:
    #     cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), 1)
    # show_imgs(img)

    return mean_dist


if __name__ == "__main__":
    from ma_darts.cv.utils import translation_matrix, rotation_matrix

    M_true = rotation_matrix(0, True)
    M_pred = rotation_matrix(10, True)

    m = homography_similarity(M_true, M_pred)
    print(m)
