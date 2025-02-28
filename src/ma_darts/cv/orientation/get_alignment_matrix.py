import cv2
import numpy as np


def get_alignment_matrix(
    src_pts: list[tuple[float, float]],
    dst_pts: list[tuple[float, float]],
    cy: int,
    cx: int,
) -> np.ndarray:
    # If there are not enough points, we return
    if len(src_pts) < 4:
        return np.eye(3)

    # In case we still have some outliers, we randomly sample 75% of the src and dst points
    # and find a homography between these point sets. In the end, we take the mean of all homographies
    n_tries = 16
    ransac_percent = 0.75
    Ms = []
    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    src_center = [cx, cy]
    dst_center = [400, 400]
    n_ransac_points = max(4, int(ransac_percent * len(src_pts)))
    for i in range(n_tries):
        try_indices = np.random.permutation(len(src_pts))[:n_ransac_points]
        try_src = src_pts[try_indices]
        try_dst = dst_pts[try_indices]
        M, _ = cv2.findHomography(
            np.insert(try_src, 0, src_center, axis=0),  # always add center points
            np.insert(try_dst, 0, dst_center, axis=0),
            method=cv2.RANSAC,
        )
        if M is not None:
            Ms.append(M)

    if len(Ms) == 0:
        print("ERROR: Could not find alignment homography.")
        return np.eye(3)
    M = np.median(Ms, axis=0)
    return M
