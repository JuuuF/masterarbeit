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
    n_tries = len(src_pts)
    min_points = min(10, len(src_pts))
    Ms = []
    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    src_center = [cx, cy]
    dst_center = [400, 400]

    for i in range(n_tries):
        # Randomly select points
        indices = np.random.choice(len(src_pts), min_points, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]

        # Add mid point
        src_sample = np.insert(src_sample, 0, src_center, axis=0)
        dst_sample = np.insert(dst_sample, 0, dst_center, axis=0)

        M, _ = cv2.findHomography(
            src_sample,
            dst_sample,
            method=cv2.RANSAC,
        )
        if M is not None:
            Ms.append(M)

    if len(Ms) == 0:
        print("ERROR: Could not find alignment homography.")
        return np.eye(3)
    M = np.median(Ms, axis=0)
    return M
