import cv2
import numpy as np
from ma_darts.cv.utils import points_transformation, show_imgs


def get_alignment_matrix(
    src_pts: list[tuple[float, float]],
    dst_pts: list[tuple[float, float]],
    cy: int,
    cx: int,
):
    # If there are not enough points, we return
    if len(src_pts) < 4:
        return np.eye(3)

    n_tries = len(src_pts) * 3
    use_top_n = 1

    src_pts = np.array(src_pts, np.float32)
    dst_pts = np.array(dst_pts, np.float32)
    src_center = [cx, cy]
    dst_center = [400, 400]

    img = np.zeros((800, 800, 3), np.uint8)
    for p in dst_pts:
        cv2.circle(img, [int(p[0]), int(p[1])], 4, (255, 0, 0), 2, cv2.LINE_AA)

    Ms = []
    total_dists = []

    for i in range(n_tries):
        # Randomly sample 3 extra points
        indices = np.random.choice(len(src_pts), 5, replace=False)

        # Sample indices and add center point
        src_sample = np.vstack([src_center, src_pts[indices]])
        dst_sample = np.vstack([dst_center, dst_pts[indices]])

        # Get homography
        M, _ = cv2.findHomography(src_sample, dst_sample)
        if M is None:
            continue

        # Transform all src points
        src_pts_transformed = points_transformation(src_pts[:, ::-1], M)[:, ::-1]

        # Calculate distances
        dists = np.linalg.norm(
            src_pts_transformed - dst_pts,
            axis=1,
            ord=2,
        )

        # Save info
        Ms.append(M)
        total_dists.append(np.sum(dists))

    if len(Ms) == 0:
        print("ERROR: Could not find alignment homography.")
        return np.eye(3)

    # Sort by distances
    sort_ids = np.argsort(total_dists)
    Ms = np.array(Ms)[sort_ids]
    distances = np.array(total_dists)[sort_ids]

    # Use top-n measures
    top_Ms = Ms[:use_top_n]
    M_final = np.median(top_Ms, axis=0)

    # src_pts_transformed = points_transformation(src_pts[:, ::-1], M_final)[:, ::-1]
    # final_dists = np.linalg.norm(src_pts_transformed - dst_pts, axis=1, ord=2).sum()
    # print(final_dists)
    return M_final
