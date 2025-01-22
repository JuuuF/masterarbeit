import os
import cv2
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# from rich import print
from scipy.ndimage import label, center_of_mass

from ma_darts.cv.cv import extract_center
from ma_darts.cv.utils import draw_polar_line, show_imgs

added_keys = [
    "h_line_r",
    "h_line_theta",
    "v_line_r",
    "v_line_theta",
    "orientation_points",
    "ellipse_cx",
    "ellipse_cy",
    "ellipse_w",
    "ellipse_h",
    "ellipse_theta",
    "undistortion_homography",
    "dart_positions",
    "dart_positions_undistort",
    "dart_iou",
    "dart_tips_covered",
    "cv_ellipse_cy",
    "cv_ellipse_cx",
]


def is_prepared(sample_info: pd.Series):
    return all(k in sample_info.keys() for k in added_keys)


class ImageUtils:

    def load_sample_imgs(sample_info: pd.Series) -> tuple[
        np.ndarray,  # (y, x, 3)
        np.ndarray,  # (y, x, 3)
        np.ndarray,  # (y, x, 3)
        np.ndarray,  # (y, x, 3)
    ]:
        img_dir = os.path.join(sample_info["OUT_DIR"], str(sample_info.sample_id))
        img = cv2.imread(os.path.join(img_dir, "render.png"))
        img_orient = cv2.imread(
            os.path.join(img_dir, "mask_Board_Orientation.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        img_area = cv2.imread(
            os.path.join(img_dir, "mask_Darts_Board_Area.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        img_intersections = ImageUtils.load_intersections_img(sample_info)
        if any(i is None for i in [img, img_orient, img_area, img_intersections]):
            raise AssertionError(f"Images at {img_dir} could not be read.")
        return img, img_orient, img_area, img_intersections

    def load_intersections_img(sample_info: pd.Series):
        intersections_img = MaskActions.load_mask(
            sample_info.out_file_template.format(filename="mask_Intersections.png")
        )
        dart_masks = [
            MaskActions.load_mask(
                sample_info.out_file_template.format(filename=f"mask_Dart_{i}.png")
            )
            for i in range(1, 4)
        ]
        dart_masks = list(enumerate(dart_masks))

        out_img = np.zeros(
            (intersections_img.shape[0], intersections_img.shape[1], 3), np.uint8
        )
        current_idx = -1
        while dart_masks:
            current_idx += 1
            current_idx %= len(dart_masks)
            dart_id, dart_mask = dart_masks[current_idx]

            # Cont number of intersections between dart mask and intersection image
            intersection = np.logical_and(intersections_img, dart_mask)
            contours, _ = cv2.findContours(
                np.uint8(intersection) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            n_intersections = len(contours)

            # only a single intersection means we are certain which dart we are looking at
            if n_intersections == 1:
                # Remove current mask
                dart_masks.pop(current_idx)
                current_idx -= 1

                # add dart index as color into the output intersection image
                out_img[:, :, dart_id] = np.uint8(intersection) * 255

                # Remove intersection point from image
                intersections_img = np.logical_and(
                    intersections_img, np.logical_not(intersection)
                )
        return out_img

    def points_from_intersection(
        intersection_img: np.ndarray,  # (y, x)
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        # Resolve clusterings
        intersection_img //= 255  # 0/1 values

        # Label connected components
        labeled_array, n_clusters = label(
            intersection_img,
            structure=np.ones((3, 3)),
        )
        assert n_clusters == 2, "Found too many clusters. Or none."

        # Find centers
        centroids = center_of_mass(
            intersection_img, labeled_array, range(1, n_clusters + 1)
        )
        return centroids

    def intersect_imgs(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
        intersected = np.bitwise_and(img_a, img_b)
        # clean up images
        intersected = cv2.morphologyEx(
            intersected,
            cv2.MORPH_OPEN,
            kernel=(2, 2),
            iterations=1,
        )
        return intersected

    def undistort(
        img: np.ndarray,  # (y, x, 3)
        orientation_points: tuple[float, float, float, float],  # (t, r, b, l)
    ) -> tuple[np.ndarray, np.ndarray]:  # (800, 800, 3), (3, 3)

        # Data as used in the paper
        dst_size = 800
        margin = 100

        def get_dst_points() -> np.ndarray:  # (5, 2): top, right, bottom, left, center
            c = dst_size / 2
            dst_pts = []

            r = dst_size // 2 - margin
            for angle in range(4):
                t = np.deg2rad(angle * 90 - 9)
                px = c + r * np.sin(t)
                py = c - r * np.cos(t)
                dst_pts.append((py, px))

            # dst_pts.append((c, c))

            return np.array(dst_pts, np.float32)  # (4, 2): t, r, b, l, c

        def transform_image(
            img: np.ndarray,  # (y, x, 3)
            src_pts: np.ndarray,  # (5, 2)
            dst_pts: np.ndarray,  # (5, 2)
        ) -> tuple[np.ndarray, np.ndarray]:  # (800, 800, 3), (3, 3)
            # we need to switch the element order from y, x to x, y because cv2 is special
            src_pts = src_pts[:, ::-1]
            dst_pts = dst_pts[:, ::-1]

            H, _ = cv2.findHomography(src_pts, dst_pts)
            img = cv2.warpPerspective(img, H, (dst_size, dst_size))

            return img, H

        # Get points
        src_pts = np.array(
            orientation_points,
        )
        dst_pts = get_dst_points()

        img_undistorted, undistortion_homography = transform_image(
            img, src_pts, dst_pts
        )

        return img_undistorted, undistortion_homography


class LinAlg:

    def polar_line_intersection(
        line_a: tuple[float, float],
        line_b: tuple[float, float],
    ) -> tuple[int, int]:
        r1, theta1 = line_a
        r2, theta2 = line_b

        A1 = np.cos(theta1)
        B1 = np.sin(theta1)
        C1 = r1
        A2 = np.cos(theta2)
        B2 = np.sin(theta2)
        C2 = r2

        A = np.array([[A1, B1], [A2, B2]])
        B = np.array([C1, C2])
        x, y = np.linalg.solve(A, B)

        return round(y), round(x)

    def get_line_eq(
        p: tuple[int, int],
        q: tuple[int, int],
    ) -> tuple[float, float]:
        y0, x0 = p
        y1, x1 = q

        A = y1 - y0
        B = x0 - x1
        C = y0 * (x1 - x0) - x0 * (y1 - y0)

        r = abs(C) / np.sqrt(A**2 + B**2)
        theta = np.arctan2(B, A)

        return r, theta


class MaskActions:
    def load_mask(filepath: str) -> np.ndarray:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return img

    def get_lines_from_point_masks(
        img: np.ndarray,  # (y, x)
    ) -> tuple[
        tuple[float, float], tuple[float, float], tuple[float, float, float, float]
    ]:

        def get_moment_center(
            M: dict,
        ) -> tuple[float, float]:
            assert M["m00"] != 0, "Could not find line center. Whoops."
            cy = M["m01"] / M["m00"]
            cx = M["m10"] / M["m00"]
            return cy, cx

        def extract_centers(
            img: np.ndarray,  # (y, x)
        ) -> list[tuple[float, float]]:
            _, img = cv2.threshold(img, 64, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            centers = [get_moment_center(cv2.moments(c)) for c in contours]
            return centers

        centers = extract_centers(img)
        top = sorted(centers, key=lambda c: c[0])[0]  # min y
        bot = sorted(centers, key=lambda c: c[0])[-1]  # max y
        lft = sorted(centers, key=lambda c: c[1])[0]  # min x
        rgt = sorted(centers, key=lambda c: c[1])[-1]  # max x

        # Don't touch, the order is important!
        line_v = LinAlg.get_line_eq(top, bot)
        line_h = LinAlg.get_line_eq(rgt, lft)

        return line_v, line_h, (top, rgt, bot, lft)

    def get_ellipse_from_mask(
        img: np.ndarray,  # (y, x)
    ) -> tuple[tuple[int, int], tuple[int, int], float]:

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        points = np.column_stack(np.where(thresh.transpose() > 0))
        hull = cv2.convexHull(points)[:, 0]

        (cx, cy), (w, h), theta = cv2.fitEllipseDirect(
            hull
        )  # this function will output w <= h

        return (cx, cy), (w, h), theta

    def get_dart_positions(
        sample_info: pd.Series,
        img: np.ndarray,  # (y, x, 3)
    ) -> list[tuple[float, float]]:

        centers = []
        for i in range(3):
            # Get contour
            _, dart_intersect_img = cv2.threshold(
                img[:, :, i], 64, 255, cv2.THRESH_BINARY
            )
            contour = cv2.findContours(
                dart_intersect_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0][0]

            # Extract center point
            M = cv2.moments(contour)
            cy = M["m01"] / M["m00"]
            cx = M["m10"] / M["m00"]
            cy /= img.shape[0]
            cx /= img.shape[1]
            centers.append((cy, cx))

        return centers

    def calculate_darts_iou(sample_info: pd.Series):

        def unite(*imgs: list[np.ndarray]) -> np.ndarray:
            union = imgs[0]
            for img in imgs[1:]:
                union = np.logical_or(union, img)
            return union

        darts_masks = [
            MaskActions.load_mask(
                sample_info["out_file_template"].format(filename=f"mask_Dart_{i}.png")
            )
            for i in range(1, 4)
        ]
        union = unite(darts_masks)
        intersections = []
        for i, dart_a in enumerate(darts_masks):
            for dart_b in darts_masks[i + 1 :]:
                intersection = np.logical_and(dart_a, dart_b)
                intersections.append(intersection)
        intersections = unite(intersections)
        iou = np.count_nonzero(intersections) / np.count_nonzero(union)
        return iou

    def count_covered_tips(sample_info: pd.Series) -> int:
        total_intersections = 0
        tips = MaskActions.load_mask(
            sample_info["out_file_template"].format(filename="mask_Intersections.png")
        )

        for i in range(1, 4):
            dart_mask = MaskActions.load_mask(
                sample_info["out_file_template"].format(filename=f"mask_Dart_{i}.png")
            )
            intersection = np.logical_and(tips, dart_mask)

            # Contours = dart cover count, but darts cover their own tips (0-1 contours expected)
            contours, _ = cv2.findContours(
                np.uint8(intersection) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if (n_intersections := len(contours)) > 1:
                total_intersections += n_intersections - 1

        return total_intersections


def prepare_sample(sample_info: pd.Series, debug: bool = False):

    # Load sample images
    img, img_orient, img_area, img_intersections = ImageUtils.load_sample_imgs(
        sample_info
    )

    # Extract orientation lines
    line_v, line_h, orientation_points = MaskActions.get_lines_from_point_masks(
        img_orient
    )
    sample_info["h_line_r"] = line_h[0] / img.shape[1]
    sample_info["h_line_theta"] = line_h[1]
    sample_info["v_line_r"] = line_v[0] / img.shape[0]
    sample_info["v_line_theta"] = line_v[1]
    sample_info["orientation_points"] = tuple(
        (round(o[0]), round(o[1])) for o in orientation_points
    )  # (t, r, b, l)

    # Extract ellipse
    ellipse = MaskActions.get_ellipse_from_mask(img_area)
    sample_info["ellipse_cx"] = round(ellipse[0][0])
    sample_info["ellipse_cy"] = round(ellipse[0][1])
    sample_info["ellipse_w"] = round(ellipse[1][0])
    sample_info["ellipse_h"] = round(ellipse[1][1])
    sample_info["ellipse_theta"] = ellipse[2]

    # Undistort image
    img_undist, undistortion_homography = ImageUtils.undistort(img, orientation_points)
    sample_info["undistortion_homography"] = undistortion_homography

    # Extract dart positions
    dart_positions = MaskActions.get_dart_positions(sample_info, img_intersections)
    sample_info["dart_positions"] = dart_positions

    # Extract undistorted dart positions
    img_intersections_undist = cv2.warpPerspective(
        img_intersections, undistortion_homography, img_undist.shape[:2][::-1]
    )
    dart_positions_undist = MaskActions.get_dart_positions(
        sample_info, img_intersections_undist
    )
    sample_info["dart_positions_undistort"] = dart_positions_undist

    # Calculate Darts IoU
    dart_iou = MaskActions.calculate_darts_iou(sample_info)
    sample_info["dart_iou"] = dart_iou

    # Check if dart tip is covered
    amuont_tips_covered = MaskActions.count_covered_tips(sample_info)
    sample_info["dart_tips_covered"] = amuont_tips_covered

    # Add Training Data utils
    cy, cx = extract_center(img)
    sample_info["cv_ellipse_cy"] = cy
    sample_info["cv_ellipse_cx"] = cx

    # Save results
    sample_info.sort_index(inplace=True)
    cv2.imwrite(
        sample_info.out_file_template.format(filename="undistort.png"), img_undist
    )
    with open(sample_info.out_file_template.format(filename="info_new.pkl"), "wb") as f:
        pickle.dump(sample_info, f)
    if not debug:
        return sample_info

    # -------------------------------------------
    # DEBUG CODE
    # draw some lines
    draw_polar_line(img, *line_h, color=(0, 255, 0))
    draw_polar_line(img, *line_v, color=(0, 255, 0))
    ellipse_draw = (
        (round(ellipse[0][0]), round(ellipse[0][1])),
        (round(ellipse[1][0] / 2), round(ellipse[1][1] / 2)),
        ellipse[2],
    )
    cv2.ellipse(img, *ellipse_draw, 0, 360, color=(255, 0, 0), thickness=1)

    for angle in range(4):
        c = 400
        margin = 100
        angle *= 90
        angle -= 360 / 40  # shift by half a field
        dx = c + (c - margin) * np.sin(np.deg2rad(angle))
        dy = c - (c - margin) * np.cos(np.deg2rad(angle))
        px = int(c + 2 * (dx - c))
        py = int(c + 2 * (dy - c))
        cv2.line(img_undist, (c, c), (px, py), (0, 255, 0), lineType=cv2.LINE_AA)
    cv2.line(img_undist, (c, 0), (c, 2 * c), (255, 255, 255), 1)
    cv2.line(img_undist, (0, c), (c * 2, c), (255, 255, 255), 1)
    cv2.circle(
        img_undist, (400, 400), radius=300, color=(255, 0, 0), lineType=cv2.LINE_AA
    )

    show_imgs(original=img, undistorted=img_undist)
    return sample_info


if __name__ == "__main__":
    data_dir = "data/generation/out_val"
    samples = [d for d in os.listdir(data_dir) if d.isnumeric()]
    samples = sorted(samples, key=int)

    def process_sample(id):
        info_path = os.path.join(data_dir, str(id), "info.pkl")
        print(info_path)
        if not os.path.exists(info_path):
            print("non-existing")
            return
        with open(info_path, "rb") as f:
            sample_info = pickle.load(f)
        sample_info["sample_id"] = id
        prepare_sample(sample_info)

    from multiprocessing import Pool

    with Pool(os.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_sample, samples), total=len(samples)))
