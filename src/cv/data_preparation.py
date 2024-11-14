import os
import cv2
import numpy as np
from rich import print

IMG_DIR = "data/generation/out"


def load_imgs(
    id: int = 0,
) -> tuple[
    np.ndarray,  # (y, x, 3)
    np.ndarray,  # (y, x, 3)
    np.ndarray,  # (y, x, 3)
]:
    id = f"{id:04d}"
    img = cv2.imread(os.path.join(IMG_DIR, id + ".png"))
    img_orient = cv2.imread(
        os.path.join(IMG_DIR, id + "_mask_Board_Orientation.png"), cv2.IMREAD_GRAYSCALE
    )
    img_area = cv2.imread(
        os.path.join(IMG_DIR, id + "_mask_Darts_Board_Area.png"), cv2.IMREAD_GRAYSCALE
    )
    return img, img_orient, img_area


img, img_orient, img_area = load_imgs(id=0)


class ImageUtils:

    def draw_polar_line(
        img: np.ndarray,  # (y, x)
        rho: float,
        theta: float,
        color: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 1,
    ) -> np.ndarray:  # (y, x)
        # Calculate the starting and ending points of the line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # Define two points on the line based on the line equation in Cartesian form
        pt1 = (int(x0 + 5000 * -b), int(y0 + 5000 * a))
        pt2 = (int(x0 - 5000 * -b), int(y0 - 5000 * a))

        # Draw the line on the image
        cv2.line(
            img,
            pt1=pt1,
            pt2=pt2,
            color=color,
            thickness=thickness,
        )
        return img

    def intersect_imgs(
        img_a: np.ndarray,  # (y, x)
        img_b: np.ndarray,  # (y, x)
    ) -> np.ndarray:  # (y, x)
        intersections = np.float32(img_a) * img_b
        intersections /= intersections.max()
        thresh = cv2.threshold(intersections, 0.5, 1, cv2.THRESH_BINARY)[1]
        return np.uint8(thresh * 255)

    def points_from_intersection(
        intersection_img: np.ndarray,  # (y, x)
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        ys, xs = np.nonzero(intersection_img)  # [3*y1 + 3*y2], [3*x1 + 3*x2]
        pos_a = (
            round(ys[:3].mean()),
            round(xs[:3].mean()),
        )  # top / left
        pos_b = (
            round(ys[3:].mean()),
            round(xs[3:].mean()),
        )  # bottom / right
        return pos_a, pos_b


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


def get_lines_from_point_masks(
    img: np.ndarray,  # (y, x)
) -> tuple[tuple[float, float], tuple[float, float]]:

    def get_moment_center(
        M: dict,
    ) -> tuple[float, float]:
        cy = M["m01"] / M["m00"]
        cx = M["m10"] / M["m00"]
        return round(cy), round(cx)

    def extract_centers(
        img: np.ndarray,  # (y, x)
    ) -> list[tuple[float, float]]:
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    return line_v, line_h


def get_ellipse_from_mask(
    img: np.ndarray,  # (y, x)
) -> tuple[tuple[int, int], tuple[int, int], float]:

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    points = np.column_stack(np.where(thresh.transpose() > 0))
    hull = cv2.convexHull(points)[:, 0]

    (cx, cy), (w, h), theta = cv2.fitEllipse(hull)

    return (round(cx), round(cy)), (round(w / 2), round(h / 2)), theta


def undistort(
    img: np.ndarray,  # (y, x, 3)
    ellipse: tuple[tuple[int, int], tuple[int, int], float],
    line_h: tuple[float, float],
    line_v: tuple[float, float],
) -> np.ndarray:  # (800, 800, 3)

    # Data as used in the paper
    dst_size = 800
    margin = 100

    def get_src_pts(
        ellipse: tuple[tuple[int, int], tuple[int, int], float],
        line_h: tuple[float, float],
        line_v: tuple[float, float],
    ) -> np.ndarray:  # (5, 2): top, left, bottom, right, center
        (ellipse_cx, ellipse_cy), (ellipse_w, ellipse_h), ellipse_theta = ellipse

        # Get res img
        img_w = ellipse_cx + int(max(ellipse_w, ellipse_h) * 1.5)
        img_h = ellipse_cy + int(max(ellipse_w, ellipse_h) * 1.5)
        img_ellipse = np.zeros((img_h, img_w), np.uint8)
        cv2.ellipse(img_ellipse, *ellipse, 0, 360, color=255, thickness=1)

        # horizontal points
        img_line_h = ImageUtils.draw_polar_line(
            np.zeros_like(img_ellipse), *line_h, thickness=2
        )
        h_intersections = ImageUtils.intersect_imgs(img_ellipse, img_line_h)
        left, right = ImageUtils.points_from_intersection(h_intersections)

        # vertical points
        img_line_v = ImageUtils.draw_polar_line(
            np.zeros_like(img_ellipse), *line_v, thickness=2
        )
        v_intersections = ImageUtils.intersect_imgs(img_ellipse, img_line_v)
        top, bot = ImageUtils.points_from_intersection(v_intersections)

        # center point
        c_intersection = ImageUtils.intersect_imgs(img_line_h, img_line_v)
        center = np.mean(np.nonzero(c_intersection), axis=1)
        center = (round(center[0]), round(center[1]))

        return np.array([top, left, bot, right, center], np.float32)

    def get_dst_points() -> np.ndarray:  # (5, 2): top, left, bottom, right, center
        c = dst_size / 2
        dst_pts = []

        for angle in range(4):
            angle *= 90
            angle -= 360 / 40  # shift by half a field
            px = c + (c - margin) * np.sin(np.deg2rad(angle))
            py = c - (c - margin) * np.cos(np.deg2rad(angle))
            dst_pts.append((py, px))

        dst_pts.append((c, c))

        return np.array(dst_pts, np.float32)  # t, l, b, r, c

    def transform_image(
        img: np.ndarray,  # (y, x, 3)
        src_pts: np.ndarray,  # (5, 2)
        dst_pts: np.ndarray,  # (5, 2)
    ) -> np.ndarray:  # (800, 800, 3)
        # we need to switch the element order from y, x to x, y because cv2 is special
        src_pts = src_pts[:, ::-1]
        dst_pts = dst_pts[:, ::-1]

        H, _ = cv2.findHomography(src_pts, dst_pts)
        img = cv2.warpPerspective(img, H, (dst_size, dst_size))

        return img

    # Get points
    src_pts = get_src_pts(ellipse, line_h, line_v)
    dst_pts = get_dst_points()

    img_undistorted = transform_image(img, src_pts, dst_pts)

    return img_undistorted


line_v, line_h = get_lines_from_point_masks(img_orient)
ellipse = get_ellipse_from_mask(img_area)
res = undistort(img, ellipse, line_h, line_v)

# draw some lines
ImageUtils.draw_polar_line(img, *line_h, color=(0, 255, 0))
ImageUtils.draw_polar_line(img, *line_v, color=(0, 255, 0))
cv2.ellipse(img, *ellipse, 0, 360, color=(255, 0, 0), thickness=1)

for angle in range(4):
    c = 400
    margin = 100
    angle *= 90
    angle -= 360 / 40  # shift by half a field
    dx = c + (c - margin) * np.sin(np.deg2rad(angle))
    dy = c - (c - margin) * np.cos(np.deg2rad(angle))
    px = int(c + 2 * (dx - c))
    py = int(c + 2 * (dy - c))
    cv2.line(res, (c, c), (px, py), (0, 255, 0), lineType=cv2.LINE_AA)
cv2.circle(res, (400, 400), radius=300, color=(255, 0, 0), lineType=cv2.LINE_AA)

cv2.imshow("original", img)
cv2.imshow("undistorted", res)
cv2.waitKey()
