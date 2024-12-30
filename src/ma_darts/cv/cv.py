import os
import cv2
import numpy as np
from scipy.signal import find_peaks

from ma_darts.cv.utils import draw_polar_line, show_imgs, draw_polar_line_through_point
from ma_darts.cv.utils import (
    rotation_matrix,
    translation_matrix,
    shearing_matrix,
    scaling_matrix,
    apply_matrix,
)

img_paths = [
    "dump/thomas.png",
    "data/darts_references/jess/001_0-0-1.jpg",
    "data/darts_references/jess/018_1-DB-DB.jpg",
    "data/darts_references/jess/022_2-2-18.jpg",
    "data/darts_references/jess/061_6-7-T4.jpg",
    "data/darts_references/jess/084_10-6-4.jpg",
    "data/darts_references/jess/129_19-2-6.jpg",
    "dump/test/double.png",
    # "data/generation/out/0/render.png",
    # "data/generation/out/6/render.png",
    # "data/generation/out/7/render.png",
    # "data/generation/out/8/render.png",
    # "dump/test/x_90.png",
    # "dump/test/x_67_5.png",
    # "dump/test/x_45.png",
    # "dump/test/x_22_5.png",
    # "dump/test/y_90.png",
    # "dump/test/y_67_5.png",
    # "dump/test/y_45.png",
    # "dump/test/y_22_5.png",
    "dump/test/0001.jpg",
    # "dump/test/0002.jpg",
    # "dump/test/0003.jpg",
    # "data/paper/imgs/d1_02_16_2020/IMG_2858.JPG",
    "dump/test/test_img.png",
    "dump/test/test.png",
    # "data/paper/imgs//d2_02_23_2021_3/DSC_0003.JPG",
]


class Utils:

    def load_img(filepath: str, show: bool = False) -> np.ndarray:
        if not os.path.exists(filepath):
            return None
        img = cv2.imread(filepath)
        if show:
            show_imgs(input=img, block=False)
        return img

    def downsample_img(img: np.ndarray) -> np.ndarray:
        while max(*img.shape[:2]) > 1600:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        return img

    def non_maximum_suppression(img):

        from scipy.ndimage import maximum_filter

        max_filtered = maximum_filter(img, size=7, mode="constant")
        suppressed = (img == max_filtered) * img
        return suppressed

    def points_to_polar_line(p1, p2):
        y2, x2 = p2
        y1, x1 = p1

        if x2 >= x1:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        dy = y2 - y1
        dx = x2 - x1
        theta = np.arctan2(-dx, dy)

        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        theta %= np.pi

        return rho, theta

    def point_point_dist(p1, p2):
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def get_sobel(k: int, theta: float = np.pi / 2):
        # Create coordinate grid
        x, y = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))

        # Rotate coordinates
        x_rot = x * np.cos(theta) + y * np.sin(theta)
        y_rot = -x * np.sin(theta) + y * np.cos(theta)

        # Sobel filter formula: x * exp(-x^2 - y^2)
        sobel = x_rot / ((x_rot**2 + y_rot**2) + 1e-5)

        # Normalize
        sobel -= sobel.mean()
        sobel /= np.sum(np.abs(sobel))

        return sobel

    def get_edge_filter(k: int, theta: float = np.pi / 2) -> np.ndarray:
        x, y = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
        x_rot = x * np.cos(theta) + y * np.sin(theta)
        filter = np.sign(x_rot)

        # Normalize
        filter -= filter.mean()
        filter /= np.sum(np.abs(filter))

        return filter

    def point_line_distance(y: int, x: int, rho: float, theta: float) -> float:
        """
        rho = x * cos(theta) + y * sin(theta)
        => x * cos(theta) + y * sin(theta) - rho = 0
        => a = cos(theta), b = sin(theta), c = -rho
        => ax + by + c = 0
        -> https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        # sqrt(a^2 + b^2) = sqrt(sin^2 + cos^2) = 1
        => dist = | cos(theta) * x0 + sin(theta) * y0 - rho |
        """
        dist = abs(np.cos(theta) * x + np.sin(theta) * y - rho)
        return dist

    def point_theta_to_polar_line(
        pt: tuple[int, int], theta: float
    ) -> tuple[float, float]:
        y, x = pt
        rho = x * np.cos(theta) + y * np.sin(theta)
        return rho, theta

    def polar_line_intersection(rho_a, theta_a, rho_b, theta_b):
        sin_ta = np.sin(theta_a)
        sin_tb = np.sin(theta_b)
        cos_ta = np.cos(theta_a)
        cos_tb = np.cos(theta_b)

        det = cos_ta * sin_tb - sin_ta * cos_tb

        # No intersection
        if abs(det) < 1e-10:
            return (0, 0)

        y = (rho_b * cos_ta - rho_a * cos_tb) / det
        x = (rho_a * sin_tb - rho_b * sin_ta) / det

        return y, x

    def create_combined_img(
        imgs,
        target_w: int = 1440,
        target_h: int = 2560 - 125,
        failed: bool = False,
    ):
        if imgs is None:
            return

        failed = failed or any(["fail" in l.lower() for l, _ in imgs])

        bg_color = 50 if not failed else 10

        # Convert to color
        imgs = [
            (label, cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) if len(i.shape) == 2 else i)
            for label, i in imgs
        ]

        # Set image sizes
        n_imgs = len(imgs)
        grid_cols = int(np.ceil(np.sqrt(n_imgs)))
        grid_rows = int(np.ceil(n_imgs / grid_cols))

        cell_w = target_w // grid_cols
        cell_h = target_h // grid_rows

        resized_imgs = []
        for label, img in imgs:
            h, w = img.shape[:2]
            scale = min(cell_w / w, cell_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_img = cv2.resize(
                img, (new_w - 2, new_h - 2), interpolation=cv2.INTER_AREA
            )

            # Calculate padding to center the image
            pad_top = (cell_h - new_h) // 2
            pad_bottom = cell_h - new_h - pad_top
            pad_left = (cell_w - new_w) // 2
            pad_right = cell_w - new_w - pad_left

            # Apply padding
            padded_img = np.pad(
                resized_img,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=bg_color,
            )

            # Add title
            txt_params = dict(
                org=(5, 20),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                padded_img,
                label,
                color=(0, 0, 0),
                thickness=2,
                **txt_params,
            )
            cv2.putText(
                padded_img,
                label,
                color=(255, 255, 255),
                thickness=1,
                **txt_params,
            )

            # Add border
            padded_img = np.pad(
                padded_img,
                ((1, 1), (1, 1), (0, 0)),
                mode="constant",
                constant_values=bg_color,
            )
            resized_imgs.append(padded_img)

        # Combine into image
        res = np.full((grid_rows * cell_h, grid_cols * cell_w, 3), bg_color, np.uint8)

        for i, img in enumerate(resized_imgs):
            row, col = divmod(i, grid_cols)
            x0 = col * cell_w
            x1 = x0 + cell_w
            y0 = row * cell_h
            y1 = y0 + cell_h
            res[y0:y1, x0:x1] = img
        return res


class Edges:

    def edge_detect(
        img: np.ndarray, kernel_size: int = 5, show: bool = False
    ) -> np.ndarray:

        # Convert img to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Increase contrast
        img = cv2.convertScaleAbs(img, alpha=1.5)

        # Blur image
        img = cv2.blur(img, (2 * kernel_size + 1, 2 * kernel_size + 1))
        img = np.float32(img) / 255

        # Find edges
        filter_x = Utils.get_sobel(7)
        sobel_x = cv2.filter2D(img, -1, filter_x)
        sobel_y = cv2.filter2D(img, -1, filter_x.T)

        # combine gradients
        sobel_img = cv2.magnitude(sobel_x, sobel_y)
        sobel_edges = np.uint8(sobel_img / sobel_img.max() * 255)

        _, edges = cv2.threshold(sobel_edges, 100, 255, cv2.THRESH_BINARY)

        # show_imgs(img=img, sobel_x=sobel_x, sobel_y=sobel_y, sobel_edges=sobel_edges, edges=edges, block=False)
        if show:
            show_imgs(edges=edges, block=False)

        global combined_img
        if combined_img:
            combined_img.append(("Edge Detection", edges))
        return edges

    def skeleton(img: np.ndarray, show: bool = False) -> np.ndarray:
        skeleton = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        done = False
        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()

            done = cv2.countNonZero(img) == 0

        if show:
            show_imgs(skeleton=skeleton, block=False)
        global combined_img
        if combined_img:
            combined_img.append(("Skeletonized Edges", skeleton))
        return skeleton


class Lines:

    def extract_lines(
        img: np.ndarray,
        rho: int = 1,
        theta: float = np.pi / 180 / 10,
        threshold: int = 25,
        show: bool = False,
    ) -> list[tuple[float, float, float, float, float]]:

        # Dilate to make lines thicker
        dilation_size = 2
        img = cv2.dilate(
            img,
            kernel=np.ones((dilation_size, dilation_size), np.uint8),
        )

        # Find lines as points
        lines = cv2.HoughLinesP(
            img,
            rho=rho,
            theta=theta,
            threshold=threshold,
        )[:, 0]
        res = img * 0

        # Add lengths to lines
        lines = map(
            lambda x: (
                (x[1], x[0]),  # p1
                (x[3], x[2]),  # p2
                np.sqrt((x[3] - x[1]) ** 2 + (x[2] - x[0]) ** 2),  # length
            ),
            lines,
        )
        # Remove small lines
        lines = filter(lambda x: x[-1] > 5, lines)
        # sort by length
        lines = sorted(lines, key=lambda x: x[-1], reverse=True)
        # Normalize lengths
        max_len = max(x[-1] for x in lines)
        lines = map(lambda x: (x[0], x[1], x[2] / max_len), lines)
        # Add polar line representation
        lines = map(
            lambda x: (
                x[0],  # p1
                x[1],  # p2
                x[2],  # length
                *Utils.points_to_polar_line(x[0], x[1]),  # rho [-n..n], theta [0..π)
            ),
            lines,
        )
        lines = list(lines)

        global combined_img
        if show or combined_img:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            line_img = np.zeros_like(img)
            for p1, p2, length, rho, theta in lines:
                p1 = (int(p1[1]), int(p1[0]))
                p2 = (int(p2[1]), int(p2[0]))
                cv2.line(
                    line_img,
                    p1,
                    p2,
                    tuple(np.random.randint(256) for _ in range(3)),
                    1,
                    lineType=cv2.LINE_AA,
                )
                color = np.random.randint(128) + 128
                # cv2.circle(img, p1, 4, (color, 0, 0), lineType=cv2.LINE_AA)
                # cv2.circle(img, p2, 4, (color, 0, 0), lineType=cv2.LINE_AA)
            out = cv2.addWeighted(img, 0.25, line_img, 1, 1.0)
            if show:
                show_imgs(lines=out, block=False)
            if combined_img:
                combined_img.append(("Found Lines", out))

        return lines  # (p1, p2, length, rho, theta)

    def bin_lines_by_angle(
        lines: list[tuple[float, float, float, float, float]],
        n_bins: int = 10,
        angle_offset: float = 0,
    ) -> list[list[tuple[float, float, float, float, float]]]:
        # Extract thetas
        thetas = [l[-1] for l in lines]
        # Get bin indices
        bin_angles = np.arange(0, np.pi + np.pi / n_bins, np.pi / n_bins) + np.deg2rad(
            angle_offset
        )
        bin_indices = np.digitize(thetas, bin_angles, right=False)
        # Sort lines into bins
        lines_binned = [[] for _ in range(n_bins)]
        for i, bin_idx in enumerate(bin_indices):
            lines_binned[bin_idx - 1].append(lines[i])
        return lines_binned

    def get_center_point(
        img_shape: tuple[int, int],
        lines_binned: list[list[tuple[float, float, float, float, float]]],
        show: bool = False,
    ) -> tuple[int, int]:

        # Create one image for each bin
        bin_imgs = []
        # Iterate over bins
        for i, bin_lines in enumerate(lines_binned):
            bin_img = np.zeros((img_shape[0], img_shape[1]), np.uint8)
            # Add all lines onto the bin image
            for line in bin_lines:
                draw_polar_line(bin_img, *line[-2:], color=(1, 1, 1))
            bin_imgs.append(bin_img)

        # Accumulate bin lines
        acc = np.sum(bin_imgs, axis=0)
        # Blur and discretize
        acc = np.float32(acc) / acc.max()
        acc = cv2.blur(acc, (3, 3))
        acc = np.uint8(acc * 20)

        # Get center point(s)
        cy, cx = np.nonzero(acc == acc.max())

        # We take the average in case there are multiple maximum points
        cy = round(np.mean(cy))
        cx = round(np.mean(cx))

        global combined_img
        if show or combined_img:
            acc = np.uint8(np.float32(acc) / acc.max() * 255)
            acc = cv2.cvtColor(acc, cv2.COLOR_GRAY2BGR)
            cv2.circle(acc, (cx, cy), 10, (255, 0, 0), lineType=cv2.LINE_AA)
            if show:
                show_imgs(center_point=acc, block=False)
            if combined_img:
                combined_img.append(("Center Point", acc))
        return cy, cx

    def filter_lines_by_center_dist(
        lines: list[tuple[float, float, float, float, float]],
        cy: int,
        cx: int,
        max_center_dist: float = 10,
    ) -> list[tuple[float, float, float, float, float]]:

        lines_filtered = []
        for line_idx, line in enumerate(lines):
            rho, theta = line[-2:]
            dist = Utils.point_line_distance(cy, cx, rho, theta)

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

    def get_rough_line_angles(
        img_shape: tuple[int, int],
        lines: list[
            tuple[float, float, float, float, float]
        ],  # p1, p2, length (normalized), center distance [px], rho, theta
        cy: int,
        cx: int,
        show: bool = False,
    ):
        # Draw lines onto black canvas
        line_img = np.zeros(img_shape, np.uint8)
        for line in lines:
            cv2.line(
                line_img, line[0][::-1], line[1][::-1], (255, 255, 255), thickness=2
            )

        # Line positions to origin
        y_indices, x_indices = np.nonzero(line_img)
        y_shifted, x_shifted = y_indices - cy, x_indices - cx

        # Weigh each points by its distance to the center
        diag = np.sqrt(img_shape[0] ** 2 + img_shape[1] ** 2)
        center_dist = np.sqrt(x_shifted**2 + y_shifted**2)
        center_dist_norm = center_dist / diag
        weights = 1 / (center_dist_norm + 1)

        # For each point, get line angle
        angles = np.arctan2(y_shifted, x_shifted)  # -pi, pi
        angles = (angles + np.pi) % np.pi

        # Quantize into bins
        angle_step = 0.5
        bins = np.arange(0, np.pi, np.deg2rad(angle_step))
        bin_indices = np.digitize(angles, bins) - 1

        # Accumulate bin sizes
        accumulator = np.zeros(len(bins), np.float32)
        for bin_idx, weight in zip(bin_indices, weights):
            accumulator[bin_idx] += weight

        # Smooth accumulated sizes
        smooth_degrees = 5
        window_size = int(np.ceil(smooth_degrees / angle_step))
        kernel = np.ones(window_size) / window_size
        acc_extended = np.concatenate(
            [accumulator[-window_size:], accumulator, accumulator[:window_size]]
        )

        # Double-smoothing to prevent mini-peaks
        acc_extended_smooth = np.convolve(acc_extended, kernel, mode="same")
        acc_extended_smooth = np.convolve(acc_extended_smooth, kernel, mode="same")

        # Find peaks
        peaks_extended, _ = find_peaks(acc_extended_smooth)

        # Normalize peaks to non-extended indices
        peaks = peaks_extended - window_size
        peaks = peaks[peaks >= 0]
        peaks = peaks[peaks < len(bins)]

        # Get peak values
        acc_smooth = acc_extended_smooth[window_size:-window_size]
        peak_values = acc_smooth[peaks]
        peak_thetas = np.deg2rad(angle_step) * peaks
        peak_thetas = (peak_thetas + np.pi / 2) % np.pi

        if len(peak_values) > 10:
            # Remove smallest peaks
            peak_cutoff = sorted(peak_values)[-10]
            cutoff_indices = peak_values >= peak_cutoff
            peaks = peaks[cutoff_indices]
            peak_values = peak_values[cutoff_indices]
            peak_thetas = peak_thetas[cutoff_indices]
        elif len(peaks) < 10:
            # Interpolate peaks
            step_size = int(np.median(np.diff(peaks)))

            extended = []
            for i, peak in enumerate(peaks[:-1]):
                extended.append(peaks[i])
                gap = peaks[i + 1] - peak
                if gap > step_size * 1.5:
                    n_missing = round(gap / step_size) - 1
                    missing_values = [
                        peak + step_size * (j + 1) for j in range(n_missing)
                    ]

                    extended.extend(missing_values)
            extended.append(peaks[-1])

            # TODO: interpolate peak_thetas and peak_values for added values

        global combined_img
        if show or combined_img:
            res = img // 8
            for line in lines:
                cv2.line(
                    res,
                    line[0][::-1],
                    line[1][::-1],
                    color=(0, 255, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
            for t, v in zip(peak_thetas, peak_values / peak_values.max()):
                intensity = v / 2 + 0.5
                draw_polar_line_through_point(
                    res,
                    (cy, cx),
                    t,
                    color=(255, 0, 0),
                    intensity=intensity,
                )

            # with open("dump/out.txt", "w") as f:
            #     for i, (v, vs) in enumerate(zip(accumulator, acc_smooth)):
            #         common = min(v, vs)
            #         diff = max(v, vs) - common
            #         delimiter = "+" if vs > v else "."

            #         main_bar = "#" if i in peaks else ">"
            #         print(
            #             i,
            #             "\t",
            #             main_bar * int(common),
            #             delimiter * int(diff),
            #             sep="",
            #             file=f,
            #         )
            if show:
                show_imgs(field_separating_lines=res, block=False)
            if combined_img:
                combined_img.append(("Field-Separating Lines", res))

        thetas = sorted(peak_thetas)
        return thetas


class Orientation:

    def align_angles(
        lines_filtered: list[tuple[float, float, float, float, float]],
        thetas: list[float],
        img_shape: tuple[int, int],
        show: bool = False,
    ) -> list[tuple[float, float]]:
        rho_guess = np.sqrt(img_shape[0] ** 2 + img_shape[1] ** 2) / 2

        def fit_polar_line_to_points(
            points, weights, initial_theta
        ) -> tuple[float, float]:
            def objective(params):
                rho, theta = params
                res = (
                    sum(
                        w * Utils.point_line_distance(*pt, rho, theta) ** 2
                        for pt, w in zip(points, weights)
                    )
                    + 1e-5
                )
                return res

            initial_guess = Utils.point_theta_to_polar_line((cy, cx), theta)

            # for p in points:
            #     print(Utils.point_line_distance(*p, *initial_guess))
            #     img[p[0], p[1]] = 255

            # draw_polar_line(img, *initial_guess)
            # show_imgs(img)
            # return initial_guess

            bounds = [(-rho_guess * 2, rho_guess * 2), (0, np.pi)]

            from scipy.optimize import minimize

            result = minimize(
                objective,
                initial_guess,
                bounds=bounds,
                method="L-BFGS-B",
                options={"gtol": 1e-5, "ftol": 1e-6},
            )

            if result.success:
                return result.x  # rho, theta

            print("WARNING: Could not terminate line fitting:", result.message)
            return result.x

        # Filter lines
        line_bins = [[] for _ in range(len(thetas))]
        for line in lines_filtered:
            min_dist = np.Inf
            for i, theta in enumerate(thetas):
                dist_1 = abs(theta - line[-1])
                dist_2 = abs(theta - (line[-1] - np.pi))
                dist = min(dist_1, dist_2)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            line_bins[min_idx].append(line)

        out_lines = []
        for theta, lines in zip(thetas, line_bins):
            if len(lines) == 0:
                out_lines.append(Utils.point_theta_to_polar_line((cy, cx), theta))
                continue

            points = [(cy, cx)]
            for line in lines:
                points.append(line[0])
                points.append(line[1])

            lengths = [line[2] for line in lines]
            mean_points = [
                (
                    (line[0][0] + line[1][0]) // 2,
                    (line[0][1] + line[1][1]) // 2,
                )
                for line in lines
            ]
            weights = [
                l / (Utils.point_point_dist((cy, cx), mp) + 1e-2)
                for l, mp in zip(lengths, mean_points)
            ]
            # weight for each point of a line
            weights = [w for w in weights for _ in (0, 1)]
            weights.insert(0, max(weights) * 2)  # weight for center point
            # normalize
            weights = [w - min(weights) for w in weights]
            weights = [w / max(weights) for w in weights]

            # print(points, weights, theta)
            rho_, theta_ = fit_polar_line_to_points(points, weights, theta)
            # print("--")
            # print(theta, theta_)
            out_lines.append((rho_, theta_))

        global combined_img
        if show or combined_img:
            out = img.copy()
            for rho, theta in out_lines:
                draw_polar_line(out, rho, theta)
            if show:
                show_imgs(lines_aligned=out, block=False)
            if combined_img:
                combined_img.append(("Aligned Angles", out))
        return out_lines

    def center_point_from_lines(
        lines: list[tuple[float, float]],
    ) -> tuple[float, float]:
        ys = []
        xs = []
        for i, line_a in enumerate(lines):
            for line_b in lines[i + 1 :]:
                y, x = Utils.polar_line_intersection(*line_a, *line_b)
                ys.append(y)
                xs.append(x)
        cy = np.mean(ys)
        cx = np.mean(xs)
        return cy, cx

    def undistort_by_lines(
        cy: int,
        cx: int,
        lines: list[tuple[float, float]],
        show: bool = False,
    ):

        def theta_change_shear_y(theta, shear_y):
            """
            slope = dy / dx
            mapping:
                dy/dx -> dy / (dy * s + dx)
                        = (dy / dx) / (1 + s * (dy / dx))
                        = slope / (1 + s * slope)
            in out left-handed coordinate system, the shearing is inverted, so we use:
                dy/dx -> slope / (1 - s * slope)
            mapping not as straightforward as we change the denominator
            """
            slope = np.tan(theta)  # get the slope

            slope_ = slope / (1 - shear_y * slope)  # this is the funny mapping

            theta_ = np.arctan(slope_)  # convert slope to angle
            return theta_

        def visualize_matrix(M):
            nonlocal src_start, src
            global cy, cx, img
            res = img.copy()
            for t in src_start:
                draw_polar_line_through_point(
                    res, (int(cy), int(cx)), t, thickness=3, color=(255, 255, 255)
                )

            res = apply_matrix(res, M)

            for t in dst:
                draw_polar_line_through_point(
                    res, (int(cy), int(cx)), t, color=(255, 0, 0), thickness=2
                )
            for t in src:
                draw_polar_line_through_point(
                    res, (int(cy), int(cx)), t, color=(0, 0, 0), thickness=1
                )
            cv2.putText(
                res,
                "white=transformed initial thetas, blue='dst' thetas, black='src' thetas",
                org=(0, 20),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(255, 255, 255),
                thickness=1,
            )
            return res

        angle_step = np.pi / 10
        dst_start = np.arange(0, np.pi, angle_step) + angle_step / 2
        src_start = np.array([l[1] for l in lines])

        transformation_matrices = []

        for start_line in range(10):

            # Initialize Values
            M = np.eye(3)
            src = src_start.copy()
            dst = dst_start.copy()

            # -----------------------------
            # 1. Translate center to origin
            M_trans_a = translation_matrix(-cx, -cy)
            M_trans_b = translation_matrix(cx, cy)

            # update transformation matrix
            M = M_trans_a @ M

            # -----------------------------
            # 2. Align src vertival line with destination vertical line
            # output: src[i] -> dst[i]
            t_src = src[start_line]
            t_dst = dst[start_line]
            rot_angle = t_dst - t_src
            M_rot = rotation_matrix(rot_angle)

            # update transformation matrix
            M = M_rot @ M

            # update src points
            src += rot_angle

            # draw
            if show:
                res = visualize_matrix(M_trans_b @ M)
                show_imgs(first_rotate_to_align_single_line=res, block=False)

            # -----------------------------
            # 3. Vertical alignment
            # Goal: src[i] = dst[0] = 0
            rotation_alignment_angle = angle_step / 2 + start_line * angle_step
            M_rot_v = rotation_matrix(-rotation_alignment_angle)

            # update transformation matrix
            M = M_rot_v @ M

            # update src and dst points
            src -= rotation_alignment_angle
            dst -= rotation_alignment_angle

            # draw
            if show:
                res = visualize_matrix(M_trans_b @ M)
                show_imgs(second_rotate_aligned_to_vertical=res, block=False)

            # -----------------------------
            # 4. Vertical shearing
            # Goal:
            #   - src[i] = 0
            #   - src[i+5] = 90° = np.pi / 2
            horizontal_line_idx = (start_line + 5) % len(lines)
            t_src = src[horizontal_line_idx]
            t_dst = dst[horizontal_line_idx]
            shear_amount = t_dst - t_src

            M_shear = shearing_matrix(shear_amount)

            # update transformation matrix
            M = M_shear @ M

            # update src points
            src = theta_change_shear_y(src, shear_amount)
            src[src < 0] += np.pi

            # draw
            if show:
                res = visualize_matrix(M_trans_b @ M)
                show_imgs(third_shear_to_fit_horizontal=res, block=False)

            # -----------------------------
            # 5. Vertical scaling
            # Goal: align rest of lines as good as possible

            # convert angles to slopes
            slopes_src = np.tan(src)
            slopes_dst = np.tan(dst)
            # remove already aligned angles
            slopes_src = np.delete(slopes_src, [start_line, horizontal_line_idx])
            slopes_dst = np.delete(slopes_dst, [start_line, horizontal_line_idx])
            # calculate scaling
            scales = slopes_src / slopes_dst
            scale = np.mean(scales)

            M_scale = scaling_matrix(y=scale)

            # update transformation matrix
            M = M_scale @ M

            # upate src points
            src = np.arctan(np.tan(src) / scale)

            # draw
            if show:
                res = visualize_matrix(M_trans_b @ M)
                show_imgs(fourth_scale_to_fit_rest_of_lines=res, block=False)

            # -----------------------------
            # 6. Undo vertical alignment
            M_rot_v_inv = rotation_matrix(rotation_alignment_angle)

            # update transformation matrix
            M = M_rot_v_inv @ M

            # update src and dst points
            src += rotation_alignment_angle
            dst += rotation_alignment_angle

            # draw
            if show:
                res = visualize_matrix(M_trans_b @ M)
                show_imgs(vertical_scaling=res, block=False)

            # -----------------------------
            # 7. Re-Translate into center

            M = M_trans_b @ M

            transformation_matrices.append(M)

        # - # - # - # - # - # - # - # - # - #

        M = np.mean(transformation_matrices, axis=0)

        global combined_img
        if show or combined_img:
            global img
            res = apply_matrix(img, M, True)
            if show:
                show_imgs(img_undistort=res, block=False)
            if combined_img:
                combined_img.append(("Undistorted Angles", res))

        return M

    def find_orientation_points(
        img: np.ndarray,
        cy: int,
        cx: int,
        show: bool = False,
    ) -> list[list[tuple[int, str]]]:

        # Convert image to logpolar representation
        max_r = max(
            Utils.point_point_dist((cy, cx), (0, 0)),
            Utils.point_point_dist((cy, cx), (0, img.shape[1])),
            Utils.point_point_dist((cy, cx), (img.shape[0], 0)),
            Utils.point_point_dist((cy, cx), (img.shape[0], img.shape[1])),
        )
        logpolar = cv2.warpPolar(
            img,
            dsize=(
                int(max_r),
                1000,
            ),  # I use 1000 because this is enough resolution and is easy to calculate with
            center=(int(cx), int(cy)),
            maxRadius=int(max_r),
            flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS,
        )
        if show:
            show_imgs(logpolar=logpolar, block=False)

        # -----------------------------
        # Find corners
        def find_corners(img):
            # img = cv2.blur(img, ksize=(7, 7))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = cv2.GaussianBlur(img, (5, 5), 0)

            block_size = 2
            kernel_size = 3
            k = 0.04

            corners = cv2.cornerHarris(
                img,
                block_size,
                kernel_size,
                k,
            )
            threshold = 0.01 * corners.max()

            corners = cv2.threshold(
                corners, threshold, corners.max(), cv2.THRESH_BINARY
            )[1]
            corners /= corners.max()
            corners = cv2.dilate(corners, None)

            # corners = np.abs(corners)
            # corners /= corners.max()
            # corners = np.uint8(corners * 255)
            # # Threshold
            # corners = cv2.threshold(corners, 100, 255, cv2.THRESH_BINARY)[1]

            return corners

        corners = find_corners(logpolar)  # (1000, x)

        # -----------------------------
        # Find corners on intersection lines
        corner_band_width = 6

        line_ys = (
            np.arange(0, corners.shape[0], corners.shape[0] // 20)
            + corners.shape[0] // 40
        )
        corner_strips = [
            corners[i - corner_band_width : i + corner_band_width] for i in line_ys
        ]
        corner_strip_values = [np.max(s, axis=0) for s in corner_strips]

        # Find positions of corners on strips
        corner_positions = []
        for strip in corner_strip_values:
            nonzeros = np.nonzero(strip)[0]
            if len(nonzeros) == 0:
                corner_positions.append([])
                continue
            diffs = np.diff(nonzeros)
            breaks = np.where(diffs > 1)[0] + 1
            groups = np.split(nonzeros, breaks)
            centers = [round(np.mean(g)) for g in groups]
            corner_positions.append(centers)

        # Visualize corner strips
        if show:
            corners_ = corners.copy()
            for i in line_ys:
                corners_[i - corner_band_width : i + corner_band_width][
                    corners_[i - corner_band_width : i + corner_band_width] == 0
                ] = 60
            show_imgs(corner_strips=corners_, block=False)

        search_distance = 2
        search_width = 8

        # -----------------------------
        # Get Colors
        def sort_logpolar_into_strips(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

            # Slice logpolar aling fields
            splits = np.arange(0, img.shape[0], img.shape[0] // 20) + img.shape[0] // 40
            slices = np.split(img, splits)
            white = slices[::2]
            black = slices[1::2]

            # Combine cut-off strip
            white[0] = np.concatenate([white[-1], white[0]], axis=0)
            white.pop(-1)

            # Combine white and black strips to single images
            white = np.vstack(white)
            black = np.vstack(black)

            return white, black

        def get_colors(logpolar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            strip_w, strip_b = sort_logpolar_into_strips(logpolar)
            white_colors = np.median(strip_w, axis=0).astype(np.uint8)
            black_colors = np.median(strip_b, axis=0).astype(np.uint8)

            # Skip bullseye
            diff = np.abs(np.int16(white_colors) - black_colors).astype(np.uint8)
            diff = cv2.convertScaleAbs(diff, alpha=2)
            diff = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)[1]
            field_start = np.nonzero(diff)[0][0] if len(np.nonzero(diff)[0]) > 0 else 0

            # Get white color
            def extract_field_color(
                colors: np.ndarray, field_start: int
            ) -> tuple[np.ndarray, int]:
                rough_jump = 10
                color_different_threshold = 60
                # Start off with a general color guess, based on a somewhat arbitrary area
                general_color = (
                    colors[field_start : field_start + rough_jump].mean(axis=0)
                ).astype(np.int16)

                # Look along the image
                # until we see a color that differs enough from the general color
                field_end = field_start + rough_jump
                while (
                    field_end < len(colors) - 1
                    and np.abs(general_color - colors[field_end]).sum()
                    < color_different_threshold
                ):
                    field_end += 1

                # Calculate field color based on found area
                field_color = np.median(colors[field_start:field_end], axis=0).astype(
                    np.uint8
                )
                return field_color, field_end

            white, field_end_w = extract_field_color(white_colors, field_start)
            black, field_end_b = extract_field_color(black_colors, field_start)

            return white, black

        white, black = get_colors(logpolar)

        # -----------------------------
        # Check points for surrounding color
        surrounding_width = 14
        middle_deadspace = 2
        color_threshold = 30
        intrude = (surrounding_width - middle_deadspace) // 2
        inner_ring_a = []
        inner_ring_b = []
        outer_ring_a = []
        outer_ring_b = []
        for i, points in enumerate(corner_positions):
            y = 25 + 50 * i
            for x in points:
                surrounding = logpolar[
                    y - surrounding_width : y + surrounding_width,
                    x - surrounding_width : x + surrounding_width,
                ]
                # Find partial fields in surrounding area
                top_left = surrounding[:intrude, :intrude]
                top_right = surrounding[:intrude, -intrude:]
                bottom_left = surrounding[-intrude:, :intrude]
                bottom_right = surrounding[-intrude:, -intrude:]
                # Extract mean colors from fields
                color_top_left = top_left.mean(axis=0).mean(axis=0)
                color_top_right = top_right.mean(axis=0).mean(axis=0)
                color_bottom_left = bottom_left.mean(axis=0).mean(axis=0)
                color_bottom_right = bottom_right.mean(axis=0).mean(axis=0)
                # Determine if fields are black or white
                top_left_white = abs((color_top_left - white).mean()) < color_threshold  # fmt: skip
                top_left_black = abs((color_top_left - black).mean()) < color_threshold  # fmt: skip
                top_right_white = abs((color_top_right - white).mean()) < color_threshold  # fmt: skip
                top_right_black = abs((color_top_right - black).mean()) < color_threshold  # fmt: skip
                bottom_left_white = abs((color_bottom_left - white).mean()) < color_threshold  # fmt: skip
                bottom_left_black = abs((color_bottom_left - black).mean()) < color_threshold  # fmt: skip
                bottom_right_white = abs((color_bottom_right - white).mean()) < color_threshold  # fmt: skip
                bottom_right_black = abs((color_bottom_right - black).mean()) < color_threshold  # fmt: skip
                # Sort color cases
                if top_left_black and bottom_left_white:
                    surrounding_normalized = surrounding
                    inner_ring_a.append((y, x, surrounding_normalized))

                if top_left_white and bottom_left_black:
                    surrounding_normalized = surrounding[::-1]  # flip vertical
                    inner_ring_b.append((y, x, surrounding_normalized))

                if top_right_black and bottom_right_white:
                    surrounding_normalized = surrounding[:, ::-1]  # flip horizontal
                    outer_ring_a.append((y, x, surrounding_normalized))

                if top_right_white and bottom_right_black:
                    surrounding_normalized = surrounding[::-1, ::-1]  # flip both
                    outer_ring_b.append((y, x, surrounding_normalized))

        # -----------------------------
        # Get common surrounding
        surroundings = (
            [x[2] for x in inner_ring_a]
            + [x[2] for x in inner_ring_b]
            + [x[2] for x in outer_ring_a]
            + [x[2] for x in outer_ring_b]
        )
        global combined_img
        if len(outer_ring_a + outer_ring_b) < 2:
            print(
                "ERROR: Not enough valid orientation points found (possibly too many outliers)."
            )
            if combined_img:
                logpolar[corners != 0] = 255
                combined_img.append(("FAILED: Not enough orientation points", logpolar))
            return None
        if len(surroundings) == 0:
            print("ERROR: No valid surroundings found.")
            if combined_img:
                logpolar[corners != 0] = 255
                combined_img.append(
                    ("FAILED: no orientation point surroundings found", logpolar)
                )
            return None
        mean_surrounding = np.median(surroundings, axis=0).astype(np.uint8)
        if show:
            show_imgs(mean_surrounding=mean_surrounding, block=False)

        # -----------------------------
        # Classify surroundings
        def is_correct_surrounding(
            mean_surrounding: np.ndarray, surrounding: np.ndarray, show: bool = False
        ) -> tuple[bool, np.ndarray]:
            similarity_threshold = 0.5

            # -------------------------
            # SSIM
            def ssim_score(patch_1, patch_2):
                from skimage.metrics import structural_similarity as ssim

                patch_1 = cv2.cvtColor(patch_1, cv2.COLOR_BGR2HSV)
                patch_2 = cv2.cvtColor(patch_2, cv2.COLOR_BGR2HSV)

                similarity = ssim(patch_1, patch_2, multichannel=True, channel_axis=2)

                return np.clip(similarity, 0, 1)

            # -------------------------
            def lab_areas(patch_1, patch_2):
                # Use LAB color space to emphasize typical red and green colors
                lab_1 = cv2.cvtColor(patch_1, cv2.COLOR_BGR2LAB)
                lab_2 = cv2.cvtColor(patch_2, cv2.COLOR_BGR2LAB)

                mid = lab_1.shape[0] // 2
                # Only look at red and green parts
                red_1 = lab_1[:mid, mid:]
                green_1 = lab_1[mid:, mid:]

                red_2 = lab_2[:mid, mid:]
                green_2 = lab_2[mid:, mid:]

                # Calculate patch differences
                diff_red = np.abs(np.int16(red_2) - red_1)
                sum_red = diff_red.sum(axis=-1)
                mean_red = sum_red.mean()

                diff_green = np.abs(np.int16(green_2) - green_1)
                sum_green = diff_green.sum(axis=-1)
                mean_green = sum_green.mean()

                mean_total = (mean_red + mean_green) / 2

                # Calculate similarity using exponential falloff
                #   - high input = low similarity
                #   - higher falloff value = stricter similarity
                #   - falloff = -0.008: 50% similarity at 86
                falloff = 0.01
                similarity = np.exp(-falloff * mean_total)
                return similarity

            # Compare colors in LAB color space
            similarity_lab = lab_areas(mean_surrounding, surrounding)
            # User SSIM for structural similarity
            similarity_ssim = ssim_score(mean_surrounding, surrounding)

            similarity = (similarity_lab + similarity_ssim) / 2
            is_orientation_point = similarity > similarity_threshold

            # Draw results
            if show:
                surrounding = surrounding.copy()
                c = (0, 255, 0) if is_orientation_point else (0, 0, 255)
                surrounding[:2] = c
                surrounding[:, :2] = c
                surrounding[:, -2:] = c
                surrounding[0] = (0, 0, 0)
                surrounding[:, 0] = (0, 0, 0)
                surrounding[:, -1] = (0, 0, 0)

                # similarity indication
                end = int(surrounding.shape[1] * similarity)
                end = np.clip(end, 0, surrounding.shape[1])
                surrounding[-2:, :end] = (255, 255, 255)
                surrounding[-2:, end:] = (0, 0, 0)
                surrounding[-1, int(similarity_threshold * surrounding.shape[1])] = (
                    255,
                    0,
                    0,
                )
            return is_orientation_point, surrounding

        prepare_show_img = show or combined_img
        if prepare_show_img:
            logpolar_ = logpolar.copy()

        keeps: list[tuple[int, int, str]] = []
        for y, x, surrounding in inner_ring_a:
            # Check if valid surrounding
            is_orientation_point, surrounding_ = is_correct_surrounding(
                mean_surrounding, surrounding, show=prepare_show_img
            )
            if is_orientation_point:
                keeps.append((y, x, "inner"))

            if show:
                # Draw point visualization
                cv2.circle(logpolar_, (x, y), 5, (255, 255, 255))
                cv2.circle(logpolar_, (x, y), 2, (0, 0, 255), -1)
                # Place surrounding on image: top left
                if y - surrounding.shape[0] >= 0 and x - surrounding.shape[1] >= 0:
                    logpolar_[
                        y - surrounding.shape[0] : y, x - surrounding.shape[1] : x
                    ] = surrounding_

        for y, x, surrounding in inner_ring_b:
            # Check if valid surrounding
            is_orientation_point, surrounding_ = is_correct_surrounding(
                mean_surrounding, surrounding, show=show
            )
            if is_orientation_point:
                keeps.append((y, x, "inner"))

            if prepare_show_img:
                # Draw point visualization
                cv2.circle(logpolar_, (x, y), 5, (255, 255, 255))
                cv2.circle(logpolar_, (x, y), 2, (0, 255, 0), -1)
                # Place surrounding on image: top left
                if y - surrounding.shape[0] >= 0 and x - surrounding.shape[1] >= 0:
                    logpolar_[
                        y - surrounding.shape[0] : y, x - surrounding.shape[1] : x
                    ] = surrounding_

        for y, x, surrounding in outer_ring_a:
            # Check if valid surrounding
            is_orientation_point, surrounding_ = is_correct_surrounding(
                mean_surrounding, surrounding, show=prepare_show_img
            )
            if is_orientation_point:
                keeps.append((y, x, "outer"))

            if prepare_show_img:
                # Draw point visualization
                cv2.circle(logpolar_, (x, y), 5, (0, 0, 0))
                cv2.circle(logpolar_, (x, y), 2, (0, 0, 255), -1)
                # Place surrounding on image: top right
                if (
                    y - surrounding.shape[0] >= 0
                    and x + surrounding.shape[1] < logpolar_.shape[1]
                ):
                    logpolar_[
                        y - surrounding.shape[0] : y, x : x + surrounding.shape[1]
                    ] = surrounding_

        for y, x, surrounding in outer_ring_b:
            # Check if valid surrounding
            is_orientation_point, surrounding_ = is_correct_surrounding(
                mean_surrounding, surrounding, show=prepare_show_img
            )
            if is_orientation_point:
                keeps.append((y, x, "outer"))

            if prepare_show_img:
                # Draw point visualization
                cv2.circle(logpolar_, (x, y), 5, (0, 0, 0))
                cv2.circle(logpolar_, (x, y), 2, (0, 255, 0), -1)
                # Place surrounding on image: top right
                if (
                    y - surrounding.shape[0] >= 0
                    and x + surrounding.shape[1] < logpolar_.shape[1]
                ):
                    logpolar_[
                        y - surrounding.shape[0] : y, x : x + surrounding.shape[1]
                    ] = surrounding_

        if sum(1 for p in keeps if p[-1] == "outer") < 2:
            print("ERROR: Too few orientation points!")
            return None

        if show:
            show_imgs(positions=logpolar_, block=False)
        if combined_img:
            combined_img.append(("Logpolar Orientation Points", logpolar_))

        # -----------------------------
        # Sort keeps into bins
        def y_to_angle_bin(
            keeps: list[tuple[int, int, str]]
        ) -> list[list[tuple[int, str]]]:
            out = [[] for _ in range(10)]
            for y, x, position in keeps:
                i = (y - 25) // 50

                # If we are in the left half, we invert x,
                # indicating that we move backwards from the center
                if i >= 10:
                    x *= -1
                    i %= 10
                out[i].append((x, position))
            return out

        positions: list[list[tuple[int, str]]] = y_to_angle_bin(keeps)
        # Resolve logpolar bins to real bins
        # Logpolar distortion starts at 3 o'clock while we start rotation at 12
        positions = positions[5:] + positions[:5]
        for i in range(5):
            positions[i] = [(-p[0], p[1]) for p in positions[i]]

        return positions  # (x, inner/outer)

    def structure_orientation_candidates(
        orientation_point_candidates: list[list[tuple[int, str]]],
        cy: int,
        cx: int,
        show: bool = False,
    ):
        # Find outer distances
        outer_dists = []
        for angle_positions in orientation_point_candidates:
            outer_dists += [abs(p[0]) for p in angle_positions if p[1] == "outer"]

        mean_triple_inner_r = np.median(outer_dists)
        mean_triple_inner_std = np.std(outer_dists)

        # FIXME: It's not a good idea to use the standard deviation
        #        since that's really high for skewed images and zero for perfect images.
        #        In the first case, outside points may be falsely classified
        #        and in the second case, all outer triples are falsely classified
        #           -> no std = threshold at inner radius
        double_threshold = mean_triple_inner_r + 4 * mean_triple_inner_std

        # TODO: correct radii
        print("TODO: CV.structure_orientation_candidates: Get board radii")
        r_triple_inner = 170
        r_triple_outer = 190
        r_double_inner = 480
        # print(outer_dists)

        src = []
        dst = []

        global combined_img
        prepare_show_img = show or combined_img
        if prepare_show_img:
            img = img_undistort.copy()

        for i, angle_positions in enumerate(orientation_point_candidates):
            theta = np.pi / 20 + i * np.pi / 10
            for r, pos in angle_positions:
                src_y = cy - np.cos(theta) * r
                src_x = cx + np.sin(theta) * r
                if pos == "outer":
                    # triple ring - outside
                    dst_r = r_triple_outer * (1 if r > 0 else -1)
                elif abs(r) > double_threshold:
                    # TODO: make the radius threshold be dependent on this angle's "outer" radius
                    #       if that does not exist, use the neighboring, iteratively
                    #       if there are no neighbors (= no "outers"), use double_threshold
                    # double ring
                    dst_r = r_double_inner * (1 if r > 0 else -1)
                else:
                    # triple ring - inside
                    dst_r = r_triple_inner * (1 if r > 0 else -1)
                dst_y = 400 - np.cos(theta) * dst_r
                dst_x = 400 + np.sin(theta) * dst_r
                src.append((src_x, src_y))
                dst.append((dst_x, dst_y))

                if prepare_show_img:
                    if abs(abs(dst_r) - r_triple_outer) < 1e-3:
                        c = (255, 0, 0)
                    elif abs(abs(dst_r) - r_double_inner) < 1e-3:
                        c = (0, 255, 0)
                    else:
                        c = (250, 250, 250)
                    cv2.circle(img, (int(src_x), int(src_y)), 5, c, thickness=2)
                    cv2.circle(
                        img,
                        (int(dst_x), int(dst_y)),
                        5,
                        [int(i * 0.75) for i in c],
                        thickness=2,
                    )
                    cv2.line(
                        img,
                        (int(src_x), int(src_y)),
                        (int(dst_x), int(dst_y)),
                        (255, 0, 0),
                    )

        if prepare_show_img:
            cv2.circle(img, (cx, cy), int(double_threshold), (127, 127, 127))
        if show:
            show_imgs(projection_mapping=img, block=False)
        if combined_img:
            combined_img.append(("Orientation Point Projections", img))

        return src, dst  # (x, y), (x, y)

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
        ransac_amount = max(4, int(ransac_percent * len(src_pts)))
        for i in range(n_tries):
            try_indices = np.random.permutation(len(src_pts))[:ransac_amount]
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


def extract_center(img: np.ndarray):
    edges = Edges.edge_detect(img)
    skeleton = Edges.skeleton(edges)
    lines = Lines.extract_lines(skeleton)  # (p1, p2, length, rho, theta)
    lines_binned = Lines.bin_lines_by_angle(lines)
    cy, cx = Lines.get_center_point(img.shape, lines_binned)
    return cy, cx


# -----------------------------------------------

if __name__ == "__main__":
    all_lines = []

    # img_paths.reverse()
    for f in img_paths:
        combined_img = None
        combined_img = []

        from time import time

        start = time()

        # -----------------------------
        # Load Image
        print(f)
        img_full = Utils.load_img(f, show=False)
        if img_full is None:
            print("WARNING: Could not load image path", f)
            continue
        img = Utils.downsample_img(img_full)

        # show_imgs(input=img, block=False)
        if combined_img is not None:
            combined_img.append((f"Input: {f}", img))

        # -----------------------------
        # EDGES

        # Detect Edges
        edges = Edges.edge_detect(img, show=False)
        # Skeletonize edges
        skeleton = Edges.skeleton(edges, show=False)

        # -----------------------------
        # LINES

        # Extract lines
        lines = Lines.extract_lines(skeleton, show=False)

        # Bin lines by angle
        lines_binned = Lines.bin_lines_by_angle(lines)

        # Find Board Center
        cy, cx = Lines.get_center_point(img.shape, lines_binned, show=False)

        # Filter Lines by Center Distance
        lines_filtered = Lines.filter_lines_by_center_dist(
            lines, cy, cx
        )  # p1, p2, length (normalized), center distance [px], rho, theta

        thetas = Lines.get_rough_line_angles(
            img.shape[:2], lines_filtered, cy, cx, show=False
        )
        if len(thetas) != 10:
            print("ERROR: Could not find all lines!")
            if combined_img:
                combined_img = Utils.create_combined_img(combined_img, failed=True)
                show_imgs(combined_img)
            continue

        # -----------------------------
        # ORIENTATION

        # Align lines by filtered edges
        lines = Orientation.align_angles(
            lines_filtered, thetas, img.shape[:2], show=False
        )

        # Calculate better center coordinates
        cy, cx = Orientation.center_point_from_lines(lines)

        # Get undistortion matrix
        M_undistort = Orientation.undistort_by_lines(cy, cx, lines, show=False)

        # Undistort image
        img_undistort = apply_matrix(img, M_undistort)

        cx_undistort, cy_undistort = (M_undistort @ np.array([cx, cy, 1]))[:2]

        orientation_point_candidates = Orientation.find_orientation_points(
            img_undistort, int(cy_undistort), int(cx_undistort), show=False
        )
        if orientation_point_candidates is None:
            if combined_img:
                combined_img = Utils.create_combined_img(combined_img, failed=True)
                show_imgs(combined_img)
            continue

        src_pts, dst_pts = Orientation.structure_orientation_candidates(
            orientation_point_candidates,
            int(cy_undistort),
            int(cx_undistort),
            show=False,
        )

        M_align = Orientation.get_alignment_matrix(
            src_pts, dst_pts, int(cy_undistort), int(cx_undistort)
        )

        # Combine all matrices
        scale = img.shape[0] / img_full.shape[0]

        M_full = np.eye(3)
        M_full = scaling_matrix(scale) @ M_full  # downscale to calculation size
        M_full = M_undistort @ M_full  # undistort
        M_full = M_align @ M_full  # align to correct scale and orientation

        res = apply_matrix(img_full, M_full, adapt_frame=True)
        # show_imgs(aligned=res, block=False)

        if combined_img:
            combined_img.append(("Aligned Image", res))
            combined_img = Utils.create_combined_img(combined_img)
            show_imgs(combined_img)


# -------------------------------------------------------------------------------------------------


class Unused:

    def ellipse_detect(edges: np.ndarray):
        if len(edges.shape) == 2:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        print(edges.dtype)
        # edges[edges < 127] = 0
        # edges[edges != 0] = 255
        show_imgs(edges)

    def paper_ellipses_old(edges):
        """
        A NEW EFFICIENT ELLIPSE DETECTION METHOD
        https://www.wellesu.com/10.1109/ICPR.2002.1048464
        """

        # edges = cv2.Canny(img, 50, 150)
        def dist(x_1, y_1, x_2, y_2):
            return np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)

        # -------------------------------------------
        # equations
        def eq_1(x_1, x_2):
            return (x_1 + x_2) / 2

        eq_2 = eq_1

        def eq_3(x_1, y_1, x_2, y_2):
            return dist(x_1, y_1, x_2, y_2) / 2

        def eq_4(x_1, y_1, x_2, y_2):
            return np.atan((y_2 - y_1) / (x_2 - x_1))

        # def eq_6(a, d, tau):
        #     a2d2 = a**2 * d**2
        #     return (a2d2 * np.sin(tau)**2) / (a2d2 * eq_6()**2)

        # def eq_6(a, d, f, tau):
        #     return (a**2 + d**2 - f**2) / (2 * a * d)

        # -------------------------------------------

        # 1. Store all edge pixels in a 1D-array
        edges_arr = [
            (
                y,
                x,
            )
            for x in range(edges.shape[1])
            for y in range(edges.shape[0])
            if edges[y, x] > 50
        ]

        # 2. clear accumulator array
        accumulator = []

        # 3. for each pixel (x_1, y_1): 4 - 14
        for i, (y_1, x_1) in enumerate(edges_arr):
            # 4. for each other pixel (x_1, y_2)...
            for j, (y_2, x_2) in enumerate(edges_arr[i + 1 :], i + 1):
                # ... if the distance greater then required least distance, carry on
                least_dist = 10
                if dist(x_1, y_1, x_2, y_2) > least_dist:
                    # 5. calculate the center, orientation and length of major axis
                    x0 = eq_1(x_1, y_2)
                    y0 = eq_2(y_1, y_2)
                    a = eq_3(x_1, y_1, x_2, y_2)
                    alpha = eq_4(x_1, y_1, x_2, y_2)

                    # 6. for each third pixel (x, y) ...
                    for y, x in edges:
                        # ... if the distance between (x, y) and (x_0, y_0) is greater then least required, carry on
                        if (d := dist(x, y, x0, y0)) > least_dist:
                            # 7. Calculate length minor axis

                            cos_tau = (a**2 + d**2 - f**2) / (2 * a * d)
                            b2 = (a**2 * d**2 * np.sin(alpha)) / (a**2 - d**2 * cos_tau)

        show_imgs(edges)

    def find_lines(edge_img):
        def line_eq(p1, p2):
            a = p1[1] - p2[1]
            b = p2[0] - p1[0]
            c = p1[0] * p2[1] - p2[0] * p1[1]
            return a, b, -c

        def find_intersection(l1, l2):
            a1, b1, c1 = l1
            a2, b2, c2 = l2
            det = a1 * b2 - a2 * b1
            if det == 0:
                return None

            # Using Cramer's rule to find intersection
            x = (b2 * c1 - b1 * c2) / det
            y = (a1 * c2 - a2 * c1) / det
            return round(x), round(y)

        # Find lines
        lines = cv2.HoughLinesP(
            edge_img,
            rho=1,
            theta=np.pi / 180,
            threshold=150,
            minLineLength=50,
            maxLineGap=10,
        )  # (n, 1, 4)

        if lines is None:
            return [], []
        lines = lines[:, 0]  # (n, 4)

        line_eqs = [line_eq((x1, y1), (x2, y2)) for x1, y1, x2, y2 in lines]

        # Find line intersections
        intersections = []
        for i, l1 in enumerate(line_eqs):
            for l2 in line_eqs[i + 1 :]:
                if i := find_intersection(l1, l2):
                    intersections.append(i)
        print(intersections)

        # Draw lines
        line_img = np.zeros(edge_img.shape[:2], dtype=np.uint8)
        for x1, y1, x2, y2 in lines:
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)

        show_imgs(edge_img, line_img)

    def keypoints(img: np.ndarray):
        orb = cv2.ORB_create()
        kps, desc = orb.detectAndCompute(img, None)
        out = img.copy()
        for kp in kps:
            out[int(kp.pt[1]), int(kp.pt[0])] = (255, 0, 0)
        show_imgs(out)

    def fit_sine_curve(img: np.ndarray, hough_space: np.ndarray):

        from scipy.optimize import curve_fit

        def sine_curve(x, A, B, C, D):
            return A * np.cos(B * x + C) + D

        xdata = []
        ydata = []
        intensities = []
        for y, cols in enumerate(hough_space):
            for x, intensity in enumerate(cols):
                if intensity > 0.1:
                    xdata.append(x)
                    ydata.append(y)
                    intensities.append(intensity)

        params = (np.pi / 2 * img.shape[0], 0.02, 0, img.shape[0] // 2)
        # params, covariance = curve_fit(
        #     f=sine_curve,
        #     xdata=xdata,
        #     ydata=ydata,
        #     p0=params,
        #     sigma=intensities,
        # )
        x = hough_space.shape[1]
        x_span = np.linspace(0, x - 1, x)
        y_fit = sine_curve(x_span, *params)

        res = cv2.cvtColor(hough_space, cv2.COLOR_GRAY2BGR)
        for y, x in zip(y_fit, x_span):
            if 0 <= y < res.shape[0]:
                res[int(y), int(x)] = (255, 0, 0)

        res = cv2.resize(
            res, (res.shape[1] * 4, res.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        return res

    def matching(img: np.ndarray):
        filter_size = 45
        filters_dir = os.path.join("data", "cv", "filters")
        filter_paths = [
            os.path.join(filters_dir, f)
            for f in os.listdir(filters_dir)
            if f.endswith(".png")
        ]
        filters = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in filter_paths]
        n_filters = len(filters)
        for f in range(n_filters):
            filter = filters[f]
            for i in [2, 3]:
                squished_x = cv2.resize(
                    filter.copy(), (filter.shape[1] // i, filter.shape[0])
                )
                squished_y = cv2.resize(
                    filter.copy(), (filter.shape[1], filter.shape[0] // i)
                )
                filters.append(squished_x)
                filters.append(squished_y)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, -1]
        orig_shape = img.shape[:2]
        while min(*img.shape[:2]) >= filter_size:
            res = []
            for f in filters:
                # Filter image
                img_filtered = 1 - cv2.matchTemplate(
                    img, f, method=cv2.TM_CCOEFF_NORMED
                )
                img_filtered = cv2.resize(
                    img_filtered,
                    (orig_shape[1], orig_shape[0]),
                    interpolation=cv2.INTER_NEAREST_EXACT,
                )
                img_filtered[: f.shape[0], : f.shape[1]] = f

                # Normalize by abs
                # img_filtered -= 0.5
                # img_filtered = abs(img_filtered)
                # img_filtered *= 2

                res.append(img_filtered)

            out = np.ones(orig_shape, dtype=np.float32)
            for r in res:
                out *= r

            # Rectangle in img
            img_ = img.copy()
            cv2.rectangle(img_, (0, 0, 45, 45), (127, 127, 127))
            show_imgs(img_, *res, out)
            img = cv2.pyrDown(img)
        # exit()

    def circles(img: np.ndarray):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=10,
            param1=200,
            param2=15,
            minRadius=10,
            maxRadius=min(*img.shape[:2]) // 2,
        )[0]

        print(len(circles))
        res = np.zeros_like(img, dtype=np.float32)
        for i, (cx, cy, r) in enumerate(circles, 1):
            fac = i / len(circles)
            col = 1 - fac
            cx = round(cx)
            cy = round(cy)
            if 0 <= cx < res.shape[1] and 0 <= cy < res.shape[0]:
                res[round(cy), round(cx)] += col
            res += cv2.circle(
                res * 0, (round(cx), round(cy)), round(r), (col, col, col)
            )
        # res = cv2.blur(res, (7, 7))
        res /= res.max()
        show_imgs(img, res)
        return res

    def feature_matching(img: np.ndarray):
        dartboard_img = cv2.imread(
            os.path.join("data", "cv", "dartboard.png"), cv2.IMREAD_GRAYSCALE
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, desc1 = sift.detectAndCompute(img, None)
        kp2, desc2 = sift.detectAndCompute(dartboard_img, None)

        flann = cv2.FlannBasedMatcher(
            indexParams=dict(algorithm=1, trees=5),
            searchParams=dict(checks=50),
        )
        matches = flann.knnMatch(desc1, desc2, k=2)

        good = [m for m, n in matches]
        # good = []
        # for m, n in matches:
        #     if m.distance < 0.7 * n.distance:
        #         good.append(m)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        matches_mask = mask.ravel().tolist()

        h, w = img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst = cv2.perspectiveTransform(pts, M)

        res = cv2.polylines(dartboard_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_params = dict(
            matchColor=(0, 255, 0),  # draw matches in green color
            singlePointColor=None,
            matchesMask=matches_mask,  # draw only inliers
            flags=2,
        )
        res = cv2.drawMatches(img, kp1, res, kp2, good, None, **draw_params)
        show_imgs(img, res)

    def hough_ellipse(
        edges,
        min_major_axis=10,
        min_votes=10,
        top_n=10,
        max_pixels=200,
    ):
        """
        Source: https://github.com/boechat107/imgproc_scripts/blob/master/hough_ellipse.m
        """
        from scipy.spatial.distance import cdist
        from collections import defaultdict

        height, width = edges.shape
        ys, xs = np.nonzero(edges)
        pixels = np.column_stack((xs, ys))
        if len(pixels) > max_pixels:
            pixels = pixels[
                np.random.choice(pixels.shape[0], max_pixels, replace=False)
            ]
        max_b = int(max(width, height) / 2)

        ellipses = []

        # Precompute pairwise distances between points and apply conditions
        dist_matrix = cdist(pixels, pixels)
        valid_pairs = np.argwhere(
            (dist_matrix > min_major_axis) & (dist_matrix < max(width, height) / 2)
        )

        for n, (i, j) in enumerate(valid_pairs):
            print(f"{n+1}/{len(valid_pairs)}", end="\r")
            x1, y1 = pixels[i]
            x2, y2 = pixels[j]
            dist_12 = dist_matrix[i, j]

            # Midpoint, major axis length, and orientation
            x0, y0 = (x1 + x2) / 2, (y1 + y2) / 2
            a = dist_12 / 2
            alpha = np.arctan2(y2 - y1, x2 - x1)

            # Accumulator dictionary for minor axis values
            acc = defaultdict(int)

            # Compute for each third point
            d03_array = np.hypot(pixels[:, 0] - x0, pixels[:, 1] - y0)
            f_array = np.hypot(pixels[:, 0] - x2, pixels[:, 1] - y2)

            # Filter to points within range to be valid `x3, y3` candidates
            valid_third_points = np.where((d03_array < a) & (d03_array > 0))[0]

            # Vectorize b calculation
            for idx in valid_third_points:
                d03, f = d03_array[idx], f_array[idx]
                cos2_tau = ((a**2 + d03**2 - f**2) / (2 * a * d03)) ** 2
                sin2_tau = 1 - cos2_tau

                try:
                    b = int(
                        np.sqrt((a**2 * d03**2 * sin2_tau) / (a**2 - d03**2 * cos2_tau))
                    )
                    if 0 < b < max_b:
                        acc[b] += 1
                except ValueError:
                    continue  # skip invalid sqrt values

            # Find the best minor axis and accumulate results
            if not acc:
                continue

            best_b, max_votes_found = max(acc.items(), key=lambda item: item[1])
            if max_votes_found >= min_votes:
                ellipses.append((max_votes_found, x0, y0, a, best_b, alpha))

        # Sort ellipses by votes and return the top `top_n`
        ellipses = sorted(ellipses, key=lambda x: x[0], reverse=True)[:top_n]
        if ellipses:
            return [(x0, y0, a, b, alpha) for _, x0, y0, a, b, alpha in ellipses]

        print("No ellipses detected!")
        return None

    def line_detect(edge_img: np.ndarray) -> np.ndarray:

        lsd = cv2.createLineSegmentDetector()
        lines = lsd.detect(edge_img)[0][:, 0]

        res = np.zeros_like(edge_img)
        for x1, y1, x2, y2 in lines:
            cv2.line(res, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))
        show_imgs(edge_img, res)
        return lines

    def extend_line(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1

        x1 -= dx
        x2 += dx
        y1 -= dy
        y2 += dy
        return x1, y1, x2, y2

    def fit_ellipse(edges: np.ndarray):
        # Make image square
        if edges.shape[0] != edges.shape[1]:
            max_size = max(*edges.shape[:2])
            temp = np.zeros((max_size, max_size), np.uint8)
            dy = (temp.shape[0] - edges.shape[0]) // 2
            dx = (temp.shape[1] - edges.shape[1]) // 2
            temp[dy : dy + edges.shape[0], dx : dx + edges.shape[1]] = edges
            edges = temp

        # Extract points
        y, x = np.nonzero(edges)

        # Solve normal system of equations using SVD
        Z = np.column_stack((x**2, x * y, x**2, x, y, np.ones_like(x)))
        _, _, V = np.linalg.svd(Z)
        params = V[-1, :]

        A, B, C, D, E, F = params

        # Calculate center
        cx = (C * D - B * E) / (B**2 - A * C)
        cy = (A * E - B * D) / (B**2 - A * C)

        # Calculate the semi-major and semi-minor axes (a, b) and the orientation angle (theta)
        up = 2 * (A * E**2 + C * D**2 + F * B**2 - 2 * B * D * E - A * C * F)
        down1 = (B**2 - A * C) * (
            (C - A) * np.sqrt(1 + (4 * B**2) / ((A - C) ** 2)) - (C + A)
        )
        down2 = (B**2 - A * C) * (
            (A - C) * np.sqrt(1 + (4 * B**2) / ((A - C) ** 2)) - (C + A)
        )
        a = np.sqrt(abs(up / down1))
        b = np.sqrt(abs(up / down2))

        # Calculate orientation of the ellipse
        theta = 0.5 * np.arctan2(B, A - C)

        center = (round(cx), round(cy))
        axes = (round(max(a, b)), round(min(a, b)))

        res = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.ellipse(
            res,
            center=center,
            axes=axes,
            angle=theta,
            startAngle=0,
            endAngle=360,
            color=(255, 0, 0),
        )

        show_imgs(edges, res)
        exit()

    def transfer_hough_space(img, edge_img: np.ndarray) -> np.ndarray:
        rho = 0.1
        theta = np.pi / 180 * 2
        threshold = None
        lines = cv2.HoughLinesWithAccumulator(
            edge_img,
            rho,
            theta,
            threshold,
        )[
            :, 0
        ]  # (rho [-diag..diag], theta [-pi..pi], intensity [0..?])

        # Normalize strengths
        lines[:, 2] /= lines[:, 2].max()

        # Create result image
        res = np.zeros_like(edge_img)
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        res = res.astype(np.float32)

        # Sort lines by intensity
        lines = sorted(lines, key=lambda x: x[2], reverse=True)
        # lines = lines[:1000]
        lines = np.array(lines)
        # Calculate output size
        diag = np.sqrt(edge_img.shape[0] ** 2 + edge_img.shape[1] ** 2)
        y = int(np.ceil(diag))
        x = 180

        # Normalize rho
        lines[:, 0] += y

        # Theta to degrees
        lines[:, 1] = lines[:, 1] / np.pi * 180

        # Normalize strengths
        lines[:, 2] /= lines[:, 2].max()

        hough_space = np.zeros((y * 2, x), dtype=np.float32)
        hough_space[
            lines[:, 0].astype(np.int32), lines[:, 1].astype(np.int32)
        ] += lines[:, 2]
        hough_space /= hough_space.max()
        hough_space = (hough_space * 255).astype(np.uint8)
        return hough_space

    def detect_hough_lines(img, edge_img: np.ndarray):
        rho = 0.1
        theta = np.pi / 180 / 10
        threshold = None
        lines = cv2.HoughLinesWithAccumulator(
            edge_img,
            rho,
            theta,
            threshold,
        )[
            :, 0
        ]  # (rho [-diag..diag], theta [-pi..pi], intensity [0..?])

        # Calculate output size
        diag = np.sqrt(edge_img.shape[0] ** 2 + edge_img.shape[1] ** 2)
        y = int(np.ceil(diag))

        # Normalize rho
        lines[:, 0] += y

        # Theta to degrees
        lines[:, 1] = np.rad2deg(lines[:, 1])

        # Normalize strengths
        lines[:, 2] /= lines[:, 2].max()

        # Create hough space
        hough_space = np.zeros((y * 2, 180), dtype=np.float32)
        ys = lines[:, 0].astype(np.int32)
        xs = lines[:, 1].astype(np.int32)
        for y, x in zip(ys, xs):

            hough_space[y, x] += lines[:, 2]

        # Normalize Hough space
        hough_space /= hough_space.max()
        hough_space = (hough_space * 255).astype(np.uint8)
        hough_space = hough_space.T

        # Non-maximum suppression
        # hough_space = cv2.blur(hough_space, (3, 3))
        # hough_space = Utils.non_maximum_suppression(hough_space)

        return hough_space

    def color_spaces(img: np.ndarray):

        def cvt(img):
            img = np.float32(img) - 128  # -128..127
            img = abs(img)  # 0..128
            img /= img.max()  # Normalize
            # img = np.sqrt(img)
            img *= 255
            img = np.uint8(img)
            return img

        good_imgs = []

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab = cvt(lab)
        good_imgs.append(lab[:, :, 1])  # a

        luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        luv = np.float32(luv)
        luv -= np.median(luv)
        luv = abs(luv)
        luv /= luv.max()
        luv = np.uint8(luv * 255)
        good_imgs.append(luv[:, :, 1])  # u

        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycc = cvt(ycc)
        good_imgs.append(ycc[:, :, 1])

        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv = cvt(yuv)
        # good_imgs.append(yuv[:, :, 2])  # v

        # -------------------------------------------
        # Concat
        combined = np.concatenate(good_imgs, axis=1)
        while combined.shape[1] > 2500:
            combined = cv2.pyrDown(combined)

        # Sum
        sum_img = np.zeros(img.shape[:2], np.float32)
        for i in good_imgs:
            sum_img += i
        sum_img /= sum_img.max()
        sum_img = np.uint8(255 * sum_img)

        # Mult
        mult_img = np.ones(img.shape[:2], np.float32)
        for i in good_imgs:
            i = np.float32(i) / 255
            mult_img *= i * 0.9 + 0.1
        mult_img /= mult_img.max()
        mult_img = np.uint8(255 * mult_img)

        res = np.float32(sum_img) + np.float32(mult_img)
        res_out = np.uint8(res / 2)

        # show_imgs(img, combined, sum_img, mult_img, res)
        # return res_out

        # ===========================================

        color_spaces = [
            cv2.COLOR_BGR2GRAY,
            # cv2.COLOR_BGR2HLS,
            # cv2.COLOR_BGR2HSV,
            cv2.COLOR_BGR2LAB,
            cv2.COLOR_BGR2LUV,
            # cv2.COLOR_BGR2RGB,
            # cv2.COLOR_BGR2RGBA,
            # cv2.COLOR_BGR2XYZ,
            cv2.COLOR_BGR2YCrCb,
            # cv2.COLOR_BGR2YUV,
        ]
        res = np.zeros(
            (img.shape[0] * 4, img.shape[1] * len(color_spaces), 3), np.uint8
        )
        dy, dx = img.shape[:2]
        res[:dy, :dx] = img

        def add_img(res, img, x, y):
            y0 = y * dy
            x0 = x * dx
            res[y0 : y0 + dy, x0 : x0 + dx] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return res

        for x, cs in enumerate(color_spaces):
            cs_img = cv2.cvtColor(img, cs)
            if len(cs_img.shape) == 2:
                cs_img = np.expand_dims(cs_img, -1)
            for y in range(cs_img.shape[-1]):
                res = add_img(res, cs_img[:, :, y], x, y)
        while res.shape[1] > 2000 or res.shape[0] > 1000:
            res = cv2.resize(res, (res.shape[1] // 2, res.shape[0] // 2))
        show_imgs(res)
        return res_out

        sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
        lab_a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1]

        # comb = (np.float32(sat) / 255) * (np.float32(lab_a) / 255)
        # comb /= comb.max()
        # comb = np.uint8(comb * 255)

        show_imgs(img, sat, lab_a)
        return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        print(hls.shape)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        print(hsv.shape)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        print(lab.shape)
        show_imgs(img, gray)
        show_imgs(img, hls[:, :, 0], hls[:, :, 1], hls[:, :, 2])
        show_imgs(img, hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2])
        show_imgs(img, lab[:, :, 0], lab[:, :, 1], lab[:, :, 2])

    def fit_ellipse_2(skeleton, cy, cx):
        def ellipse_residuals(params, points, center):
            a, b, theta = params
            y_c, x_c = center
            # Translate ellipse to center
            y = points[:, 0] - y_c
            x = points[:, 1] - x_c

            # Rotate points by theta
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            x_rot = x * cos_theta + y * sin_theta
            y_rot = y * cos_theta - x * sin_theta

            residuals = (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1
            return residuals

        edge_img = cv2.dilate(skeleton.copy(), (3, 3), iterations=2)
        points_y, points_x = np.nonzero(edge_img)
        points = np.vstack([points_y, points_x]).T
        points[:, 0] -= cy
        points[:, 1] -= cx

        initial_guess = (50, 50, 0)

        # from scipy.optimize import least_squares
        # res = least_squares(ellipse_residuals, initial_guess, args=(points, (cy, cx)))
        # a, b, theta = res.x
        ellipse = cv2.fitEllipse(points)
        a, b = ellipse[1]
        theta = ellipse[2]
        a = int(a / 2)
        b = int(b / 2)

        print()
        print(a, b, theta)

        cv2.ellipse(
            img, (cx, cy), (int(a), int(b)), theta, 0, 360, color=(255, 255, 255)
        )
        show_imgs(img)

    def get_hough_space(image, edges: np.ndarray) -> np.ndarray:
        import matplotlib.pyplot as plt

        # Hough Transform parameters
        height, width = edges.shape
        diag_len = int(np.sqrt(height**2 + width**2))  # Maximum possible rho
        rho_max = diag_len  # Range for rho (-diag_len to +diag_len)
        theta_max = 180  # Number of theta values (0 to 180 degrees)

        # Initialize Hough accumulator
        accumulator = np.zeros((2 * rho_max, theta_max), dtype=np.float32)
        theta_range = np.deg2rad(np.arange(0, theta_max))  # Convert to radians

        # Map edge points to Hough space
        for y in range(height):
            print(y, height, end="\r")
            for x in range(width):
                if edges[y, x]:  # If edge pixel
                    for theta_index, theta in enumerate(theta_range):
                        rho = int(x * np.cos(theta) + y * np.sin(theta))
                        accumulator[rho + rho_max, theta_index] += 1
        print(" " * 10, end="")

        # Display original image, edges, and Hough space
        # plt.figure(figsize=(15, 5))

        # plt.subplot(1, 3, 1)
        # plt.title("Original Image")
        # plt.imshow(image, cmap="gray")

        # plt.subplot(1, 3, 2)
        # plt.title("Edges")
        # plt.imshow(edges, cmap="gray")

        # plt.subplot(1, 3, 3)
        # plt.title("Hough Space")
        # plt.imshow(
        #     accumulator, cmap="hot", aspect="auto", extent=[0, 180, -rho_max, rho_max]
        # )
        # plt.xlabel("Theta (degrees)")
        # plt.ylabel("Rho (pixels)")
        # plt.colorbar(label="Votes")

        # plt.tight_layout()
        # plt.show()
        accumulator /= accumulator.max()
        accumulator = np.uint8(accumulator * 255)
        res = cv2.applyColorMap(accumulator, cv2.COLORMAP_JET)
        return res

    def find_center(edges: np.ndarray):

        return img

        acc_thresh = cv2.threshold(acc, 200, 255, cv2.THRESH_BINARY)[1]
        suppressed_mask = Utils.non_maximum_suppression(acc_thresh)
        suppressed = np.zeros_like(acc)
        suppressed[suppressed_mask != 0] = acc[suppressed_mask != 0]

        suppressed -= 200
        suppressed = np.float32(suppressed) / 55
        suppressed = np.uint8(suppressed * 255)

        return suppressed

    def find_board_lines(
        cy: int,
        cx: int,
        lines_binned: list[
            tuple[
                tuple[int, int],  # p1
                tuple[int, int],  # p2
                float,  # length
                float,  # distance
                float,  # rho
                float,  # theta
            ]
        ],
        lines_binned_filtered,
    ):
        def center_radius(y, x):
            return np.sqrt((cx - x) ** 2 + (cy - y) ** 2)

        # Growing radii on grayscale image
        points = []
        max_r = img.shape[0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1]
        gray = abs(gray - 128)
        # gray = cv2.convertScaleAbs(gray, alpha=2, beta=0)
        # gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        for r in range(max_r):
            circle_img = cv2.circle(gray * 0, (cx, cy), r, (255, 255, 255))
            intersection = np.logical_and(circle_img, gray)
            n_points = np.count_nonzero(intersection)
            points.append(n_points)
            # show_imgs(gray, np.uint8(intersection) * 255)
        res = np.zeros((max(points) + 1, max_r))
        for x, y in enumerate(points):
            res[y, x] = 255
        show_imgs(gray, res)
        return

        # Growing radii on skeleton image
        points = []
        max_r = img.shape[0]
        for r in range(max_r):
            circle_img = cv2.circle(skeleton * 0, (cx, cy), r, (255, 255, 255))
            intersection = np.logical_and(skeleton, circle_img)
            n_points = np.count_nonzero(intersection)
            points.append(n_points)
        res = np.zeros((max(points) + 1, max_r))
        for x, y in enumerate(points):
            res[y, x] = 255
        res[20] = 127
        show_imgs(res, skeleton)
        return

        # Look for similarities in line start and end point distances from center
        bin_radii = []
        max_r = 0
        # Look for board radii
        for bin_idx, bin_lines in enumerate(lines_binned):
            bin_radii.append([])
            for line in bin_lines:
                line_start = line[0]
                line_end = line[1]
                r1 = center_radius(*line_start)
                r2 = center_radius(*line_end)
                max_r = max([max_r, r1, r2])
                bin_radii[-1] += [r1, r2]
            bin_radii[-1] = sorted(bin_radii[-1])
        max_r = int(np.ceil(max_r))

        res = np.zeros((500, max_r))
        for bin_idx, radii in enumerate(bin_radii):
            for r in radii:
                res[bin_idx * 50 : (bin_idx + 1) * 50, int(r)] = 255
        show_imgs(img, skeleton, res)
        # exit()

    def ray_extensions(img, lines_binned_filtered):
        def within_img(y, x):
            if x < 0 or y < 0:
                return False
            if y >= img.shape[0]:
                return False
            if x >= img.shape[1]:
                return False
            return True

        # Prepare image
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.convertScaleAbs(img, alpha=1, beta=0)

        # Get line orientations
        bin_thetas = []
        for bin_lines in lines_binned_filtered:
            bin_thetas.append([])
            for line in bin_lines:
                bin_thetas[-1].append(line[-1])
        mean_thetas = [np.median(t) if t else -1000 for t in bin_thetas]  # (n_bins,)

        # Get inbetween rays
        ray_directions = []
        for t1, t2 in zip(mean_thetas, mean_thetas[1:] + [mean_thetas[0] + np.pi]):
            if t1 < -100 or t2 < -100:
                ray_directions.append(0)

            ray_directions.append((t1 + t2) / 2)

        # March along rays
        rays = [
            [] for _ in ray_directions * 2
        ]  # twice the amount of rays since we go both forward and backward

        ray_length = img.shape[0]
        for step in range(ray_length):
            for i, theta in enumerate(ray_directions):
                px = cx + int(-np.sin(theta) * step)
                py = cy + int(np.cos(theta) * step)
                if within_img(py, px):
                    intensity = img[py, px]
                    rays[i].append(intensity)

                px = cx - int(-np.sin(theta) * step)
                py = cy - int(np.cos(theta) * step)
                if within_img(py, px):
                    intensity = img[py, px]
                    rays[i + 10].append(intensity)

        res = np.zeros((500, ray_length), np.uint8)
        for y, ray in enumerate(rays):
            if len(ray) == 0:
                continue
            res[y * 50 : (y + 1) * 50, : len(ray)] = ray[: res.shape[1]]

        # show_imgs(res)
        return res

        # Draw lines
        for theta in ray_directions:
            if theta is None:
                continue
            x0 = cx + int(-np.sin(theta) * 1000)
            y0 = cy + int(np.cos(theta) * 1000)
            x1 = cx - int(-np.sin(theta) * 1000)
            y1 = cy - int(np.cos(theta) * 1000)
            img_ = img * 0
            cv2.line(img_, (x0, y0), (x1, y1), (0, 0, 0), 3)
            cv2.line(img_, (x0, y0), (x1, y1), (255, 255, 255), 1)

        show_imgs(cv2.addWeighted(img, 1, img_, 0.5, 0))

    def check_shearing_angles_calculation():
        def theta_change_x(theta, shear_x):
            """
            slope = dy/dx
            mapping: dy/dx -> (dx + s) / dx
            mapping straightforward since we change the numerator
            """
            slope = np.tan(theta)
            slope_ = slope + shear_x
            theta_ = np.arctan(slope_)
            return theta_

        def theta_change_y(theta, shear_y):
            """
            slope = dy / dx
            mapping: dy/dx -> dy / (dy * s + dx)
                            = (dy / dx) / (1 + s * (dy / dx))
                            = slope / (1 + s * slope)
            mapping not as straightforward as we change the denominator
            """
            slope = np.tan(theta)

            slope_ = slope / (1 + shear_y * slope)

            theta_ = np.arctan(slope_)
            print(
                "\n",
                f"theta={theta}",
                f"theta={round(np.rad2deg(theta))}",
                f"slope={slope:.04f}",
                f"shearing={shearing}",
                f"slope_={slope_:04f}",
                f"theta_={round(np.rad2deg(theta_))}",
                f"theta_={theta_}",
                sep="\n",
            )
            return theta_

        img = cv2.imread(img_paths[0])
        img = cv2.pyrDown(img)
        cy = img.shape[0] // 2
        cx = img.shape[1] // 2

        shearing = 1

        M_t_a = translation_matrix(-cx, -cy)
        M_t_b = translation_matrix(cx, cy)
        M_shear = shearing_matrix(-shearing)
        M = np.eye(3)
        M = M_t_a @ M
        M = M_shear @ M
        M = M_t_b @ M

        # Draw lines
        thetas = np.arange(0, np.pi, np.pi / 8)
        colors = [
            (127, 127, 127),
            (0, 0, 255),
            (0, 255, 0),
            (0, 255, 255),
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (255, 255, 255),
        ]

        # thetas = thetas[1:2]
        for i, theta in enumerate(thetas):
            draw_polar_line_through_point(
                img, (cy, cx), theta, color=colors[i], thickness=6
            )

        # Warp image
        img_ = apply_matrix(img, M)

        # Calculate resulting lines
        for i, theta in enumerate(thetas):
            # theta_ = theta_change_x(theta, shearing)
            theta_ = theta_change_y(theta, shearing)
            draw_polar_line_through_point(
                img_,
                (cy, cx),
                theta_,
                thickness=2,
                color=colors[i],
                intensity=1,
            )
        show_imgs(img, img_)
        exit()

    def view_along_lines(img: np.ndarray, cy: int, cx: int) -> list[np.ndarray]:
        lines = []
        angle_step = np.pi / 10
        thetas = np.arange(0, np.pi, angle_step) + angle_step / 2
        diag = int(np.ceil(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)))

        search_distance = angle_step / 3

        def get_position(cy, cx, dy, dx, step) -> tuple[int, int]:
            y = cy + step * dy
            x = cx + step * dx
            return x, y

        def get_edge_intersection(
            cy: int,
            cx: int,
            theta: float,
            height: int,
            width: int,
        ):
            dy = -np.cos(theta)
            dx = np.sin(theta)

            t_values = []

            if dy < 0:  # top edge
                t_top = -cy / dy
                t_values.append(t_top)
            elif dy > 0:  # bottom edge
                t_bottom = (height - 1 - cy) / dy
                t_values.append(t_bottom)

            if dx < 0:  # left edge
                t_left = -cx / dx
                t_values.append(t_left)
            elif dx > 0:  # right edge
                t_right = (width - 1 - cx) / dx
                t_values.append(t_right)

            # Find the smallest positive t
            t_exit = min(t for t in t_values if t > 0)

            # Compute the intersection point
            x_exit = cx + t_exit * dx
            y_exit = cy + t_exit * dy

            return (int(y_exit), int(x_exit))

        def get_slice(
            img: np.ndarray,
            cy: int,
            cx: int,
            theta_l: float,
            theta_r: float,
            diag: int,
        ):
            # Mask out slice
            mask = cv2.ellipse(
                img=np.zeros_like(img),
                center=(cx, cy),
                axes=(diag, diag),
                angle=0,
                startAngle=np.rad2deg(theta_l) - 90,
                endAngle=np.rad2deg(theta_r) - 90,
                color=(1, 1, 1),
                thickness=-1,
            )
            slice = img * mask
            # Align slice to be vertical
            rot_angle = theta_l + (theta_r - theta_l) / 2
            rot_angle *= -1
            # This was intended to get the correct output size,
            # but just shifting everything to the bottom also does the trick
            # edge_l = get_edge_intersection(cy, cx, theta_l, *img.shape[:2])
            # edge_r = get_edge_intersection(cy, cx, theta_r, *img.shape[:2])

            M = np.eye(3)
            M_trans_a = translation_matrix(-cx, -cy)
            M = M_trans_a @ M
            M_rot = rotation_matrix(rot_angle)
            M = M_rot @ M

            M_trans_b = translation_matrix(cx, img.shape[0])
            M = M_trans_b @ M
            slice = apply_matrix(slice, M)
            return slice

        def get_slice_values(slice):
            nonzero_mask = slice != 0
            sums = np.sum(slice * nonzero_mask, axis=1)
            counts = np.sum(nonzero_mask, axis=1)
            counts[counts == 0] = -1
            mean_values = sums / counts
            mean_values[counts == -1] = 0
            return np.uint8(mean_values)

        def find_edges(slices):
            slices = [cv2.cvtColor(s, cv2.COLOR_BGR2HSV)[:, :, 1:] for s in slices]
            kernel = np.array([-1, -1, -1, 0, 1, 1, 1], np.float32)
            kernel /= np.abs(kernel).sum()
            edges = [cv2.filter2D(s, cv2.CV_32F, kernel) for s in slices]
            for i, e in enumerate(edges):
                sat = e[:, :, 0]
                sat = np.abs(sat)
                sat /= sat.max()

                val = e[:, :, 1]
                val = np.abs(val)
                val /= val.max()

                edge = np.maximum(sat, val)
                edges[i] = np.maximum(sat, val)

            edges = [
                np.transpose(
                    np.kron(e * 255, np.ones((1, 50))).astype(np.uint8), (1, 0)
                )
                for e in edges
            ]
            return edges

        transform = lambda img: np.transpose(
            np.kron(
                img if len(img.shape) == 3 else np.expand_dims(img, -1),
                np.ones((1, 1, 1), np.uint8),
            ),
            (1, 0, 2),
        )

        max_r = min(img.shape[:2]) // 2
        M = max_r / np.log(img.shape[1])
        logpolar = cv2.warpPolar(
            img,
            dsize=(int(diag), 1000),
            center=(int(cx), int(cy)),
            maxRadius=int(diag),
            flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS,
        )

        start_y = 1000 / 20
        start_y /= 2
        for i in range(20):
            y = int(start_y + i * 1000 / 20)
            cv2.line(logpolar, (0, y), (logpolar.shape[1], y), color=(255, 255, 255))

        lab = cv2.cvtColor(logpolar, cv2.COLOR_BGR2LAB)
        start_y = 1000 / 20
        start_y /= 2
        for i in range(20):
            y = int(start_y + i * 1000 / 20)
            cv2.line(lab, (0, y), (lab.shape[1], y), color=(255, 255, 255))
        l = lab[:, :, 0]
        a = np.abs(np.int32(lab[:, :, 1]) - 128).astype(np.uint8)
        b = lab[:, :, 2]

        hsv = cv2.cvtColor(
            cv2.convertScaleAbs(logpolar, alpha=1.5, beta=2.0), cv2.COLOR_BGR2HSV
        )
        # hsv = cv2.threshold(hsv, 220, 255, cv2.THRESH_BINARY)[1]
        s = hsv[:, :, 2]

        show_imgs(logpolar, a, s, block=False)
        return

        dt = np.pi / 10
        n_slices = 20 * 10
        start_theta = -np.pi / 20

        slices = []
        dt = 2 * np.pi / n_slices
        for s in range(n_slices):
            theta_start = start_theta + s * dt
            theta_end = start_theta + (s + 1) * dt
            slice = get_slice(
                img,
                cy,
                cx,
                theta_l=theta_start,
                theta_r=theta_end,
                diag=diag,
            )
            slice_values = get_slice_values(slice)  # y, 3
            slice_values = np.expand_dims(slice_values, 1)  # y, 1, 3
            slices.append(slice_values)
            # show_imgs(slice)
        slices = np.concatenate(slices, axis=1)
        show_imgs(transform(slices), block=False)
        return

        slices_top = []
        slices_bottom = []
        for theta in thetas:
            start_theta = theta - dt

            theta -= dt / 3
            start_theta += dt / 3
            # print(start_theta, theta)
            # Top slices
            slice = get_slice(
                img,
                cy,
                cx,
                theta_l=start_theta,
                theta_r=theta,
                diag=diag,
            )
            slice_values = get_slice_values(slice)  # y, 3
            slice_values = np.expand_dims(slice_values, 1)  # y, 1, 3
            slices_top.append(slice_values)
            # Bottom slice
            slice = get_slice(
                img,
                cy,
                cx,
                theta_l=start_theta + np.pi,
                theta_r=theta + np.pi,
                diag=diag,
            )
            slice_values = get_slice_values(slice)
            slice_values = np.expand_dims(slice_values, 1)
            slices_bottom.append(slice_values)
            continue
        slices = np.concatenate(slices_top + slices_bottom, axis=1)

        # Convert slice colors
        slices_lab = cv2.cvtColor(slices, cv2.COLOR_BGR2LAB)
        slice_a = slices_lab[:, :, 1]
        slice_a = np.int16(slice_a)
        slice_a -= 128
        slice_a = np.abs(slice_a)
        slice_a = np.uint8(slice_a)
        slice_a *= 2

        # Find edges
        kernel = np.array([-1, -1, 1, 1], np.float32)
        kernel /= np.abs(kernel).sum()
        edges = cv2.filter2D(slice_a, cv2.CV_32F, kernel)
        edges /= np.abs(edges).max()
        edges += 1
        edges /= 2
        edges *= 255
        print(edges.min(), edges.max())
        edges = np.uint8(edges)
        edges = np.expand_dims(edges, -1)

        cv2.destroyAllWindows()
        show_imgs(transform(slices))
        return
        scale = 0.01
        p = 0
        while True:
            p += 1
            p %= 360
            print(p)
            phase = np.deg2rad(p)

            x = np.arange(0, 2 * np.pi, np.pi / 10)
            stretches = (scale / 2) * (np.sin(x + phase) + 1) + 1

            stretched_image = []
            for c, s in enumerate(stretches):
                col = edges[:, c]
                stretched_col = cv2.resize(
                    col,
                    (1, int(len(col) * s)),
                    interpolation=cv2.INTER_LINEAR,
                )
                stretched_image.append(stretched_col)
            largest_strip = max(len(s) for s in stretched_image)
            stretched_image = cv2.hconcat(
                [
                    np.pad(strip, [[largest_strip - strip.shape[0], 0], [0, 0]])
                    for strip in stretched_image
                ]
            )

            show_imgs(
                edges=transform(edges),
                stretched_edges=transform(stretched_image),
                block=True,
            )
        # show_imgs(transform(slices), transform(slice_a), transform(edges), transform(stretched_image))
        # exit()
        """
        for i, theta in enumerate(thetas):
            slice_upper = get_slice(
                img,
                cy,
                cx,
                theta_l=theta - search_distance,
                theta_r=theta + search_distance,
                diag=diag,
            )
            slice_lower = get_slice(
                img,
                cy,
                cx,
                theta_l=theta - search_distance + np.pi,
                theta_r=theta + search_distance + np.pi,
                diag=diag,
            )
            slice_upper_left, slice_upper_right = slice_values(
                slice_upper, cx
            )  # y, 1, 3
            slice_lower_left, slice_lower_right = slice_values(
                slice_lower, cx
            )  # y, 1, 3

            diff = len(slice_upper_left) - len(slice_upper_right)
            append = np.zeros((abs(diff), 1, 3), np.uint8)
            if diff < 0:
                slice_lower_left = np.concatenate([append, slice_lower_left], axis=0)
                slice_upper_left = np.concatenate([append, slice_upper_left], axis=0)
            elif diff > 0:
                slice_lower_right = np.concatenate([append, slice_lower_right], axis=0)
                slice_upper_right = np.concatenate([append, slice_upper_right], axis=0)

            slices = np.concatenate(
                [
                    slice_upper_left,
                    slice_upper_right,
                    slice_lower_left,
                    slice_lower_right,
                ],
                axis=1,
            )

            edges = find_edges(
                [
                    slice_upper_left,
                    slice_upper_right,
                    slice_lower_left,
                    slice_lower_right,
                ]
            )

            show_imgs(*edges)
            continue
            colors = np.concatenate(
                [
                    s[:, :, c]
                    for s in [
                        slice_upper_left,
                        slice_upper_right,
                        slice_lower_left,
                        slice_lower_right,
                    ]
                    for c in range(3)
                ],
                axis=1,
            )
            colors = np.kron(colors, np.ones((1, 25))).astype(np.uint8)
            show_imgs(colors=np.transpose(colors, (1, 0)), block=False)

            slices = np.uint8(slices)

            slices = cv2.blur(slices, (3, 1))

            show = np.kron(slices, np.ones((1, 50, 1))).astype(np.uint8)
            show_imgs(slices=np.transpose(show, (1, 0, 2)), block=False)

            edge_kernel = np.expand_dims(
                np.array([-1, -1, -1, 0, 0, 1, 1, 1], np.float32), -1
            )
            edge_kernel /= np.abs(edge_kernel).sum()
            edges = cv2.filter2D(
                cv2.cvtColor(slices, cv2.COLOR_BGR2GRAY), cv2.CV_32F, edge_kernel
            )
            edges -= edges.min()
            edges /= edges.max()
            edges = np.uint8(edges * 255)

            edges = np.kron(edges, np.ones((1, 50)))
            edges = np.uint8(edges)

            show_imgs(np.transpose(edges, (1, 0)))
        """

        # show_imgs(img)

        # ------------------------
        # Color Filter

        def apply_channel_filter(img, kernel):
            filtered = np.zeros(img.shape[:2], np.float32)
            for c in range(img.shape[-1]):
                img_c = img[:, :, c]
                res = cv2.filter2D(img_c, cv2.CV_32F, kernel[c])
                # res = np.expand_dims(res, -1)
                filtered += res
            # filtered = np.abs(filtered)
            filtered[filtered < 0] = 0

            filtered /= filtered.max()
            # filtered = Utils.non_maximum_suppression(filtered)
            # filtered = np.concatenate(filtered, axis=-1)
            # show_imgs(img=img, filtered=filtered)
            filtered = np.uint8(255 * filtered)
            return filtered

        def get_split_kernel(top_left, top_right, bottom_left, bottom_right, size=1):
            if type(size) == int:
                size = (size, size)

            kernel = []
            for i in range(3):
                kernel_c = np.array(
                    [
                        [(top_left[i] - 128) / 255, (top_right[i] - 128) / 255],
                        [(bottom_left[i] - 128) / 255, (bottom_right[i] - 128) / 255],
                    ],
                    np.float32,
                )
                kernel_c = np.kron(kernel_c, np.ones((size[0], size[1]), np.float32))
                kernel_c /= np.abs(kernel_c).sum()
                kernel.append(kernel_c)
            return kernel

        colors = {
            "black": np.array([25, 25, 25]),
            "white": np.array([180, 220, 245]),
            "red": np.array([30, 45, 250]),
            "green": np.array([50, 145, 55]),
        }
        ksize = (7, 2)
        kernel_a = get_split_kernel(
            top_left=colors["white"],
            top_right=colors["white"],  # green
            bottom_left=colors["black"],
            bottom_right=colors["black"],  # red
            size=ksize,
        )
        kernel_b = get_split_kernel(
            top_left=colors["black"],
            top_right=colors["black"],  # red
            bottom_left=colors["white"],
            bottom_right=colors["white"],  # green
            size=ksize,
        )
        kernel_c = get_split_kernel(
            top_left=colors["red"],
            top_right=colors["black"],
            bottom_left=colors["green"],
            bottom_right=colors["white"],
            size=ksize,
        )
        kernel_d = get_split_kernel(
            top_left=colors["green"],
            top_right=colors["white"],
            bottom_left=colors["red"],
            bottom_right=colors["black"],
            size=ksize,
        )
        res_a = apply_channel_filter(logpolar, kernel_a)
        res_b = apply_channel_filter(logpolar, kernel_b)
        # res_c = apply_channel_filter(logpolar, kernel_c)
        # res_d = apply_channel_filter(logpolar, kernel_d)
        res = np.int32(res_a) + np.int32(res_b)  # + np.int32(res_c) + np.int32(res_d)
        res = np.float32(res) / res.max()
        res = np.uint8(res * 255)
        show_imgs(
            logpolar=logpolar,
            res_a=res_a,
            res_b=res_b,
            # res_c=res_c,
            # res_d=res_d,
            res=res,
            block=False,
        )

        # lines_img = np.zeros()
        return

        col_median = np.median(logpolar, axis=0).astype(np.uint8)

        col_mean = np.mean(logpolar, axis=0).astype(np.uint8)

        show_imgs(
            logpolar,
            np.kron(np.expand_dims(col_mean, 0), np.ones((50, 1, 1), np.uint8)),
        )
        exit()

        running_mean = np.float32(col_mean[0])
        mean_len = 2
        res = []
        for x in range(1, col_mean.shape[0] // 20):
            col = col_mean[x]
            print("col =", col)
            print("running_mean =", np.uint8(running_mean))
            diff = np.abs(running_mean - col)
            if diff.max() > 60:
                print("\t----\t----", np.uint8(running_mean), "\n")
                res.append(running_mean)
                running_mean = col
                mean_len = 2
            print("diff =", np.uint8(diff))
            running_mean = (mean_len - 1) / mean_len * running_mean + 1 / mean_len * col
            print("factors:", (mean_len - 1), "/", mean_len, " ;", 1, "/", mean_len)
            print("\t->", np.uint8(running_mean))
            print()
            mean_len += 1
        res = np.vstack(res)
        res = np.expand_dims(res, 0).astype(np.uint8)
        show_imgs(
            mean=col_mean, median=col_median, logpolar=logpolar, res=res, block=False
        )
        return

    def get_radial_line_points(img: np.ndarray, cy: int, cx: int) -> list[float]:
        thetas = np.arange(0, np.pi, np.pi / 10)
        thetas += np.deg2rad(1)  # this ugly, but necessary, trust me.

        # cv2.destroyAllWindows()

        rays = []
        for theta in thetas:
            line_img = draw_polar_line_through_point(
                np.zeros((img.shape[:2]), np.uint8), (cy, cx), theta, thickness=1
            )
            ys, xs = np.nonzero(line_img)

            # find center point idx in line
            breaking = False
            # in case the line does not go directly through the center point,
            # we look around the point to try and find it
            sequence = [
                (i // 2) * (1 if i % 2 == 0 else -1) for i in range(1, 10)
            ]  # 0, 1, -1, 2, -2
            for dy in sequence:
                for dx in sequence:
                    center_idx = np.where((ys == cy + dy) & (xs == cx + dx))
                    if len(center_idx) == 0 or len(center_idx[0]) == 0:
                        continue
                    center_idx = center_idx[0][0]
                    breaking = True
                    break
                if breaking:
                    break

            # first portion
            ys_start = ys[:center_idx][::-1]
            xs_start = xs[:center_idx][::-1]
            ys_end = ys[center_idx:]
            xs_end = xs[center_idx:]

            if theta > np.pi / 2:
                ys_start, ys_end = ys_end, ys_start
                xs_start, xs_end = xs_end, xs_start

            rays.append(img[ys_start, xs_start])
            rays.append(img[ys_end, xs_end])
        rays = rays[::2] + rays[1::2]
        res = np.zeros((500, max(len(r) for r in rays), 3), np.uint8)
        for i, ray in enumerate(rays):
            for j, c in enumerate(ray):
                res[i * 50 : (i + 1) * 50, j] = c
        show_imgs(res, block=False)
        # exit()

    def edge_detect_(img: np.ndarray) -> np.ndarray:
        def _detect(img):
            # Blur + filter in x and y
            img = cv2.GaussianBlur(img, (3, 3), 0)
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

            # Take abs value to include both positive and negative extremas
            grad_x = abs(grad_x)
            grad_y = abs(grad_y)

            grad = np.maximum(grad_x, grad_y)

            grad = cv2.convertScaleAbs(grad)
            return grad

        # Ensure grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur
        img = cv2.blur(img, (7, 7))
        img = cv2.blur(img, (7, 7))

        # Increase contrast
        img = cv2.convertScaleAbs(img, alpha=2.0, beta=0)

        # Detect edges
        edges = cv2.Canny(img, 0, 255)

        return edges

    def get_line_angles_old(lines: list[tuple[float, float, float, float, float]]):
        line_lengths = [l[2] for l in lines]
        line_angles = [l[-1] for l in lines]

        # Smooth line angles
        from scipy.signal import savgol_filter

        line_angles_smoothed = savgol_filter(line_angles, window_length=5, polyorder=2)

        # Fort lines by bins
        angle_step = np.deg2rad(4.5)
        bins = np.arange(0, np.pi + angle_step, angle_step)
        bin_indices = np.digitize(line_angles_smoothed, bins, right=False)

        angle_bins = [0 for _ in range(len(bins))]
        for line, bin_idx in zip(lines, bin_indices):
            line_length = line[2]
            line_rho = line[-2]
            line_theta = line[-1]

            p1_dist = Utils.point_point_dist(line[0], (cy, cx))
            p2_dist = Utils.point_point_dist(line[1], (cy, cx))
            min_dist = min(p1_dist, p2_dist)

            draw_polar_line(img, line_rho, line_theta, intensity=line_length)

            # TODO: soft binning
            # angle_bins[bin_idx - 2] += line_length * min_dist * abs((target_angle - angle) / angle_step)  # deadzone, falloff
            angle_bins[bin_idx - 1] += line_length * min_dist

        max_bin = max(angle_bins)
        for val in angle_bins:
            val /= max_bin
            val *= 30
            val = int(val)
            print("#" * val)
        return
