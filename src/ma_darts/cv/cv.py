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

create_debug_img = True
debug_out_images = []

if __name__ == "__main__":
    img_paths_custom = [
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
        "/home/justin/Downloads/test2.jpg",
        "/home/justin/Downloads/test.jpg",
    ]

    # Generated images
    img_paths_gen = [
        os.path.join("data/generation/out", i, "render.png")
        for i in sorted(os.listdir("data/generation/out"), key=lambda x: int(x))
    ]
    # img_paths = img_paths[46:]

    # Paper images
    img_paths_paper = [
        os.path.join("data/paper/imgs", d, f)
        for d in os.listdir("data/paper/imgs")
        for f in os.listdir(os.path.join("data/paper/imgs", d))
    ]
    np.random.shuffle(img_paths_paper)
    img_paths_paper = img_paths_paper[:200]

    # Own References
    img_paths_jess = [
        os.path.join("data/darts_references/jess", f)
        for f in os.listdir("data/darts_references/jess")
    ]

    img_paths = (
        []
        # add paths
        + img_paths_gen
        + img_paths_paper
        + img_paths_jess
    )
    np.random.shuffle(img_paths)

    img_paths[0] = "data/generation/out/75/render.png"


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

    def show_debug_img(
        target_w: int = 2560,
        target_h: int = 1080 - 125,
        failed: bool = False,
    ) -> None:
        if not create_debug_img:
            return
        global debug_out_images
        imgs = debug_out_images.copy()

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

        show_imgs(**{"Debug Output": res})

    def append_debug_img(img: np.ndarray, name: str = "<no name>") -> None:
        if not create_debug_img:
            return
        global debug_out_images
        debug_out_images.append((name, img))

    def clear_debug_img() -> None:
        global debug_out_images
        debug_out_images = []

    def display_line_peaks(acc, acc_smooth):

        acc /= acc_smooth.max() * 1.5
        colors_base = (
            cv2.applyColorMap(
                np.uint8(acc_smooth.reshape(-1, 1, 1) * 255), cv2.COLORMAP_SUMMER
            ).astype(np.float32)
            / 4
        )
        colors_base = np.uint8(colors_base)

        acc_smooth /= acc_smooth.max() * 1.5
        colors = cv2.applyColorMap(
            np.uint8(acc_smooth.reshape(-1, 1, 1) * 255), cv2.COLORMAP_WINTER
        )

        out_base = np.zeros((360, 360, 3), np.uint8)
        for i, a in enumerate(acc):
            out_base[: round(a * 360), i] = colors_base[i]
        out_base = out_base[::-1]

        out = np.zeros((360, 360, 3), np.uint8)
        for i, a in enumerate(acc_smooth):
            out[: round(a * 360), i] = colors[i]
        out = out[::-1]

        out = cv2.addWeighted(out, 1.0, out_base, 0.75, 1.0)
        show_imgs(out)


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

        Utils.append_debug_img(edges, "Edge Detection")
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
        Utils.append_debug_img(skeleton, "Skeletonized Edges")
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

        if show or create_debug_img:
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
            Utils.append_debug_img(out, "Found Lines")

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

        # out = np.zeros((img_shape[0], img_shape[1], 3), np.float32)
        # colors = [
        #     (166, 206, 227),
        #     (31, 120, 180),
        #     (178, 223, 138),
        #     (51, 160, 44),
        #     (251, 154, 153),
        #     (227, 26, 28),
        #     (253, 191, 111),
        #     (255, 127, 0),
        #     (202, 178, 214),
        #     (106, 61, 154),
        # ]
        # colors = [
        #     (180, 119, 31),
        #     (14, 127, 255),
        #     (44, 160, 44),
        #     (40, 39, 214),
        #     (189, 103, 148),
        #     (75, 86, 140),
        #     (194, 119, 227),
        #     (127, 127, 127),
        #     (34, 189, 188),
        #     (207, 190, 23),
        # ]
        # for i, bin_img in enumerate(bin_imgs):
        #     bin_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        #     col = colors[i]  # [int(((c / 255)**1.2)*255) for c in colors[i]]
        #     bin_img = np.float32(bin_img * col) / 255
        #     out += bin_img
        # out /= out.max()
        # out = cv2.blur(out, (3, 3))
        # out = np.uint8(out * 255)
        # show_imgs(out)

        if show or create_debug_img:
            acc = np.uint8(np.float32(acc) / acc.max() * 255)
            acc = cv2.cvtColor(acc, cv2.COLOR_GRAY2BGR)
            cv2.circle(acc, (cx, cy), 10, (255, 0, 0), lineType=cv2.LINE_AA)
            if show:
                show_imgs(center_point=acc, block=False)
            Utils.append_debug_img(acc, "Center Point")
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

        # Utils.display_line_peaks(accumulator, acc_smooth)

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

        if show or create_debug_img:
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
            Utils.append_debug_img(res, "Field-Separating Lines")

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

        if show or create_debug_img:
            out = img.copy()
            for rho, theta in out_lines:
                draw_polar_line(out, rho, theta)
            if show:
                show_imgs(lines_aligned=out, block=False)
            Utils.append_debug_img(out, "Aligned Angles")
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

        if show or create_debug_img:
            global img
            res = apply_matrix(img, M, True)
            if show:
                show_imgs(img_undistort=res, block=False)
            Utils.append_debug_img(res, "Undistorted Angles")

        return M

    def find_orientation_points(
        img: np.ndarray,
        cy: int,
        cx: int,
        show: bool = False,
    ) -> list[list[tuple[int, str]]]:
        def get_logpolar(img: np.ndarray, max_r: int, cy: int, cx: int) -> np.ndarray:
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
            return logpolar

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

        def get_black_white_and_center_size(
            logpolar: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
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

            return white, black, field_start

        def find_logpolar_corners(img):
            # This looks weird, but is intended
            # Convert RGB -> Lab -> Gray
            # This increases the contrast for corners
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.GaussianBlur(img, (7, 7), 0)

            block_size = 2
            kernel_size = 5
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
            corners = np.uint8(corners * 255)

            return corners

        def to_cryv(patch: np.ndarray) -> np.ndarray:

            # black / white different
            # black / red+green different
            # white / red+green different
            # -> red/green similar

            YCrCb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
            Cr = YCrCb[:, :, 1:2]  # white / green, red / green
            Y = YCrCb[:, :, 0:1]  # black / red, black / white, white / green

            HSV = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            V = HSV[:, :, 2:3]  # black / red

            res = np.concatenate(
                [
                    Cr,
                    Y,
                    V,
                ],
                axis=-1,
            )
            return res

        def is_black(patch, black_cryv):
            b_Cr, b_Y, b_V = black_cryv
            Cr, Y, V = patch

            diff_Cr = np.abs(b_Cr - Cr)
            diff_Y = np.abs(b_Y - Y)
            diff_V = np.abs(b_V - V)

            return round(diff_Cr + diff_Y + diff_V)

        def is_white(patch, white_cryv):
            w_Cr, w_Y, w_V = np.float32(white_cryv)
            Cr, Y, V = patch

            diff_Cr = np.abs(w_Cr - Cr)
            diff_Y = np.abs(w_Y - Y)
            diff_V = np.abs(w_V - V)

            return round(diff_Cr + diff_Y + diff_V)

        def is_color(patch):
            Cr, _Y, _V = patch
            target_Cr_red = 200.0
            target_Cr_green = 30.0

            diff_Cr_red = np.abs(target_Cr_red - Cr)
            diff_Cr_green = np.abs(target_Cr_green - Cr)
            diff_Cr = min(diff_Cr_red, diff_Cr_green)

            return round(diff_Cr)

        def show_cryv(img, name=None):

            full = []
            for i in range(img.shape[-1]):
                full.append(img[:, :, i])
            res = np.hstack(full)
            if name is not None:
                show_imgs(**{name: res}, block=False)
            else:
                show_imgs(res, block=False)

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

        # Convert image to logpolar representation
        max_r = max(
            Utils.point_point_dist((cy, cx), (0, 0)),
            Utils.point_point_dist((cy, cx), (0, img.shape[1])),
            Utils.point_point_dist((cy, cx), (img.shape[0], 0)),
            Utils.point_point_dist((cy, cx), (img.shape[0], img.shape[1])),
        )
        logpolar = get_logpolar(img, max_r, cy, cx)
        if show:
            show_imgs(logpolar=logpolar, block=False)

        # -----------------------------
        # Get Colors and Rough Scaling
        white, black, center_size = get_black_white_and_center_size(logpolar)

        # -----------------------------
        # Stretch logpolar image to ensure the multiplier fields are big enough
        width_scaling = max(1, 25 / center_size) if center_size > 1 else 1
        if width_scaling > 1:
            img_resized = cv2.resize(
                img,
                (int(width_scaling * img.shape[1]), int(width_scaling * img.shape[0])),
            )
            logpolar = get_logpolar(
                img_resized,
                max_r * width_scaling,
                int(width_scaling * cy),
                int(width_scaling * cx),
            )

        # -----------------------------
        # Find corners
        corners = find_logpolar_corners(logpolar)  # (1000, x)

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
            corners_ = cv2.addWeighted(
                logpolar, 0.5, cv2.cvtColor(corners_, cv2.COLOR_GRAY2BGR), 1.0, 1.0
            )
            show_imgs(recognized_corners=corners_, block=False)

        search_distance = 2
        search_width = 8

        # -----------------------------
        # Check points for surrounding color
        surrounding_width = 14
        middle_deadspace = 2
        color_threshold = 100
        intrude = (surrounding_width - middle_deadspace) // 2
        inner_ring_a = []
        inner_ring_b = []
        outer_ring_a = []
        outer_ring_b = []

        # Convert colors into a format that emphasizes the differences
        white_cryv = to_cryv(white[None, None])[0, 0]
        black_cryv = to_cryv(black[None, None])[0, 0]

        logpolar_cryv = to_cryv(logpolar)
        # show_cryv(logpolar_cryv, name="logpolar_cryv")

        for i, points in enumerate(corner_positions):
            y = 25 + 50 * i
            for x in points:
                surrounding = logpolar[
                    y - surrounding_width : y + surrounding_width,
                    x - surrounding_width : x + surrounding_width,
                ]
                surrounding_cryv = logpolar_cryv[
                    y - surrounding_width : y + surrounding_width,
                    x - surrounding_width : x + surrounding_width,
                ]
                # Find partial fields in surrounding area
                top_left = surrounding_cryv[:intrude, :intrude]
                top_right = surrounding_cryv[:intrude, -intrude:]
                bottom_left = surrounding_cryv[-intrude:, :intrude]
                bottom_right = surrounding_cryv[-intrude:, -intrude:]
                # Extract mean colors from fields
                color_top_left = top_left.mean(axis=0).mean(axis=0)
                color_top_right = top_right.mean(axis=0).mean(axis=0)
                color_bottom_left = bottom_left.mean(axis=0).mean(axis=0)
                color_bottom_right = bottom_right.mean(axis=0).mean(axis=0)
                # Determine field colors
                top_left_black = is_black(color_top_left, black_cryv) < color_threshold
                top_left_white = is_white(color_top_left, white_cryv) < color_threshold
                top_left_color = is_color(color_top_left) < color_threshold
                top_right_black = (
                    is_black(color_top_right, black_cryv) < color_threshold
                )
                top_right_white = (
                    is_white(color_top_right, white_cryv) < color_threshold
                )
                top_right_color = is_color(color_top_right) < color_threshold
                bottom_left_black = (
                    is_black(color_bottom_left, black_cryv) < color_threshold
                )
                bottom_left_white = (
                    is_white(color_bottom_left, white_cryv) < color_threshold
                )
                bottom_left_color = is_color(color_bottom_left) < color_threshold
                bottom_right_black = (
                    is_black(color_bottom_right, black_cryv) < color_threshold
                )
                bottom_right_white = (
                    is_white(color_bottom_right, white_cryv) < color_threshold
                )
                bottom_right_color = is_color(color_bottom_right) < color_threshold

                # --------------------- #
                # DEBUGGING
                # print()
                # print("top_left:", f"black: {is_black(color_top_left, black_cryv)}", f"white: {is_white(color_top_left, white_cryv)}", f"color: {is_color(color_top_left)}", sep="\n\t")  # fmt: skip
                # print("top_right:", f"black: {is_black(color_top_right, black_cryv)}", f"white: {is_white(color_top_right, white_cryv)}", f"color: {is_color(color_top_right)}", sep="\n\t")  # fmt: skip
                # print("bottom_left:", f"black: {is_black(color_bottom_left, black_cryv)}", f"white: {is_white(color_bottom_left, white_cryv)}", f"color: {is_color(color_bottom_left)}", sep="\n\t")  # fmt: skip
                # print("bottom_right:", f"black: {is_black(color_bottom_right, black_cryv)}", f"white: {is_white(color_bottom_right, white_cryv)}", f"color: {is_color(color_bottom_right)}", sep="\n\t")  # fmt: skip
                # show_cryv(
                #     cv2.resize(
                #         surrounding_cryv,
                #         (
                #             surrounding_cryv.shape[1] * 4,
                #             surrounding_cryv.shape[0] * 4,
                #         ),
                #         interpolation=cv2.INTER_NEAREST,
                #     ),
                #     name="surrounding_cryv",
                # )
                # in-depth debugging
                # show_imgs(surrounding=surrounding, block=False)
                # show_imgs()

                left_color = top_left_color and bottom_left_color
                right_color = top_right_color and bottom_right_color

                if top_left_black and bottom_left_white and right_color:
                    surrounding_normalized = surrounding
                    inner_ring_a.append((y, x, surrounding_normalized))

                if top_left_white and bottom_left_black and right_color:
                    surrounding_normalized = surrounding[::-1]  # flip vertical
                    inner_ring_b.append((y, x, surrounding_normalized))

                if top_right_black and bottom_right_white and left_color:
                    surrounding_normalized = surrounding[:, ::-1]  # flip horizontal
                    outer_ring_a.append((y, x, surrounding_normalized))

                if top_right_white and bottom_right_black and left_color:
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
        if len(outer_ring_a + outer_ring_b) < 2:
            print(
                "ERROR: Not enough valid orientation points found (possibly too many outliers)."
            )
            if create_debug_img:
                logpolar[corners != 0] = 255
                Utils.append_debug_img(
                    logpolar, "FAILED: Not enough orientation points found."
                )
            return None
        if len(surroundings) == 0:
            print("ERROR: No valid surroundings found.")
            if create_debug_img:
                logpolar[corners != 0] = 255
                Utils.append_debug_img(
                    logpolar, "FAILED: No orientation point surroundings found."
                )
            return None
        mean_surrounding = np.median(surroundings, axis=0).astype(np.uint8)
        if show:
            show_imgs(mean_surrounding=mean_surrounding, block=False)

        # -----------------------------
        # Classify surroundings
        prepare_show_img = show or create_debug_img
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

        if prepare_show_img:
            surrounding_preview = cv2.resize(
                mean_surrounding,
                (mean_surrounding.shape[1] * 3, mean_surrounding.shape[0] * 3),
                interpolation=cv2.INTER_NEAREST,
            )
            logpolar_[
                : surrounding_preview.shape[0], -surrounding_preview.shape[1] :
            ] = surrounding_preview
            if show:
                show_imgs(positions=logpolar_, block=False)
            Utils.append_debug_img(logpolar_, "Logpolar Orientation Points")

        # -----------------------------
        # Sort keeps into bins
        positions: list[list[tuple[int, str]]] = y_to_angle_bin(keeps)
        # Resolve logpolar bins to real bins
        # Logpolar distortion starts at 3 o'clock while we start rotation at 12
        positions = positions[5:] + positions[:5]
        for i in range(5):
            positions[i] = [(-p[0], p[1]) for p in positions[i]]

        # -----------------------------
        # Re-Scale positions
        if width_scaling > 1:
            for i, pos_bin in enumerate(positions):
                positions[i] = [(int(p[0] / width_scaling), p[1]) for p in pos_bin]
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

        # TODO: correct radii
        print("TODO: CV.structure_orientation_candidates: Get board radii")
        r_triple_inner = 97 * 2
        r_triple_outer = 107 * 2
        r_double_inner = 160 * 2

        src = []
        dst = []

        prepare_show_img = show or create_debug_img
        if prepare_show_img:
            img = img_undistort.copy()

        for i, angle_positions in enumerate(orientation_point_candidates):
            theta = np.pi / 20 + i * np.pi / 10
            double_threshold = double_thresholds[i]
            for r, pos in angle_positions:
                src_y = cy - np.cos(theta) * r
                src_x = cx + np.sin(theta) * r
                if pos == "outer":
                    # triple ring - outside
                    dst_r = r_triple_outer * (1 if r > 0 else -1)
                elif abs(r) > double_threshold:
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
                        c = (0, 0, 255)
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
            threshold_points = [
                (int(cx + r * np.sin(theta)), int(cy - r * np.cos(theta)))
                for r, theta in zip(
                    double_thresholds, np.arange(0, 2 * np.pi, np.pi / 10) + np.pi / 20
                )
            ]
            for i, p in enumerate(threshold_points):
                p2 = threshold_points[(i + 1) % len(threshold_points)]
                cv2.line(img, p, p2, color=(127, 127, 127))
        if show:
            show_imgs(projection_mapping=img, block=False)
        Utils.append_debug_img(img, "Orientation Point Projections")

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
        Utils.append_debug_img(img, f"Input: {f}")

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
            if create_debug_img:
                Utils.show_debug_img(failed=True)
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

        # angle_step = np.pi / 10
        # angles = np.arange(0, np.pi, angle_step) + angle_step / 2
        # img_undistort //= 2
        # for t in angles:
        #     draw_polar_line_through_point(img_undistort, (cy_undistort, cx_undistort), t)
        # show_imgs(img_undistort)
        # exit()

        # Find possible orientation points
        orientation_point_candidates = Orientation.find_orientation_points(
            img_undistort, int(cy_undistort), int(cx_undistort), show=False
        )
        if orientation_point_candidates is None:
            if create_debug_img:
                Utils.show_debug_img(failed=True)
            continue

        # Filter out bad orientation points
        src_pts, dst_pts = Orientation.structure_orientation_candidates(
            orientation_point_candidates,
            int(cy_undistort),
            int(cx_undistort),
            show=False,
        )

        # Convert orientation points to transformation matrix
        M_align = Orientation.get_alignment_matrix(
            src_pts, dst_pts, int(cy_undistort), int(cx_undistort)
        )

        # Combine all matrices
        scale = img.shape[0] / img_full.shape[0]

        M_full = np.eye(3)
        M_full = scaling_matrix(scale) @ M_full  # downscale to calculation size
        M_full = M_undistort @ M_full  # undistort
        M_full = M_align @ M_full  # align to correct scale and orientation

        res = apply_matrix(img_full, M_full)
        res = res[:800, :800]
        cv2.circle(res, (400, 400), 10, (255, 255, 255), 2)
        cv2.circle(res, (400, 400), 3, (255, 0, 0), -1)

        Utils.append_debug_img(res, "Aligned Image")
        Utils.show_debug_img()
        Utils.clear_debug_img()
