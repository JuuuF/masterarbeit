import cv2
import numpy as np
from scipy.signal import find_peaks

from ma_darts.cv.utils import show_imgs


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

    out = cv2.addWeighted(out, 1.0, out_base, 1.0, 1.0)
    show_imgs(out)


def get_rough_line_angles(
    img_shape: tuple[int, int],
    lines: list[
        tuple[float, float, float, float, float]
    ],  # p1, p2, length (normalized), center distance [px], rho, theta
    cy: int,
    cx: int,
    show: np.ndarray | None = None,
):
    # Draw lines onto black canvas
    line_img = np.zeros(img_shape, np.uint8)
    for line in lines:
        cv2.line(line_img, line[0][::-1], line[1][::-1], (255, 255, 255), thickness=2)

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

    # display_line_peaks(accumulator, acc_smooth)

    if len(peak_values) > 10:
        # Remove smallest peaks
        peak_cutoff = sorted(peak_values)[-10]
        cutoff_indices = peak_values >= peak_cutoff
        peaks = peaks[cutoff_indices]
        peak_values = peak_values[cutoff_indices]
        peak_thetas = peak_thetas[cutoff_indices]
    elif len(peaks) < 10:
        # Interpolate peaks
        return []
        step_size = int(np.median(np.diff(peaks)))

        extended = []
        for i, peak in enumerate(peaks[:-1]):
            extended.append(peaks[i])
            gap = peaks[i + 1] - peak
            if gap > step_size * 1.5:
                n_missing = round(gap / step_size) - 1
                missing_values = [peak + step_size * (j + 1) for j in range(n_missing)]

                extended.extend(missing_values)
        extended.append(peaks[-1])

        # TODO: interpolate peak_thetas and peak_values for added values

    if show is not None:  # or create_debug_img:  # TODO: debug img
        from ma_darts.cv.utils import draw_polar_line_through_point
        res = show // 4
        for t, v in zip(peak_thetas, peak_values / peak_values.max()):
            intensity = v / 2 + 0.5
            draw_polar_line_through_point(
                res,
                (cy, cx),
                t,
                color=(255, 0, 0),
                intensity=intensity,
                thickness=2
            )
        for line in lines:
            cv2.line(
                res,
                line[0][::-1],
                line[1][::-1],
                color=(0, 255, 0),
                thickness=3,
                lineType=cv2.LINE_AA,
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
        if show is not None:
            show_imgs(field_separating_lines=res, block=False)
        # Utils.append_debug_img(res, "Field-Separating Lines")

    thetas = sorted(peak_thetas)
    return thetas
