import cv2
import numpy as np

from ma_darts.cv.utils import show_imgs, draw_polar_line


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

    if show:  # or create_debug_img:  # TODO: debug img
        acc = np.uint8(np.float32(acc) / acc.max() * 255)
        acc = cv2.cvtColor(acc, cv2.COLOR_GRAY2BGR)
        cv2.circle(acc, (cx, cy), 10, (255, 0, 0), lineType=cv2.LINE_AA)
        if show:
            show_imgs(center_point=acc, block=False)
        # Utils.append_debug_img(acc, "Center Point")  # TODO: debug img
    return cy, cx
