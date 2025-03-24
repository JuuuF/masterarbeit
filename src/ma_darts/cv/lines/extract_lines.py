import cv2
import numpy as np

from ma_darts.cv.utils import show_imgs, points_to_polar_line


def extract_lines(
    img: np.ndarray,
    rho: int = 1,
    theta: float = np.deg2rad(1),
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
            *points_to_polar_line(x[0], x[1]),  # rho [-n..n], theta [0..π)
        ),
        lines,
    )
    lines = list(lines)

    if show:  # or create_debug_img:  # TODO: debug image
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
                3,
                lineType=cv2.LINE_AA,
            )
            color = np.random.randint(128) + 128
            # cv2.circle(img, p1, 4, (color, 0, 0), lineType=cv2.LINE_AA)
            # cv2.circle(img, p2, 4, (color, 0, 0), lineType=cv2.LINE_AA)
        out = cv2.addWeighted(img, 0.25, line_img, 1, 1.0)
        if show:
            show_imgs(lines=out, block=False)
        # Utils.append_debug_img(out, "Found Lines")  # TODO: debug image

    return lines  # (p1, p2, length, rho, theta)
