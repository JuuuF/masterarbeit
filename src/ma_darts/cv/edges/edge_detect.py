import cv2
import numpy as np

from ma_darts.cv.utils import show_imgs


def get_sobel(k: int, sigma: float = None):
    if sigma is None:
        sigma = k

    # Create coordinate grid
    ax = np.arange(-k, k + 1)
    xx, yy = np.meshgrid(ax, ax)

    # Calculate the Gaussian function
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # Optionally include the normalization constant 1/(2*pi*sigma^2)
    # Comment out the next line if you prefer to omit the constant.
    kernel /= 2 * np.pi * sigma**2

    kernel[k] = 0
    kernel[k:] *= -1

    return kernel


def edge_detect(
    img: np.ndarray,
    kernel_size: int = 5,
    show: bool = False,
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
    filter_x = get_sobel(7)
    sobel_x = cv2.filter2D(img, -1, filter_x)
    sobel_y = cv2.filter2D(img, -1, filter_x.T)

    # combine gradients
    sobel_img = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges = np.uint8(sobel_img / sobel_img.max() * 255)

    _, edges = cv2.threshold(sobel_edges, 100, 255, cv2.THRESH_BINARY)

    # show_imgs(img=img, sobel_x=sobel_x, sobel_y=sobel_y, sobel_edges=sobel_edges, edges=edges, block=False)
    if show:
        show_imgs(edges=edges, block=False)

    # Utils.append_debug_img(edges, "Edge Detection")  # TODO: debug image handling
    return edges
