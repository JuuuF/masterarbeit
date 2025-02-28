import cv2
import numpy as np

from ma_darts.cv.utils import show_imgs


def skeletonize(img: np.ndarray, show: bool = False) -> np.ndarray:
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
    # Utils.append_debug_img(skeleton, "Skeletonized Edges")  # TODO: debug img
    return skeleton
