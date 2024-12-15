import cv2
import numpy as np


def show_imgs(
    *imgs: list[np.ndarray],
    block: bool = True,
    **named_imgs: dict[str, np.ndarray],
) -> None:
    for i, img in enumerate(imgs):
        cv2.imshow(f"img_{i}", img)
    for name, img in named_imgs.items():
        cv2.imshow(name, img)

    if not block:
        return

    key = cv2.waitKey()
    if key == ord("q"):
        cv2.destroyAllWindows()
        exit()
