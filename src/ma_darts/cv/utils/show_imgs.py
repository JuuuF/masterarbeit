import os
import cv2
import numpy as np


def _display_and_save(
    name: str,
    img: np.ndarray,
    out_dir: str = "dump/display_out",
) -> None:
    # Show image
    cv2.imshow(name, img)

    # Save image
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(os.path.join(out_dir, name + ".png"), img)


def show_imgs(
    *imgs: list[np.ndarray],
    block: bool = True,
    **named_imgs: dict[str, np.ndarray],
) -> None:
    for i, img in enumerate(imgs):
        _display_and_save(f"img_{i}", img)
    for name, img in named_imgs.items():
        _display_and_save(name, img)

    if not block:
        return

    ignore_keys = [
        3,  # AltGr
        9,  # Tab
        225,  # LShift
        226,  # RShift
        227,  # Alt + Strg
        228,  # LShift + RShift
        229,  # Caps-Lock
        231,  # RShift + Alt
        233,  # Alt
    ]
    quit_keys = [
        ord("q"),
        27,  # Esc
    ]
    while (key := cv2.waitKey()) in ignore_keys:
        continue

    if key in quit_keys:
        cv2.destroyAllWindows()
        exit()
    return chr(key)
