import cv2
import numpy as np

from ma_darts.cv.utils import apply_matrix


def visualize_prediction(
    img_path: str,
    res: dict,
) -> np.ndarray:

    # Load image
    img = cv2.imread(img_path)

    # Undistort image
    img = apply_matrix(img, res["undistortion_homography"], output_size=(800, 800))

    def add_text(txt, pos):
        txt_params = dict(
            img=img,
            text=txt,
            org=pos,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.25,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            **txt_params,
            thickness=2,
            color=(255, 255, 255),
        )
        cv2.putText(
            **txt_params,
            thickness=1,
            color=(0, 0, 0),
        )

    pos = res["dart_positions"]
    scr = res["scores"]
    cnf = res["confidences"]
    for (y, x), (scr_val, scr_str), c in zip(pos, scr, cnf):
        y, x = int(y), int(x)

        # Draw circle at position
        cv2.circle(img, (x, y), 2, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, y), 3, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, y), 4, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        add_text(scr_str, pos=(x + 5, y - 2))
        add_text(f"({round(c * 100)}%)", pos=(x + 5, y + 8))
    return img
