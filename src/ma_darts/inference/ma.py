import cv2
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from ma_darts import r_do, r_di, r_to, r_ti, r_bo, r_bi
from ma_darts.cv.cv import undistort_img
from ma_darts.cv.utils import show_imgs
from ma_darts.ai.models import yolo_v8_predict

wrong_img = None


def get_wrong_image() -> np.ndarray:
    global wrong_img
    if wrong_img is not None:
        return wrong_img

    # Red base image
    img = np.zeros((800, 800, 3), np.uint8)
    img[:, :, 2] = 255
    for i, r in enumerate([r_do, r_di, r_to, r_ti, r_bo, r_bi]):
        col = (255, 255, 255) if i % 2 != 0 else (0, 0, 0)
        cv2.circle(img, (400, 400), int(r), col, 2, cv2.LINE_AA)

    cv2.putText(
        img,
        "Could not undistort the image.",
        (0, 24),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=2.0,
        color=(255, 255, 255),
        thickness=1,
    )
    wrong_img = img
    return wrong_img


def inference_ma(
    img_paths: str | list[str],
    model_path: str | None = None,
    model: tf.keras.Model | None = None,
) -> list[np.ndarray]:

    iter_fn = tqdm
    if type(img_paths) != list:
        img_paths = [img_paths]
        iter_fn = lambda x: x

    ma_outputs = {
        f: {
            "undistortion_homography": np.eye(3),
            "dart_positions": np.zeros((3, 2)),
            "scores": [0 for _ in range(3)],
            "success": False,
        }
        for f in img_paths
    }

    # -------------------------------------------
    # Load Model
    print("Loading model...")
    if model_path is None and model is None:
        raise ValueError(
            "Both 'model_path' and 'model' are None. "
            "Please specify any one of these."
        )
    if model is None:
        model = tf.keras.models.load_model(model_path)
 
    for img_path in iter_fn(img_paths):
        # -------------------------------------------
        # Load Image
        print("Loading image...")
        img = cv2.imread(img_path)

        # -------------------------------------------
        # Undistortion
        print("Aligning image...")
        img_aligned, homography = undistort_img(img)  # (800, 800, 3)
        if img_aligned is None:
            # TODO: handle error case
            print("Could not undistort image.")
            while max(*img.shape[:2]) > 1600:
                img = cv2.pyrDown(img)
            return img

        # -------------------------------------------
        # Locate Darts
        print("Locating darts...")
        img_input = np.expand_dims(img_aligned, 0)  # (1, 800, 800, 3)
        img_input = np.float32(img_aligned) / 255
        pred = yolo_v8_predict(model, img_input)  # (800, 800, 3)

    return pred


if __name__ == "__main__":
    import os
    from itertools import zip_longest

    img_dir = "data/darts_references/jess"
    img_paths_jess = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    img_dir = "data/generation/out_val"
    img_paths_gen = [
        os.path.join(img_dir, f, "render.png") for f in os.listdir(img_dir)
    ]
    img_paths = [
        x
        for pair in zip_longest(img_paths_jess, img_paths_gen)
        for x in pair
        if x is not None
    ]

    model = tf.keras.models.load_model("data/ai/darts_model.keras", compile=False)
    model.load_weights("data/ai/darts/yolov8_train6.weights.h5")

    for img_path in img_paths:
        res = inference_ma(img_path, model=model)
        show_imgs(res)
