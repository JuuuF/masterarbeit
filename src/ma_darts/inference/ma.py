import cv2
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from time import time

from ma_darts import radii
from ma_darts.cv.cv import undistort_img
from ma_darts.cv.utils import show_imgs, apply_matrix
from ma_darts.ai.utils import yolo_v8_predict
from ma_darts.inference import visualize_prediction


def inference_ma(
    img_paths: str | list[str],
    model_path: str | None = None,
    model: tf.keras.Model | None = None,
    confidence_threshold: float = 0.5,
    max_outputs: int = 3,
    verbose: bool = False,
) -> list[np.ndarray]:

    added_batch = False
    iter_fn = tqdm
    if type(img_paths) != list:
        img_paths = [img_paths]
        iter_fn = lambda x: x
        added_batch = True

    ma_outputs = {
        f: {
            "undistortion_homography": np.eye(3),
            "dart_positions": [],
            "confidences": [],
            "scores": [],
            "success": False,
        }
        for f in img_paths
    }

    # -------------------------------------------
    # Load Model
    # print("Loading model...")
    if model_path is None and model is None:
        raise ValueError(
            "Both 'model_path' and 'model' are None. "
            "Please specify any one of these."
        )
    if model is None:
        model = tf.keras.models.load_model(model_path)
    model.predict(np.zeros((1, 800, 800, 3)), verbose=0)

    start = time()
    cv_times = []
    ai_times = []
    for img_path in iter_fn(img_paths):
        cv_start = time()
        # -------------------------------------------
        # Load Image
        img = cv2.imread(img_path)

        # -------------------------------------------
        # Undistortion
        homography = undistort_img(img)  # (800, 800, 3)

        if homography is None:
            ma_outputs[img_path]["success"] = False
            print("Could not undistort image.")
            while max(*img.shape[:2]) > 1600:
                img = cv2.pyrDown(img)
            cv_times.append(time() - cv_start)
            continue

        ma_outputs[img_path]["undistortion_homography"] = homography
        img_aligned = apply_matrix(img, homography, output_size=(800, 800))
        cv_times.append(time() - cv_start)

        # -------------------------------------------
        # Locate Darts
        ai_start = time()
        img_input = np.expand_dims(img_aligned, 0)  # (1, 800, 800, 3)
        img_input = np.float32(img_aligned) / 255
        outputs = yolo_v8_predict(
            model, img_input, confidence_threshold=0.1
        )  # [((y, x), (score_int, score_str), conf)] * n_preds

        # Write up to 3 outputs
        n_outputs = len(outputs["dart_positions"])
        if max_outputs is not None and max_outputs > 0:
            n_outputs = min(max_outputs, n_outputs)
        for i in range(n_outputs):
            pos = outputs["dart_positions"][i]
            scr = outputs["scores"][i]
            cnf = outputs["confidences"][i]
            ma_outputs[img_path]["dart_positions"].append(pos)
            if cnf >= confidence_threshold:
                ma_outputs[img_path]["scores"].append(scr)
                ma_outputs[img_path]["confidences"].append(cnf)
            else:
                ma_outputs[img_path]["scores"].append((0, "HIDDEN"))
                ma_outputs[img_path]["confidences"].append(1 - cnf)

        ma_outputs[img_path]["success"] = True
        ai_times.append(time() - ai_start)

    # Timing
    dt = time() - start
    sample_time = dt / len(img_paths)
    if verbose:
        print("-" * 50)
        print(f"MA inference: {dt:.03f}s -> {sample_time:.03f}s/sample")
        print(f"\tCV time: {np.sum(cv_times):.03f}s -> {np.mean(cv_times):.03f}s/sample")
        print(f"\tAI time: {np.sum(ai_times):.03f}s -> {np.mean(ai_times):.03f}s/sample")
        print("-" * 50)
    if added_batch:
        return ma_outputs[img_paths[0]]
    return ma_outputs


if __name__ == "__main__":
    import os
    from itertools import zip_longest

    img_dir = "data/darts_references/jess"
    img_paths_jess = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    img_dir = "data/generation/out_val"
    img_paths_gen = [
        os.path.join(img_dir, f, "render.png") for f in os.listdir(img_dir)
    ]
    img_dir = "data/paper/imgs/d2_03_03_2020"
    img_paths_dd = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    img_dir = "data/darts_references/home"
    img_paths_home = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]

    img_paths = [
        x
        for pair in zip_longest(
            img_paths_jess, img_paths_gen, img_paths_dd, img_paths_home
        )
        for x in pair
        if x is not None
    ]

    # model = tf.keras.models.load_model("data/ai/darts_model.keras", compile=False)
    from ma_darts.ai.models import yolo_v8_model

    model = yolo_v8_model(variant="n")
    model.load_weights("data/ai/darts/latest.weights.h5")
    # model = tf.keras.models.load_model("dump/trains/run_1/darts_model.keras")

    # img_paths = img_paths[:10]
    # ma_outputs = inference_ma(
    #     img_paths, model=model, max_outputs=None, confidence_threshold=0.0
    # )
    # for file, res in ma_outputs.items():
    #     print()
    #     print(file)
    #     if not res["success"]:
    #         print("\tNo result.")
    #         continue
    #     for k, v in res.items():
    #         print(f"\t{k}: {v}")
    # exit()

    for img_path in img_paths:
        ma_outputs = inference_ma(
            img_path,
            model=model,
            max_outputs=None,
            confidence_threshold=0.35,
            verbose=False,
        )
        img = visualize_prediction(img_path, ma_outputs)
        img = cv2.resize(img, (1000, 1000))
        show_imgs(img)
