import os
import cv2
import json
import pickle
import numpy as np
import pandas as pd

from itertools import zip_longest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from ma_darts.inference import inference_deepdarts, inference_ma, visualize_prediction
from ma_darts.cv.utils import show_imgs, apply_matrix, homography_similarity


d1_val = ["d1_02_06_2020", "d1_02_16_2020", "d1_02_22_2020"]
d1_test = ["d1_03_03_2020", "d1_03_19_2020", "d1_03_23_2020", "d1_03_27_2020", "d1_03_28_2020", "d1_03_30_2020", "d1_03_31_2020"]  # fmt: skip

d2_val = ["d2_02_03_2021", "d2_02_05_2021"]
d2_test = ["d2_03_03_2020", "d2_02_10_2021", "d2_02_03_2021_2"]


def get_img_paths(
    jess: bool = True,
    home: bool = True,
    strongbows: bool = True,
    deepdarts: bool = True,
    val: bool = True,
    interleave: bool = True,
):
    path_list = []

    # Jess data (real)
    if jess:
        img_dir = "data/darts_references/jess"
        img_paths_jess = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        path_list.append(img_paths_jess)
        # return img_paths_jess

    # Home data (real)
    if home:
        img_dir = "data/darts_references/home"
        img_paths_home = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        path_list.append(img_paths_home)
        # return img_paths_home

    # Strongbow's data (real)
    if strongbows:
        img_dir = "data/darts_references/strongbows_out"
        img_paths_sb = [
            os.path.join(img_dir, f, "render.png") for f in os.listdir(img_dir)
        ]
        path_list.append(img_paths_sb)
        # return img_paths_sb

    # Paper data (real)
    if deepdarts:
        dirs = []
        # dirs += d1_val
        # dirs += d1_test
        dirs += d2_val
        # dirs += d2_test
        img_dir = "data/paper/imgs/d2_03_03_2020"
        img_paths_dd = []
        for d in dirs:
            d = os.path.join("data/paper/imgs", d)
            img_paths_dd += [os.path.join(d, f) for f in os.listdir(d)]
        path_list.append(img_paths_dd)
        return img_paths_dd

    # Validation data (generated)
    if val:
        img_dir = "data/generation/out"
        img_paths_val = [
            os.path.join(img_dir, f, "render.png") for f in os.listdir(img_dir)
        ]
        path_list.append(img_paths_val)
        return img_paths_val

    # Combine it all
    if interleave:
        img_paths = [
            x for pair in zip_longest(*path_list) for x in pair if x is not None
        ]
    else:
        img_paths = [x for path in path_list for x in path]
    return img_paths


def get_target_homographies(img_paths):
    out = {}
    dd_labels = None
    r_sin_t = 300 * np.sin(np.deg2rad(9))
    r_cos_t = 300 * np.cos(np.deg2rad(9))
    dst_pts = np.array(
        [
            [400 - r_sin_t, 400 - r_cos_t],
            [400 + r_sin_t, 400 + r_cos_t],
            [400 - r_cos_t, 400 + r_sin_t],
            [400 + r_cos_t, 400 - r_sin_t],
        ]
    )

    for img_path in img_paths:
        # MA system
        if img_path.endswith("render.png"):
            info_path = os.path.join(os.path.dirname(img_path), "info.json")
            with open(info_path, "r") as f:
                sample_info = dict(json.load(f))
            M = np.array(sample_info["undistortion_homography"])
            out[img_path] = M
            continue

        # DeepDarts system
        dd_labels_path = os.path.abspath(os.path.join(img_path, "../../../labels.pkl"))
        if os.path.exists(dd_labels_path):
            # Load labels
            if dd_labels is None:
                with open(dd_labels_path, "rb") as f:
                    dd_labels = pickle.load(f)
                img_dir = os.path.relpath(
                    os.path.join(os.path.dirname(dd_labels_path), "imgs")
                )
                dd_labels.index = (
                    img_dir
                    + "/"
                    + dd_labels["img_folder"]
                    + "/"
                    + dd_labels["img_name"]
                )
            src_pts = np.array(dd_labels.loc[img_path, "xy"][:4]) * 800
            M, _ = cv2.findHomography(src_pts, dst_pts)
            out[img_path] = M
            continue

        raise ValueError(
            f"I don't know where to locate homographies for images in path {img_path}"
        )

    return out


if __name__ == "__main__":

    # --------------------------------------------------------------------
    # Load Image Paths

    img_paths = get_img_paths(interleave=True)
    img_paths = img_paths[:1000]
    print(img_paths)
    target_homographies = get_target_homographies(img_paths)

    # --------------------------------------------------------------------
    # Get MA Results
    from ma_darts.ai.models import yolo_v8_model

    model = yolo_v8_model(variant="s")
    model.load_weights("data/ai/darts/yolov8_train7.weights.h5")

    ma_outputs = inference_ma(img_paths, model=model, confidence_threshold=0.0)

    # --------------------------------------------------------------------
    # Get DeepDarts Results
    deepdarts_model = "deepdarts_d1"
    dd1_outputs = inference_deepdarts(img_paths, model_config=deepdarts_model)
    deepdarts_model = "deepdarts_d2"
    dd2_outputs = inference_deepdarts(img_paths, model_config=deepdarts_model)

    # for img_path, res in dd_outputs.items():
    #     img = visualize_prediction(img_path, res)
    #     show_imgs(img)

    # for img_path in img_paths:
    #     res = inference_ma(
    #         img_path, model=model, max_outputs=None, confidence_threshold=0.0
    #     )
    #     img = visualize_prediction(img_path, res)
    #     show_imgs(img)

    # --------------------------------------------------------------------
    # Evaluate CV results

    dd1_similarities = []
    dd2_similarities = []
    ma_similarities = []
    n_outs_dd1 = 0
    n_outs_dd2 = 0
    n_outs_ma = 0
    for img_path in img_paths:
        homography_true = target_homographies[img_path]

        # DeepDarts d1
        if dd1_outputs[img_path]["success"]:
            homography_dd1 = dd1_outputs[img_path]["undistortion_homography"]
            similarity_dd1 = homography_similarity(homography_true, homography_dd1)
            dd1_similarities.append(similarity_dd1)
            # img_dd = visualize_prediction(img_path, dd_outputs[img_path])
            # show_imgs(deepdarts_inference=img_dd, block=False)
        else:
            n_outs_dd1 += 1

        # DeepDarts d2
        if dd2_outputs[img_path]["success"]:
            homography_dd2 = dd2_outputs[img_path]["undistortion_homography"]
            similarity_dd2 = homography_similarity(homography_true, homography_dd2)
            dd2_similarities.append(similarity_dd2)
            # img_dd = visualize_prediction(img_path, dd_outputs[img_path])
            # show_imgs(deepdarts_inference=img_dd, block=False)
        else:
            n_outs_dd2 += 1

        if ma_outputs[img_path]["success"]:
            homography_ma = ma_outputs[img_path]["undistortion_homography"]
            similarity_ma = homography_similarity(homography_true, homography_ma)
            ma_similarities.append(similarity_ma)
            # img_ma = visualize_prediction(img_path, ma_outputs[img_path])
            # show_imgs(ma_inference=img_ma, block=False)
        else:
            n_outs_ma += 1

        # print(f"DD similarity: {str(round(similarity_dd, 2)) + 'px' if similarity_dd is not None else 'non-existing'}")
        # print(f"MA similarity: {str(round(similarity_ma, 2)) + 'px' if similarity_ma is not None else 'non-existing'}")
        # show_imgs()
    # cv2.destroyAllWindows()

    mean_similarity_dd1 = (
        np.mean(dd1_similarities) if len(dd1_similarities) > 0 else None
    )
    mean_similarity_dd2 = (
        np.mean(dd2_similarities) if len(dd2_similarities) > 0 else None
    )
    mean_similarity_ma = np.mean(ma_similarities) if len(ma_similarities) > 0 else None

    print(f"DD1 outputs: {len(dd1_similarities)}/{len(img_paths)}")
    print(f"Mean DD1 similarity: {mean_similarity_dd1}")
    print(f"DD2 outputs: {len(dd2_similarities)}/{len(img_paths)}")
    print(f"Mean DD2 similarity: {mean_similarity_dd2}")
    print(f"MA outputs: {len(ma_similarities)}/{len(img_paths)}")
    print(f"Mean MA similarity: {mean_similarity_ma}")
