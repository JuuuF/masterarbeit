import os
import cv2
import json
import pickle
import numpy as np

from itertools import zip_longest

from ma_darts.inference import inference_deepdarts, inference_ma, visualize_prediction
from ma_darts.cv.utils import show_imgs, apply_matrix, homography_similarity


def get_img_paths(
    jess: bool = True,
    home: bool = True,
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

    # Paper data (real)
    if deepdarts:
        img_dir = "data/paper/imgs/d2_03_03_2020"
        img_paths_dd = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        path_list.append(img_paths_dd)
        # return img_paths_dd[:5]

    # Validation data (generated)
    if val:
        img_dir = "data/generation/out"
        img_paths_val = [
            os.path.join(img_dir, f, "render.png") for f in os.listdir(img_dir)
        ]
        path_list.append(img_paths_val)
        return img_paths_val[50:300]

    # Combine it all
    if interleave:
        img_paths = [
            x for pair in zip_longest(*path_list) for x in pair if x is not None
        ]
    else:
        img_paths = [x for path in path_list for x in path]
    return img_paths


if __name__ == "__main__":

    # --------------------------------------------------------------------
    # Load Image Paths

    img_paths = get_img_paths(interleave=True)
    # img_paths = img_paths[-2:]

    # --------------------------------------------------------------------
    # Get DeepDarts Results
    deepdarts_model = "deepdarts_d2"
    dd_outputs = inference_deepdarts(img_paths, model_config=deepdarts_model)

    # for img_path, res in dd_outputs.items():
    #     img = visualize_prediction(img_path, res)
    #     show_imgs(img)

    # --------------------------------------------------------------------
    # Get DeepDarts Results
    from ma_darts.ai.models import yolo_v8_model

    model = yolo_v8_model(variant="s")
    model.load_weights("data/ai/darts/yolov8_train7.weights.h5")

    ma_outputs = inference_ma(img_paths, model=model, confidence_threshold=0.0)

    # for img_path in img_paths:
    #     res = inference_ma(
    #         img_path, model=model, max_outputs=None, confidence_threshold=0.0
    #     )
    #     img = visualize_prediction(img_path, res)
    #     show_imgs(img)

    # --------------------------------------------------------------------
    # Evaluate CV results

    dd_similarities = []
    ma_similarities = []
    n_outs_dd = 0
    n_outs_ma = 0
    for img_path in img_paths:
        if not img_path.endswith("render.png"):
            continue
        info_path = os.path.join(os.path.dirname(img_path), "info.json")
        if not os.path.exists(info_path):
            continue
        with open(info_path, "r") as f:
            sample_info = dict(json.load(f))
        homography_true = sample_info["undistortion_homography"]

        if dd_outputs[img_path]["success"]:
            homography_dd = dd_outputs[img_path]["undistortion_homography"]
            similarity_dd = homography_similarity(homography_true, homography_dd)
            dd_similarities.append(similarity_dd)
            # img_dd = visualize_prediction(img_path, dd_outputs[img_path])
            # show_imgs(deepdarts_inference=img_dd, block=False)
        else:
            n_outs_dd += 1

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

    mean_similarity_dd = np.mean(dd_similarities) if len(dd_similarities) > 0 else None
    mean_similarity_ma = np.mean(ma_similarities) if len(ma_similarities) > 0 else None

    print(f"DD outputs: {len(dd_similarities)}/{len(img_paths)}")
    print(
        f"Mean DD similarity: {mean_similarity_dd}"
    )
    print(f"MA outputs: {len(ma_similarities)}/{len(img_paths)}")
    print(
        f"Mean MA similarity: {mean_similarity_ma}"
    )
