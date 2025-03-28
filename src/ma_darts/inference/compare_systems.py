import os
import cv2
import json
import pickle
import numpy as np
import pandas as pd

from itertools import zip_longest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from ma_darts.inference import inference_deepdarts, inference_ma, visualize_prediction
from ma_darts.cv.utils import show_imgs, apply_matrix, homography_similarity

from ma_darts.inference.deepdarts import DeepDartsCode


d1_val = ["d1_02_06_2020", "d1_02_16_2020", "d1_02_22_2020"]
d1_test = ["d1_03_03_2020", "d1_03_19_2020", "d1_03_23_2020", "d1_03_27_2020", "d1_03_28_2020", "d1_03_30_2020", "d1_03_31_2020"]  # fmt: skip

d2_val = ["d2_02_03_2021", "d2_02_05_2021"]
d2_test = ["d2_03_03_2020", "d2_02_10_2021", "d2_02_03_2021_2"]


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
            M = np.array(sample_info.get("undistortion_homography"))
            out[img_path] = M if M.any() else None
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

        print(
            f"WARNING: I don't know where to locate homographies for images in path {img_path}. Skipping."
        )
        out[img_path] = None

    return out


def inference_on_image_paths(img_paths: list[str]):
    # Get MA results
    print("=" * 60)
    print("MA inference".center(60))
    print("=" * 60)

    from ma_darts.ai.models import yolo_v8_model

    model = yolo_v8_model(variant="n")
    # model.load_weights("data/ai/darts/train_13/yolov8_train_13_latest.weights.h5")
    model.load_weights("data/ai/darts/latest.weights.h5")
    ma_outputs = inference_ma(img_paths, model=model, confidence_threshold=0.0)

    # Get DeepDarts results
    print("=" * 60)
    print("DeepDarts d1 inference".center(60))
    print("=" * 60)
    dd1_outputs = inference_deepdarts(img_paths, model_config="deepdarts_d1")
    print("=" * 60)
    print("DeepDarts d2 inference".center(60))
    print("=" * 60)
    dd2_outputs = inference_deepdarts(img_paths, model_config="deepdarts_d2")

    # print("-" * 200)
    # print(ma_outputs)
    # print("-" * 200)
    # print(dd1_outputs)
    # print("-" * 200)
    # print(dd2_outputs)
    # exit()

    return ma_outputs, dd1_outputs, dd2_outputs


def get_img_paths_dict() -> dict[str, list[str]]:
    img_paths_dict = {}
    ignore_cv_names = []

    # -------------------------------------------
    # Test data

    img_dir = "data/generation/out_test"
    img_paths_dict["MA-test-gen"] = [
        os.path.join(img_dir, i, "render.png") for i in os.listdir(img_dir)
    ]

    # -------------------------------------------
    # Home

    img_dir = "data/darts_references/home_out"
    img_paths_dict["MA-test-real"] = [
        os.path.join(img_dir, i, "render.png") for i in os.listdir(img_dir)
    ]
    ignore_cv_names.append("MA-test-real")

    # -------------------------------------------
    # Jess

    img_dir = "data/darts_references/jess_out"
    img_paths_dict["MA-val-real"] = [
        os.path.join(img_dir, i, "render.png") for i in os.listdir(img_dir)
    ]
    ignore_cv_names.append("MA-val-real")

    # -------------------------------------------
    # DeepDarts d1_val

    img_dirs = [os.path.join("data/paper/imgs", d) for d in d1_val]
    img_paths_dict["DD-val-d1"] = [
        os.path.join(img_dir, f) for img_dir in img_dirs for f in os.listdir(img_dir)
    ]

    # -------------------------------------------
    # DeepDarts d1_test

    img_dirs = [os.path.join("data/paper/imgs", d) for d in d1_test]
    img_paths_dict["DD-test-d1"] = [
        os.path.join(img_dir, f) for img_dir in img_dirs for f in os.listdir(img_dir)
    ]

    # -------------------------------------------
    # DeepDarts d2_val

    img_dirs = [os.path.join("data/paper/imgs", d) for d in d2_val]
    img_paths_dict["DD-val-d2"] = [
        os.path.join(img_dir, f) for img_dir in img_dirs for f in os.listdir(img_dir)
    ]

    # -------------------------------------------
    # DeepDarts d2_test

    img_dirs = [os.path.join("data/paper/imgs", d) for d in d2_test]
    img_paths_dict["DD-test-d2"] = [
        os.path.join(img_dir, f) for img_dir in img_dirs for f in os.listdir(img_dir)
    ]

    return img_paths_dict, ignore_cv_names


def get_cv_score(target_homographies: dict, outputs: dict) -> float | None:
    similarities = []
    invalid_homographies = []
    for img_path in outputs.keys():
        # Check validity
        if not outputs[img_path]["success"]:
            invalid_homographies.append(img_path)
            continue

        # Get homographies
        M_true = target_homographies[img_path]
        M_pred = outputs[img_path]["undistortion_homography"]

        # Check validity again
        if M_pred is None:
            invalid_homographies.append(img_path)
            continue

        # Compare similarity
        similarity = homography_similarity(M_true, M_pred)
        similarities.append(similarity)

    # Calculate mean
    mean_similarity = np.mean(similarities) if len(similarities) > 0 else None
    valid_score = 1 - len(invalid_homographies) / max(len(outputs.keys()), 1)
    return mean_similarity, 100 * valid_score


def get_ai_targets(img_paths):
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
        if img_path.endswith("render.png") or img_path.endswith("undistort.jpg"):
            info_path = os.path.join(os.path.dirname(img_path), "info.json")
            with open(info_path, "rb") as f:
                sample_info = dict(json.load(f))
            M = sample_info.get("undistortion_homography")
            out[img_path] = {
                "dart_positions": np.array(sample_info["dart_positions_undistort"])
                * 800,
                "scores": sample_info["scores"],
                "undistortion_homography": np.array(M) if M is not None else np.eye(3),
            }

            continue

        # DD system
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
            # Get homography
            points = np.array(dd_labels.loc[img_path, "xy"])
            success, scores, dart_pos, M = DeepDartsCode.get_dart_scores(
                points, combined=True
            )
            out[img_path] = {
                "dart_positions": np.array(dart_pos[4:, ::-1]) * 800,
                "scores": scores,
                "undistortion_homography": M,
            }
            continue

        print(
            f"WARNING: I don't know where to locate infos for images in path {img_path}. Skipping."
        )
        out[img_path] = None

    return out


def greedy_match(pos_true, pos_pred, score_true, score_pred):

    pos_visible_true = [p for s, p in zip(score_true, pos_true) if s[1] != "HIDDEN"]
    pos_visible_pred = [p for s, p in zip(score_pred, pos_pred) if s[1] != "HIDDEN"]

    matches = []

    while pos_visible_true and pos_visible_pred:
        min_dist = float("inf")
        best_idx_true = None
        best_idx_pred = None

        for i, pos_t in enumerate(pos_visible_true):
            for j, pos_p in enumerate(pos_visible_pred):
                dist = np.linalg.norm(pos_t - pos_p)
                if dist < min_dist:
                    min_dist = dist
                    best_idx_true = i
                    best_idx_pred = j

        matches.append(
            (pos_visible_true[best_idx_true], pos_visible_pred[best_idx_pred], min_dist)
        )
        pos_visible_true.pop(best_idx_true)
        pos_visible_pred.pop(best_idx_pred)
    return matches


def add_text(img, txt, pos, col):
    txt_params = dict(
        img=img,
        text=txt,
        org=pos,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        **txt_params,
        thickness=2,
        color=col,
    )
    cv2.putText(
        **txt_params,
        thickness=1,
        color=(0, 0, 0),
    )


def get_ai_scores(targets: dict, outputs: dict) -> float:
    xst_scores = []
    field_scores = []
    pos_scores = []
    is_correct_scores = []
    correct_existences = []
    correct_fields = []
    PCS = []
    for img_path in outputs.keys():
        # Extract information
        pos_pred = outputs[img_path]["dart_positions"]
        pos_true = targets[img_path]["dart_positions"]
        score_pred = outputs[img_path]["scores"]
        score_true = targets[img_path]["scores"]

        # Existences comparison
        n_true = len([s for s in score_true if s[1] != "HIDDEN"])
        n_pred = len([s for s in score_pred if s[1] != "HIDDEN"])
        xst_score = 1 - abs(n_true - n_pred) / 3
        xst_score *= 100
        xst_scores.append(xst_score)

        # Fields comparison
        score_pred_ = score_pred.copy()
        n_correct = 0
        for s_true in score_true:
            if s_true in score_pred_:
                score_pred_.remove(s_true)
                n_correct += 1
        field_score = n_correct / 3
        field_score *= 100
        field_scores.append(field_score)

        # Positions comparison
        matches = greedy_match(pos_true, pos_pred, score_true, score_pred)
        total_dist = 0
        for m in matches:
            total_dist += m[-1]
        pos_score = total_dist / max(len(matches), 1)
        pos_scores.append(pos_score)

        # PCS
        n_points_true = sum(s[0] for s in score_true)
        n_points_pred = sum(s[0] for s in score_pred)
        is_correct_score = n_points_true == n_points_pred
        is_correct_score *= 100
        is_correct_scores.append(is_correct_score)

        continue

        print()
        print(f"{xst_score=}")
        print(f"{field_score=}")
        print(f"{pos_score=}")
        print(f"{is_correct_score=}")

        # Show image
        img = cv2.imread(img_path)
        img = apply_matrix(
            img, targets[img_path]["undistortion_homography"], output_size=(800, 800)
        )
        for p, s in zip(pos_true, score_true):
            org = (int(p[1]), int(p[0]))
            cv2.circle(img, org, 4, (0, 255, 0), 2, cv2.LINE_AA)
            org = (int(p[1]), int(p[0]) + 10)
            add_text(img, s[1], org, (0, 255, 0))
        for p, s in zip(pos_pred, score_pred):
            org = (int(p[1]), int(p[0]))
            cv2.circle(img, org, 4, (255, 0, 0), 2, cv2.LINE_AA)
            add_text(img, s[1], org, (255, 0, 0))

        show_imgs(img)

    mean_xst_score = np.mean(xst_scores) if len(xst_scores) > 0 else 0
    mean_field_score = np.mean(field_scores) if len(field_scores) > 0 else 0
    mean_pos_score = np.mean(pos_scores) if len(pos_scores) > 0 else 0
    mena_correct_score = np.mean(is_correct_scores) if len(is_correct_scores) > 0 else 0

    return mean_xst_score, mean_field_score, mean_pos_score, mena_correct_score


if __name__ == "__main__":

    img_paths_dict, ignore_cv_names = get_img_paths_dict()
    results = []

    ws = [10, 20, 20, 20, 20, 20, 20, 20]
    headers = [
        "Model",
        "Sample time [s]",
        "CV similarity [px]",
        "CV valid [%]",
        "XST Score [%]",
        "Field Score [%]",
        "Pos Score [px]",
        "PCS [%]",
    ]

    def prepare_value(v):
        if type(v) == str:
            return v
        if v is None:
            return "N/A"
        v = np.round(v, 2)
        return str(v)

    header = "|".join([h.center(w) for h, w in zip(headers, ws)])

    # Iterate over each evaluation set
    for eval_set_name, img_paths in img_paths_dict.items():

        print("#" * 120)
        print(eval_set_name.center(120))
        print("#" * 120)

        # Get results
        outputs_ma, outputs_d1, outputs_d2 = inference_on_image_paths(img_paths)

        # Extract times
        sample_time_ma = outputs_ma.get("sample_time")
        sample_time_d1 = outputs_d1.get("sample_time")
        sample_time_d2 = outputs_d2.get("sample_time")
        if "sample_time" in outputs_ma.keys():
            del outputs_ma["sample_time"]
        if "sample_time" in outputs_d1.keys():
            del outputs_d1["sample_time"]
        if "sample_time" in outputs_d2.keys():
            del outputs_d2["sample_time"]

        # Check CV results
        if eval_set_name not in ignore_cv_names:
            print("=" * 60)
            print("Calculating CV score".center(60))
            print("=" * 60)
            target_homographies = get_target_homographies(img_paths)
            cv_out_ma, cv_valid_ma = get_cv_score(target_homographies, outputs_ma)
            cv_out_d1, cv_valid_d1 = get_cv_score(target_homographies, outputs_d1)
            cv_out_d2, cv_valid_d2 = get_cv_score(target_homographies, outputs_d2)
        else:
            print("=" * 60)
            print("!!!    No CV score    !!!".center(60))
            print("=" * 60)
            cv_out_ma, cv_valid_ma = None, 0.0
            cv_out_d1, cv_valid_d1 = None, 0.0
            cv_out_d2, cv_valid_d2 = None, 0.0

        # Get AI results
        print("=" * 60)
        print("Calculating AI score".center(60))
        print("=" * 60)
        ai_targets = get_ai_targets(img_paths)
        ai_xst_ma, ai_field_ma, ai_pos_ma, ai_pcs_ma = get_ai_scores(
            ai_targets, outputs_ma
        )
        ai_xst_d1, ai_field_d1, ai_pos_d1, ai_pcs_d1 = get_ai_scores(
            ai_targets, outputs_d1
        )
        ai_xst_d2, ai_field_d2, ai_pos_d2, ai_pcs_d2 = get_ai_scores(
            ai_targets, outputs_d2
        )

        res_ma = "|".join(
            [
                prepare_value(v).center(w)
                for v, w in zip(
                    [
                        "MA",
                        sample_time_ma,
                        cv_out_ma,
                        cv_valid_ma,
                        ai_xst_ma,
                        ai_field_ma,
                        ai_pos_ma,
                        ai_pcs_ma,
                    ],
                    ws,
                )
            ]
        )
        res_d1 = "|".join(
            [
                prepare_value(v).center(w)
                for v, w in zip(
                    [
                        "D1",
                        sample_time_d1,
                        cv_out_d1,
                        cv_valid_d1,
                        ai_xst_d1,
                        ai_field_d1,
                        ai_pos_d1,
                        ai_pcs_d1,
                    ],
                    ws,
                )
            ]
        )
        res_d2 = "|".join(
            [
                prepare_value(v).center(w)
                for v, w in zip(
                    [
                        "D2",
                        sample_time_d2,
                        cv_out_d2,
                        cv_valid_d2,
                        ai_xst_d2,
                        ai_field_d2,
                        ai_pos_d2,
                        ai_pcs_d2,
                    ],
                    ws,
                )
            ]
        )
        print(header)
        print(res_ma)
        print(res_d1)
        print(res_d2)
        results.append("-" * (sum(ws) + len(ws)))
        results.append(eval_set_name.center(sum(ws) + len(ws)))
        results.append("-" * (sum(ws) + len(ws)))
        results.append(header)
        results.append(res_ma)
        results.append(res_d1)
        results.append(res_d2)

    print("\n\n")
    print(*results, sep="\n")

    exit()
    # --------------------------------------------------------------------
    # Load Image Paths

    img_paths = get_img_paths(interleave=True)
    target_homographies = get_target_homographies(img_paths)

    # --------------------------------------------------------------------
    # Get outputs

    outputs_ma, dd1_outputs, dd2_outputs = inference_on_image_paths(img_files)

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

        if outputs_ma[img_path]["success"]:
            homography_ma = outputs_ma[img_path]["undistortion_homography"]
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
