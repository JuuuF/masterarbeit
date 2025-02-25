import os
import cv2
import sys
import pickle
import subprocess
import numpy as np
import tensorflow as tf


_base_interpreter = os.path.join(
    os.path.expanduser("~"),
    "anaconda3/envs/ma_deepdarts/bin/python",
)
output_file = os.path.join(
    os.path.dirname(__file__),
    "dd_inference.pkl",
)

running_as_dd = len(sys.argv) > 1

# =================================================================================================


class DeepDartsCode:

    REPO_PATH = "data/paper/deep-darts-master/"
    MODELS_PATH = "data/paper/models/"
    DATA_PATH = os.path.join(REPO_PATH, "dataset", "cropped_images", "800")

    d1_val = [
        "d1_02_06_2020",
        "d1_02_16_2020",
        "d1_02_22_2020",
    ]
    d1_test = [
        "d1_03_03_2020",
        "d1_03_19_2020",
        "d1_03_23_2020",
        "d1_03_27_2020",
        "d1_03_28_2020",
        "d1_03_30_2020",
        "d1_03_31_2020",
    ]

    d2_val = [
        "d2_02_03_2021",
        "d2_02_05_2021",
    ]
    d2_test = [
        "d2_03_03_2020",
        "d2_02_10_2021",
        "d2_02_03_2021_2",
    ]

    BOARD_DICT = {
        0: 13,
        1: 4,
        2: 18,
        3: 1,
        4: 20,
        5: 5,
        6: 12,
        7: 9,
        8: 14,
        9: 11,
        10: 8,
        11: 16,
        12: 7,
        13: 19,
        14: 3,
        15: 17,
        16: 2,
        17: 15,
        18: 10,
        19: 6,
    }

    # --------------------------------------------------------------------
    # Util Functions
    def get_splits(path="./dataset/labels.pkl", dataset="d1", split="train"):
        assert dataset in ["d1", "d2"], "dataset must be either 'd1' or 'd2'"
        assert split in [
            None,
            "train",
            "val",
            "test",
        ], "split must be in [None, 'train', 'val', 'test']"

        if dataset == "d1":
            val_folders, test_folders = DeepDartsCode.d1_val, DeepDartsCode.d1_test
        else:
            val_folders, test_folders = DeepDartsCode.d2_val, DeepDartsCode.d2_test

        df = pd.read_pickle(path)
        df = df[df.img_folder.str.contains(dataset)]
        splits = {}
        splits["val"] = df[np.isin(df.img_folder, val_folders)]
        splits["test"] = df[np.isin(df.img_folder, test_folders)]
        splits["train"] = df[
            np.logical_not(np.isin(df.img_folder, val_folders + test_folders))
        ]
        if split is None:
            return splits
        else:
            return splits[split]

    def bboxes_to_xy(bboxes, max_darts=3):
        xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
        for cls in range(5):
            if cls == 0:
                dart_xys = bboxes[bboxes[:, 4] == 0, :2][:max_darts]
                xy[4 : 4 + len(dart_xys), :2] = dart_xys
            else:
                cal = bboxes[bboxes[:, 4] == cls, :2]
                if len(cal):
                    xy[cls - 1, :2] = cal[0]
        xy[(xy[:, 0] > 0) & (xy[:, 1] > 0), -1] = 1
        if np.sum(xy[:4, -1]) == 4:
            return xy
        else:
            xy = DeepDartsCode.est_cal_pts(xy)
        return xy

    def est_cal_pts(xy):
        missing_idx = np.where(xy[:4, -1] == 0)[0]
        if len(missing_idx) == 1:
            if missing_idx[0] <= 1:
                center = np.mean(xy[2:4, :2], axis=0)
                xy[:, :2] -= center
                if missing_idx[0] == 0:
                    xy[0, 0] = -xy[1, 0]
                    xy[0, 1] = -xy[1, 1]
                    xy[0, 2] = 1
                else:
                    xy[1, 0] = -xy[0, 0]
                    xy[1, 1] = -xy[0, 1]
                    xy[1, 2] = 1
                xy[:, :2] += center
            else:
                center = np.mean(xy[:2, :2], axis=0)
                xy[:, :2] -= center
                if missing_idx[0] == 2:
                    xy[2, 0] = -xy[3, 0]
                    xy[2, 1] = -xy[3, 1]
                    xy[2, 2] = 1
                else:
                    xy[3, 0] = -xy[2, 0]
                    xy[3, 1] = -xy[2, 1]
                    xy[3, 2] = 1
                xy[:, :2] += center
        else:
            # TODO: if len(missing_idx) > 1
            # print("Missed more than 1 calibration point")
            return xy * 0
        return xy

    def board_radii(r_d):
        r_board = 0.2255  # radius of full board
        r_double = 0.170  # center bull to outside double wire edge, in m (BDO standard)
        r_treble = (
            0.1074  # center bull to outside treble wire edge, in m (BDO standard)
        )
        r_outer_bull = 0.0159
        r_inner_bull = 0.00635
        w_double_treble = 0.01  # wire apex to apex for double and treble

        r_t = r_d * (r_treble / r_double)  # outer treble radius, in px
        r_ib = r_d * (r_inner_bull / r_double)  # inner bull radius, in px
        r_ob = r_d * (r_outer_bull / r_double)  # outer bull radius, in px
        w_dt = w_double_treble * (r_d / r_double)  # width of double and treble
        return r_t, r_ob, r_ib, w_dt

    def get_circle(xy):
        """
        Calculate board size and center based on orientation points
        c = center position
        r = board radius
        """
        c = np.mean(xy[:4], axis=0)
        r = np.mean(np.linalg.norm(xy[:4] - c, axis=-1))
        return c, r

    def transform(xy, img=None, angle=9, M=None):

        if xy.shape[-1] == 3:
            has_vis = True
            vis = xy[:, 2:]
            xy = xy[:, :2]
        else:
            has_vis = False

        if img is not None and np.mean(xy[:4]) < 1:
            h, w = img.shape[:2]
            xy *= [[w, h]]

        if M is None:
            c, r = DeepDartsCode.get_circle(xy)  # not necessarily a circle
            # c is center of 4 calibration points, r is mean distance from center to calibration points

            src_pts = xy[:4].astype(np.float32)
            dst_pts = np.array(
                [
                    [
                        c[0] - r * np.sin(np.deg2rad(angle)),
                        c[1] - r * np.cos(np.deg2rad(angle)),
                    ],
                    [
                        c[0] + r * np.sin(np.deg2rad(angle)),
                        c[1] + r * np.cos(np.deg2rad(angle)),
                    ],
                    [
                        c[0] - r * np.cos(np.deg2rad(angle)),
                        c[1] + r * np.sin(np.deg2rad(angle)),
                    ],
                    [
                        c[0] + r * np.cos(np.deg2rad(angle)),
                        c[1] - r * np.sin(np.deg2rad(angle)),
                    ],
                ]
            ).astype(np.float32)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        xyz = np.concatenate((xy, np.ones((xy.shape[0], 1))), axis=-1).astype(
            np.float32
        )
        xyz_dst = np.matmul(M, xyz.T).T
        xy_dst = xyz_dst[:, :2] / xyz_dst[:, 2:]

        if img is not None:
            img = cv2.warpPerspective(img.copy(), M, (img.shape[1], img.shape[0]))
            xy_dst /= [[w, h]]

        if has_vis:
            xy_dst = np.concatenate([xy_dst, vis], axis=-1)

        return xy_dst, img, M

    def get_dart_scores(
        xy,  # (7, 3): 4x orientation + <= 3x dart; (x, y, visible)
        numeric=False,
        combined=True,  # both letters and numberic values
    ):
        def output_score(score):
            if combined:
                return scores
            if numeric:
                return [s[0] for s in scores]
            return [s[1] for s in scores]

        scores = [(0, "HIDDEN") for _ in range(3)]
        positions = [np.zeros((2,), np.float32) for _ in range(3)]
        M_undistort = np.eye(3)

        valid_cal_pts = xy[:4][(xy[:4, 0] > 0) & (xy[:4, 1] > 0)]
        if xy.shape[0] <= 4 or valid_cal_pts.shape[0] < 4:  # missing calibration point
            return False, output_score(scores), positions, M_undistort

        # Undistort positions based on orientation points
        xy_undist, _, M_undistort = DeepDartsCode.transform(xy.copy(), angle=0)

        # Get board radii from orientation points
        c, r_d = DeepDartsCode.get_circle(xy)
        r_t, r_ob, r_ib, w_dt = DeepDartsCode.board_radii(r_d)

        # Extract polar coordinates from positions
        xy = xy_undist - c
        angles = np.arctan2(-xy[4:, 1], xy[4:, 0]) / np.pi * 180
        angles = [a + 360 if a < 0 else a for a in angles]  # map to 0-360
        distances = np.linalg.norm(xy[4:], axis=-1)

        # Map scores to polar coordinates
        for i, (angle, dist) in enumerate(zip(angles, distances)):
            if dist > r_d:
                scores[i] = (0, "OUT")
                continue
            if dist <= r_ib:
                scores[i] = (50, "DB")
                continue
            if dist <= r_ob:
                scores[i] = (25, "B")
                continue

            number = DeepDartsCode.BOARD_DICT[int(angle / 18)]

            if dist <= r_d and dist > r_d - w_dt:
                scores[i] = (2 * number, f"D{number}")
                continue
            if dist <= r_t and dist > r_t - w_dt:
                scores[i] = (3 * number, f"T{number}")
                continue
            scores[i] = (number, str(number))

        # numeric_scores = [None for _ in scores]
        # if numeric or combined:
        #     for i, s in enumerate(scores):
        #         if "B" in s:
        #             if "D" in s:
        #                 numeric_scores[i] = 50
        #                 continue
        #             numeric_scores[i] = 25
        #             continue
        #         if "D" in s or "T" in s:
        #             numeric_scores[i] = int(s[1:])
        #             numeric_scores[i] = (
        #                 numeric_scores[i] * 2 if "D" in s else numeric_scores[i] * 3
        #             )
        #             continue
        #         numeric_scores[i] = int(s)

        return True, output_score(scores), xy_undist, M_undistort

    # --------------------------------------------------------------------
    # Drawing

    def draw_circles(img, xy, color=(255, 255, 255)):
        c, r_d = DeepDartsCode.get_circle(xy)  # double radius
        center = (int(round(c[0])), int(round(c[1])))
        r_t, r_ob, r_ib, w_dt = DeepDartsCode.board_radii(r_d)
        for r in [r_d, r_d - w_dt, r_t, r_t - w_dt, r_ib, r_ob]:
            cv2.circle(img, center, int(round(r)), color)
        return img

    def draw(img, xy, circles, score, color=(255, 255, 0)):
        # Read data
        xy = np.array(xy)
        if xy.shape[0] > 7:
            xy = xy.reshape((-1, 2))

        # Rescale outputs to pixel values
        if np.mean(xy) < 1:
            h, w = img.shape[:2]
            xy[:, 0] *= w
            xy[:, 1] *= h

        # Draw board circles
        if xy.shape[0] >= 4 and circles:
            img = DeepDartsCode.draw_circles(img, xy)

        # Get scores
        if xy.shape[0] > 4 and score:
            scores = DeepDartsCode.get_dart_scores(xy)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        line_type = 1
        for i, [x, y] in enumerate(xy):
            if i < 4:
                c = (0, 255, 0)  # green
            else:
                c = color  # cyan
            x = int(round(x))
            y = int(round(y))
            if i >= 4:
                cv2.circle(img, (x, y), 1, c, 1)
                if score:
                    txt = str(scores[i - 4])
                else:
                    txt = str(i + 1)
                cv2.putText(img, txt, (x + 8, y), font, font_scale, c, line_type)
            else:
                cv2.circle(img, (x, y), 1, c, 1)
                cv2.putText(img, str(i + 1), (x + 8, y), font, font_scale, c, line_type)
        return img

    # --------------------------------------------------------------------
    # Actual Inference

    def load_model(config="deepdarts_d1"):
        from yacs.config import CfgNode
        from yolov4.tf import YOLOv4
        from yolov4.model.yolov4 import YOLOv4Tiny

        # Prepare config
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(
            os.path.join(DeepDartsCode.REPO_PATH, "configs", config + ".yaml")
        )
        cfg.model.name = config

        # Build model
        yolo = YOLOv4(tiny=True)  # cfg.model.tiny = True
        yolo.classes = os.path.join(
            DeepDartsCode.REPO_PATH, "classes"
        )  # {0: 'dart', 1: 'cal1', 2: 'cal2', 3: 'cal3', 4: 'cal4'}
        yolo.input_size = (cfg.model.input_size, cfg.model.input_size)  # (800, 800)
        yolo.batch_size = cfg.train.batch_size  # 4 / 16

        # Make model
        yolo._has_weights = False
        inputs = tf.keras.layers.Input(
            [yolo.input_size[1], yolo.input_size[0], 3]
        )  # (800, 800, 3)
        yolo.model = YOLOv4Tiny(
            anchors=yolo.anchors,
            num_classes=len(yolo.classes),
            xyscales=yolo.xyscales,
            activation="leaky",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        )
        yolo.model(inputs)

        # Load weights
        weights_path = os.path.join(DeepDartsCode.MODELS_PATH, config, "weights")

        yolo.model.load_weights(weights_path)

        # with open(f"data/ai/paper_model/{config}.pkl", "wb") as f:
        #     pickle.dump(yolo)

        return yolo

    def predict(model, img_paths: list = None):
        from time import time

        ma_outputs = {
            f: {
                "undistortion_homography": np.eye(3),
                "dart_positions": np.zeros((3, 2)),
                "scores": [0 for _ in range(3)],
                "success": False,
            }
            for f in img_paths
        }
        # Load data
        if img_paths is None:
            data = DeepDartsCode.get_splits(
                path="data/paper/labels.pkl",
                dataset="d1",
                split="val",  # train / val / test
            )  # columns: img_folder, img_name, bbox, xy

            data = data[:10]
            # data = data[:1000]

            img_paths = [
                os.path.join(DeepDartsCode.DATA_PATH, folder, name)
                for folder, name in zip(data.img_folder, data.img_name)
            ]

            # Extract relevant data
            """
            xys: (n, 7, 3)
                n: amount of samples
                7: 4x orientation + 3x arrow
                3: x, y, visible
            """
            xys = np.zeros((len(data), 7, 3))
            data.xy = data.xy.apply(np.array)  # convert xy to numpy array
            for i, _xy in enumerate(data.xy):
                xys[i, : _xy.shape[0], :2] = _xy
                xys[i, : _xy.shape[0], 2] = 1
            xys = xys.astype(np.float32)  # (n, 7, 3)
        else:
            xys = np.zeros(
                (len(img_paths), 7, 3), np.float32
            )  # targets are zero for new images

        # Predict images
        preds = np.zeros((len(img_paths), 4 + 3, 3))
        for i, p in enumerate(img_paths):
            print(f"{i+1}/{len(img_paths)}", end="\r")
            # Start timer at image 2, probably to skip initialization
            if i == 1:
                ti = time()

            # Read in image
            img = cv2.imread(p)
            if img is None:
                print("ERROR: Could not load image from file:", p)
                continue

            # Predict
            bboxes = model.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            preds[i] = DeepDartsCode.bboxes_to_xy(
                bboxes
            )  # (7, 3): 4x orientation + 3x dart, [x, y, visibile]

        if i > 0:
            fps = (len(img_paths) - 1) / (time() - ti)
            print(f"FPS: {fps:.2f}")

        scores = []
        mses = []
        custom_losses = []
        for i, (y_pred, y_true, path) in enumerate(zip(preds, xys, img_paths)):
            # Calculate score
            _, true_scores, _, _ = DeepDartsCode.get_dart_scores(
                y_true[:, :2], numeric=True
            )
            success, pred_scores, pos_pred, M_pred = DeepDartsCode.get_dart_scores(
                y_pred[:, :2], combined=True
            )
            ma_outputs[path]["undistortion_homography"] = M_pred
            ma_outputs[path]["dart_positions"] = 800 * pos_pred[4:]
            ma_outputs[path]["scores"] = pred_scores
            ma_outputs[path]["success"] = success

            score = abs(
                sum([s[0] for s in pred_scores]) - sum([s[0] for s in true_scores])
            )
            scores.append(score)

            mse = np.mean((y_true - y_pred) ** 2)
            mses.append(mse)
            # print(
            #     f"Target score: {target_scores} | Prediction Score: {pred_scores} | {path}",
            #     " " * 10,
            #     end="\r",
            # )

            # Show image
            # xy = preds[i]
            # xy = xy[xy[:, -1] == 1]
            # res = DeepDartsCode.draw(
            #     cv2.imread(path),
            #     xy[:, :2],
            #     circles=True,
            #     score=True,
            # )
            # cv2.imshow("Pred", res)
            # if cv2.waitKey() == ord("q"):
            #     break

        # print(" " * (len(str(target_scores) + str(pred_scores) + path) + 40), end="\r")
        ASE = np.array(scores)  # absolute score errors
        PCS = len(ASE[ASE == 0]) / len(ASE) * 100
        MASE = np.mean(ASE)

        MSE = np.mean(mses)

        print(f"Percent Correct Score (PCS): {PCS:.1f}%")
        print(f"Mean Absolute Score Error (MASE): {MASE:.2f}")
        print(f"Mean Squared Error (MSE): {MSE}")
        with open(output_file, "wb") as f:
            pickle.dump(ma_outputs, f)


def _dd_inference(img_paths: list) -> np.ndarray:

    # Load model
    print("Loading model...")
    yolo = DeepDartsCode.load_model(config="deepdarts_d1")
    print("Predicting...")
    DeepDartsCode.predict(yolo, img_paths=img_paths)

    for img_path in img_paths:
        pass
    return "resulting things"


def inference_deepdarts(
    img_paths: str,  # or list[str]
    interpreter_path: str = None,
) -> np.ndarray:

    if type(img_paths) == str:
        img_paths = [img_paths]

    img_paths = img_paths[:10]

    if interpreter_path is None:
        interpreter_path = _base_interpreter

    res = subprocess.run(
        [interpreter_path, __file__, *img_paths],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if res.returncode != 0:
        raise RuntimeError(res)

    with open(output_file, "rb") as f:
        ma_outputs = pickle.load(f)
    return ma_outputs


if __name__ == "__main__":
    if running_as_dd:
        img_paths = sys.argv[1:]
        _dd_inference(img_paths)
        exit()

    img_dir = "data/darts_references/jess"
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    ma_outputs = inference_deepdarts(img_paths)
    print(ma_outputs)
