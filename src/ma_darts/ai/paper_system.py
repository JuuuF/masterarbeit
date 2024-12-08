import os
import cv2
import numpy as np
import tensorflow as tf

from time import time

REPO_PATH = "data/paper/deep-darts-master/"
MODELS_PATH = "data/paper/models/"
DATA_PATH = os.path.join(REPO_PATH, "dataset", "cropped_images", "800")


def load_model(config="deepdarts_d1"):

    # Prepare config
    from yacs.config import CfgNode

    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(os.path.join(REPO_PATH, "configs", config + ".yaml"))
    cfg.model.name = config

    # Build model
    from yolov4.tf import YOLOv4

    yolo = YOLOv4(tiny=True)  # cfg.model.tiny = True
    yolo.classes = os.path.join(REPO_PATH, "classes")
    yolo.input_size = (cfg.model.input_size, cfg.model.input_size)  # (800, 800)
    yolo.batch_size = cfg.train.batch_size  # 4 / 16

    # Make model
    from yolov4.model.yolov4 import YOLOv4Tiny

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
    weights_path = os.path.join(MODELS_PATH, config, "weights")

    yolo.model.load_weights(weights_path)

    return yolo


def predict(model, img_paths: list = None):
    from paper_code import get_splits, bboxes_to_xy, get_dart_scores, draw

    # Load data
    if img_paths is None:
        data = get_splits(
            path="data/paper/labels.pkl",
            dataset="d1",
            split="test",  # train / val / test
        )  # columns: img_folder, img_name, bbox, xy
        img_paths = [
            os.path.join(DATA_PATH, folder, name)
            for folder, name in zip(data.img_folder, data.img_name)
        ]

        img_paths = img_paths[:10]

        # Extract relevant data
        """
        xys: (n, 7, 3)
            n: amount of samples
            7: 4x orientation + 3x arrow
            3: x, y, visible
        """
        xys = np.zeros((len(data), 7, 3))
        data.xy = data.xy.apply(np.array)
        for i, _xy in enumerate(data.xy):
            xys[i, : _xy.shape[0], :2] = _xy
            xys[i, : _xy.shape[0], 2] = 1
        xys = xys.astype(np.float32)  # (n, 7, 3)
    else:
        xys = np.zeros((len(img_paths), 7, 3), np.float32)

    # Predict images
    preds = np.zeros((len(img_paths), 4 + 3, 3))
    for i, p in enumerate(img_paths):
        print(f"{i+1}/{len(img_paths)}", end="\r")
        # Start timer at image 2, probably to skip initialization
        if i == 1:
            ti = time()

        # Read in image
        img = cv2.imread(p)

        # Predict
        bboxes = model.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        preds[i] = bboxes_to_xy(bboxes)

    fps = (len(img_paths) - 1) / (time() - ti)
    print(f"FPS: {fps:.2f}")

    scores = []
    for i, (pred, gt, path) in enumerate(zip(preds, xys, img_paths)):
        # Calculate score
        target_scores = get_dart_scores(gt[:, :2], numeric=True)
        pred_scores = get_dart_scores(pred[:, :2], numeric=True)
        score = abs(sum(pred_scores) - sum(target_scores))
        scores.append(score)
        print(
            f"Target score: {target_scores} | Prediction Score: {pred_scores}",
            " " * 10,
            end="\r",
        )

        # Show image
        xy = preds[i]
        xy = xy[xy[:, -1] == 1]
        res = draw(
            cv2.imread(path),
            xy[:, :2],
            circles=True,
            score=True,
        )
        cv2.imshow("Pred", res)
        if cv2.waitKey() == ord("q"):
            break

    print(" " * 100, end="\r")
    ASE = np.array(scores)  # absolute score errors
    PCS = len(ASE[ASE == 0]) / len(ASE) * 100
    MASE = np.mean(ASE)

    print(f"Percent Correct Score (PCS): {PCS:.1f}%")
    print(f"Mean Absolute Score Error (MASE): {MASE:.2f}")


model = load_model()
img_paths = [
    "dump/DSC_0307.JPG",
    "dump/0022_.png",
    "dump/0022.png",
]
predict(
    model,
    # img_paths=img_paths,
)
