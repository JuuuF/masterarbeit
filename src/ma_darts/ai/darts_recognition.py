import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_disable_constant_folding=true"
# os.environ["XLA_FLAGS"] = "--xla_dump_to=/masterarbeit/dump/logs"

import re
import cv2
import numpy as np
import tensorflow as tf

# GPU setup
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [
            tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=2**14 + 2**12
            )
        ],  # 16GB + 4GB = 20GB
    )

from ma_darts.ai import callbacks as ma_callbacks
from ma_darts.ai.utils import get_dart_scores, get_absolute_score_error
from ma_darts.cv.utils import show_imgs
from ma_darts.ai.models import (
    yolo_v8_model,
    yolo_to_positions_and_class,
)
from ma_darts.ai.data import dataloader_paper, dataloader_ma, dummy_ds
from ma_darts.ai.losses import (
    YOLOv8Loss,
    ExistenceLoss,
    ClassesLoss,
    PositionsLoss,
    DIoULoss,
)

from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime

from ma_darts import img_size, classes

BATCH_SIZE = 16 if "GPU_SERVER" in os.environ.keys() else 4


class Utils:
    model_checkpoint_filepath = (
        f"data/ai/checkpoints/darts/{datetime.now().strftime('%y_%m_%d-%H_%M')}/"
        + "epoch={epoch:05d}_val_loss={val_loss:06f}.weights.h5"
    )

    def get_callbacks():
        callbacks = []
        # History plotter
        hp = ma_callbacks.HistoryPlotter(
            filepath="dump/training_history.png",
            update_on="batches",
            update_frequency=9999,
            smooth_curves=True,
            dark_mode=True,
            log_scale=True,
        )
        callbacks.append(hp)

        # Model Checkpoint
        mc = ma_callbacks.ModelCheckpoint(
            filepath=Utils.model_checkpoint_filepath,
            monitor="val_loss",
            save_best_only=True,
            max_saves=10,
            save_weights_only=True,
        )
        callbacks.append(mc)

        # Prediction callback
        Xs, ys = [], []
        for ds in [ds_gen_train, ds_paper_d1, ds_strongbows, ds_gen_val, ds_paper_d2, ds_jess]:
            X, y = next(iter(ds.unbatch().batch(2).take(1)))
            Xs.append(X)
            ys.append(y)
        Xs = np.concatenate(Xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        pc = ma_callbacks.PredictionCallback(
            X=Xs,
            y=ys,
            output_file="dump/prediction.png",
            update_on="batches",
            update_frequency=2,
        )
        callbacks.append(pc)

        lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=np.power(0.1, 1 / 3),
            patience=25,
            verbose=1,
            min_lr=1e-5,
        )
        callbacks.append(lr)

        # TensorBoard
        tb = tf.keras.callbacks.TensorBoard(
            log_dir="data/ai/logs",
            histogram_freq=1,
            profile_batch=(0, 500),
        )
        # callbacks.append(tb)

        return callbacks

    def get_best_model_checkpoint():
        checkpoint_dir = os.path.dirname(Utils.model_checkpoint_filepath)
        filenames = ""
        basename = Utils.model_checkpoint_filepath.split("/")[-1]

        # Check if directory exists
        if not os.path.exists(checkpoint_dir):
            return

        # Convert basename to regex
        while basename:
            char = basename[0]
            if char == "{":
                filenames += "([0-9]|\.)+"
                while basename[0] != "}":
                    basename = basename[1:]
                basename = basename[1:]
                continue

            if char == ".":
                filenames += "\."
                basename = basename[1:]
                continue

            filenames += basename[0]
            basename = basename[1:]

        # Find files matching regex
        files = [
            os.path.join(checkpoint_dir, f)
            for f in os.listdir(checkpoint_dir)
            if re.match(filenames, f)
        ]
        if not files:
            return None

        # extract validation loss
        def extract_number(f):
            f = f.split("val_loss=")[-1]
            i = 0
            found_dot = False
            while True:
                char = f[i]
                if char.isnumeric():
                    i += 1
                    continue
                if char == ".":
                    if found_dot:
                        break
                    found_dot = True
                    i += 1
                    continue
                break
            f = float(f[:i])
            return f

        files = sorted(files, key=extract_number)
        best_file = files[0]
        return best_file

    def get_args():
        parser = ArgumentParser()

        parser.add_argument(
            "--train",
            action="store_true",
            help="Train model.",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=1000,
            help="Training epochs.",
        )
        parser.add_argument(
            "--limit_data",
            type=int,
            default=-1,
            help="Dataset size limitation.",
        )
        parser.add_argument(
            "--clear_cache",
            action="store_true",
            help="Clear dataset cache files.",
        )
        parser.add_argument(
            "--predict",
            action="store_true",
            help="Predict test dataset using model.",
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default=None,
            help="Model path (optional).",
        )
        parser.add_argument(
            "--model_type",
            type=str,
            default="s",
            help="Model architecture size. Available: n, s, m, l, x",
        )

        args = parser.parse_args()
        return args

    def check_dataset(ds):

        for Xs, ys in ds:
            imgs = np.uint8(Xs * 255)
            for img, y in zip(imgs, ys):
                pos, cls = yolo_to_positions_and_class(y)
                for (y, x), cls_idx in zip(pos, cls):
                    y, x = round(y.numpy()), round(x.numpy())
                    cv2.circle(img, (x, y), 4, (255, 255, 255), 2, cv2.LINE_AA)
                    for i, c in enumerate([0, 255]):
                        cv2.putText(
                            img,
                            classes[cls_idx.numpy() + 1],
                            org=(x + 10, y),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1,
                            color=(c, c, c),
                            thickness=2 - i,
                        )
                show_imgs(img)


class Data:

    def visualize_data_predictions(model, ds):

        preds = []

        try:
            print("Predicting...")
            for X, y_true in tqdm(ds):
                y_pred = model.predict(X, verbose=0)
                preds.extend([(x, y, y_) for x, y, y_ in zip(X, y_true, y_pred)])
        except KeyboardInterrupt:
            print("\nPrediction aborted.")

        def matrix_to_string(matrix):
            # Scale positions and format as a string
            formatted_rows = [
                f"\t({round(row[1], 1):5.1f}, {round(row[0], 1):5.1f}) | Confidence: {round(100 * row[2]):3}%"
                for row in matrix
            ]

            # Join the rows into a single string with line breaks
            return "\n".join(formatted_rows)

        def plot_points(img, pts, color, add_unsure: bool = False):
            for y, x, existing in pts:
                if existing < 0.5:
                    if add_unsure:
                        color = (0, 0, round(255 * existing * 2))
                        cv2.circle(img, (int(x), int(y)), 5, color, 2)
                        cv2.circle(img, (int(x), int(y)), 2, (255, 255, 255), -1)
                    continue
                color = tuple(round(c * (existing / 2 + 0.5)) for c in color)
                cv2.circle(img, (int(x), int(y)), 5, color, 2)
                cv2.circle(img, (int(x), int(y)), 2, (255, 255, 255), -1)

        for img, y_true, y_pred in preds:

            img = np.uint8(img * 255)
            y_true = np.array(y_true, np.float32)
            y_pred = np.array(y_pred, np.float32)

            y_true[:, 0] *= img.shape[0]
            y_true[:, 1] *= img.shape[1]
            y_pred[:, 0] *= img.shape[0]
            y_pred[:, 1] *= img.shape[1]

            scoring_true = y_true[y_true[:, -1] > 0.5, :2]
            scoring_pred = y_pred[y_pred[:, -1] > 0.5, :2]
            scores_true = get_dart_scores(
                list(scoring_true), img_size=img_size, margin=100
            )
            scores_pred = get_dart_scores(
                list(scoring_pred), img_size=img_size, margin=100
            )
            ase = get_absolute_score_error(scores_true, scores_pred)
            print()
            print("Target values:", sorted(scores_true))
            print(matrix_to_string(y_true))
            print("Predicted values:", sorted(scores_pred))
            print(matrix_to_string(y_pred))
            print("Absolute Score Error:", ase)

            plot_points(img, y_true, (0, 255, 0))
            plot_points(img, y_pred, (255, 0, 0), add_unsure=True)

            show_imgs(img)


# data_dir = "data/generation/out"
# sample_ids = [f for f in os.listdir(data_dir) if f.isnumeric()]
# sample_ids = sorted(sample_ids, key=int)

# for i in range(1000):
#     sample_id = sample_ids[i]
#     sample_info = pickle.load(open(os.path.join(data_dir, sample_id, "info.pkl"), "rb"))
#     score = sample_info.scores
#     classes = Data.extract_dart_classes(sample_info)
#     print(score, classes)
#     input()
# exit()


# -----------------------------------------------
# Command Line Arguments

args = Utils.get_args()

# -----------------------------------------------
# Get Model

if args.model_path is None:
    model = yolo_v8_model(
        variant=args.model_type,
    )
else:
    model = tf.keras.models.load_model(args.model_path, compile=False)

# if "GPU_SERVER" not in os.environ.keys():
#     model.load_weights("data/ai/darts/latest.weights.h5")

# Compile model
metrics = [
    ExistenceLoss(name="01_xst_loss", multiplier=400),
    ClassesLoss(name="02_cls_loss", multiplier=850),
    PositionsLoss(name="03_pos_loss", multiplier=0.5),
    DIoULoss(name="04_diou_loss", multiplier=0.02),
]
# metrics = [metrics for _ in range(3)]
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
    loss=YOLOv8Loss(
        square_size=0.05,
        cls_threshold=0.003,
        cls_width=0.002,
        pos_threshold=0.004,
        pos_width=0.002,
        diou_threshold=0.003,
        diou_width=0.001,
    ),
    metrics=metrics,
)
# model.summary(160)
# print(model.input_shape)
# print(model.output_shape)
# exit()

# -----------------------------------------------
# Get Data

ds_gen_train = dataloader_ma(
    "data/generation/out/",
    batch_size=BATCH_SIZE,
    shuffle=True,
    augment=True,
    cache=True,
    clear_cache=args.clear_cache,
)  # 24,576

ds_paper_d1 = dataloader_paper(
    base_dir="data/paper/",
    dataset="d1",
    split="val",
    img_size=img_size,
    shuffle=True,
    augment=True,
    batch_size=BATCH_SIZE,
    cache=True,
    clear_cache=args.clear_cache,
)  # 2000

ds_strongbows = dataloader_ma(
    "data/darts_references/strongbows_out/",
    batch_size=BATCH_SIZE,
    shuffle=True,
    augment=True,
    cache=True,
    clear_cache=args.clear_cache,
)  # 168

# 24576 + 256 + 160 = 24992
train_ds = ds_gen_train.concatenate(  # 24576
    ds_paper_d1.take(256 // BATCH_SIZE),  # + 256
).concatenate(
    ds_strongbows.take(160 // BATCH_SIZE),  # + 160
)

# Utils.check_dataset(train_ds)
# if "GPU_SERVER" not in os.environ.keys():
#     train_ds = dummy_ds(n_samples=8)

ds_gen_val = dataloader_ma(
    "data/generation/out_val/",
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment=False,
    cache=True,
    clear_cache=args.clear_cache,
)  # 256

ds_paper_d2 = dataloader_paper(
    base_dir="data/paper/",
    dataset="d2",
    split="test",
    img_size=img_size,
    shuffle=False,
    augment=False,
    batch_size=BATCH_SIZE,
    cache=True,
    clear_cache=args.clear_cache,
)  # 152

ds_jess = dataloader_ma(
    "data/darts_references/jess_out/",
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment=False,
    cache=True,
    clear_cache=args.clear_cache,
)  # 128

# 256 + 256 + 128 = 640
val_ds = ds_gen_val.concatenate(  # 256
    ds_paper_d2.take(256 // BATCH_SIZE),  # + 256
).concatenate(
    ds_jess.take(128 // BATCH_SIZE),  # + 128
)


# Utils.check_dataset(val_ds)
# if "GPU_SERVER" not in os.environ.keys():
#     val_ds = dummy_ds(n_samples=4)

# -----------------------------------------------
# Fit Model

if args.train:

    try:
        model.fit(
            train_ds,
            epochs=args.epochs,
            validation_data=val_ds,
            callbacks=Utils.get_callbacks(),
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    if best_weights := Utils.get_best_model_checkpoint():
        model.load_weights(best_weights)

    model_path = "data/ai/darts_model.keras"
    model.compile()
    model.save(model_path)
    print(f"Model saved to {model_path}.")

if args.predict:
    test_ds = Data.get_ds(
        "data/generation/out_test/",
        shuffle=False,
        augment=False,
    )
    Data.visualize_data_predictions(model, test_ds)
