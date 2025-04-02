import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from ma_darts import classes, img_size

"""
https://cdn.prod.website-files.com/6479eab6eb2ed5e597810e9e/65bf057de6cebfb11455514f_flamelink%252Fmedia%252F222874205-3873bdac-7135-4ecc-8ab2-ca18b8e13fdf.webp
"""

# ------------------------------------------------------------------------
# Small Blocks


def Conv2d(
    x: tf.Tensor,
    k: int,
    s: int,
    p: bool,
    c: int,
    name: str | None = None,
) -> tf.Tensor:
    conv = layers.Conv2D(
        filters=c,
        kernel_size=(k, k),
        strides=(s, s),
        padding="same" if p else "valid",
        name=name,
    )
    return conv(x)


def BatchNorm2d(
    x: tf.Tensor,
    name: str | None = None,
) -> tf.Tensor:
    batch_normalization = layers.BatchNormalization(
        name=name,
    )
    return batch_normalization(x)


def SiLU(
    x: tf.Tensor,
) -> tf.Tensor:
    return tf.keras.activations.silu(x)


def Concat(
    xs: list[tf.Tensor],
    axis: int = -1,
    name: str | None = None,
) -> tf.Tensor:
    concat = layers.Concatenate(name=name, axis=axis)
    return concat(xs)


class Split(layers.Layer):
    def call(self, inputs):
        return tf.split(inputs, num_or_size_splits=2, axis=-1)


def Add(
    xs: list[tf.Tensor],
    name: str | None = None,
) -> tf.Tensor:
    add = layers.Add()
    return add(xs)


def MaxPool2d(
    x: tf.Tensor,
    p: int = 5,
    name: str | None = None,
) -> tf.Tensor:

    pool = layers.MaxPool2D(
        pool_size=(p, p),
        strides=(1, 1),
        padding="same",
    )

    return pool(x)


def Upsample(
    x: tf.Tensor,
    name: str | None = None,
) -> tf.Tensor:
    us = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
    )
    return us(x)


# ------------------------------------------------------------------------
# Bigger Blocks


def Conv(
    x: tf.Tensor,
    k: int,
    s: int,
    p: bool,
    c: int,
    dropout: float = 0.0,
    name: str | None = None,
) -> tf.Tensor:
    x = Conv2d(x, k, s, p, c)
    x = BatchNorm2d(x)
    x = SiLU(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x


def SPPF(
    x: tf.Tensor,
    pool_size: int = 5,
    name: str | None = None,
) -> tf.Tensor:
    c = x.shape[-1]
    x = Conv(x, k=1, s=1, p=False, c=c)

    p1 = MaxPool2d(x, p=pool_size)
    p2 = MaxPool2d(p1, p=pool_size)
    p3 = MaxPool2d(p2, p=pool_size)

    x = Concat([x, p1, p2, p3])
    x = Conv(x, k=1, s=1, p=False, c=c)

    return x


def Bottleneck(
    x: tf.Tensor,
    shortcut: bool,
    dropout: float = 0.0,
    name: str | None = None,
) -> tf.Tensor:

    c = x.shape[-1]
    x_conv = x
    x_conv = Conv(x_conv, k=3, s=1, p=True, c=c // 2, dropout=dropout)
    x_conv = Conv(x_conv, k=3, s=1, p=True, c=c, dropout=dropout)

    if shortcut:
        x_conv = Add([x, x_conv])

    return x_conv


def Detect(
    x: tf.Tensor,
    reg_max: int,
    nc: int,
    name: str | None = None,
    dropout: float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    x_pos = x
    x_cls = x
    c = x.shape[-1]

    x_pos = Conv(x_pos, k=3, s=1, p=True, c=c, dropout=dropout)
    x_pos = Conv(x_pos, k=3, s=1, p=True, c=c, dropout=dropout)
    x_pos = Conv2d(
        x_pos, k=1, s=1, p=False, c=2 * reg_max
    )  # original: c=4 * reg_max, but we omit w and h

    x_cls = Conv(x_cls, k=3, s=1, p=True, c=c, dropout=dropout * 2)
    x_cls = Conv(x_cls, k=3, s=1, p=True, c=c, dropout=dropout * 2)
    x_cls = Conv2d(
        x_cls, k=1, s=1, p=False, c=nc * reg_max
    )  # original: c=nc, but I want a per-estimation confidence

    return x_pos, x_cls


def C2f(
    x: tf.Tensor,
    shortcut: bool,
    n: int,
    c: int,
    dropout: float = 0.0,
    name: str | None = None,
) -> tf.Tensor:

    x = Conv(x, k=1, s=1, p=False, c=c, dropout=dropout)
    x_0, x_1 = Split()(x)

    concat_tensors = [x_0, x_1]
    for i in range(n):
        x_1 = Bottleneck(x_1, shortcut=shortcut)
        concat_tensors.append(x_1)

    x = Concat(concat_tensors)

    x = Conv(x, k=1, s=1, p=False, c=c, dropout=dropout)

    return x


# ------------------------------------------------------------------------
# Custom Additions


def Reshape(
    x: tf.Tensor,
    shape: list[int],
    name: str | None = None,
) -> tf.Tensor:
    res = layers.Reshape(
        target_shape=shape,
    )
    return res(x)


def Softmax(
    x: tf.Tensor,
    axis: int = -1,
) -> tf.Tensor:
    sm = layers.Softmax(axis=axis)
    return sm(x)


def Sigmoid(
    x: tf.Tensor,
) -> tf.Tensor:
    return tf.keras.activations.sigmoid(x)


def HardSigmoid(
    x: tf.Tensor,
) -> tf.Tensor:
    return tf.keras.activations.hard_sigmoid(x)


def ClampReLU(
    x: tf.Tensor,
) -> tf.Tensor:
    return layers.ReLU(max_value=1.0)(x)


def OutputTransformation(
    x_pos: tf.Tensor,  # (None, n, n, reg_max * 2)
    x_cls: tf.Tensor,  # (None, n, n, reg_max * n_classes)
    reg_max: int,
    n_classes: int,
    out_name: str | None = None,
) -> tf.Tensor:

    # Reshaping
    x_pos = Reshape(
        x_pos,
        shape=x_pos.shape[1:3] + (2, reg_max),
    )  # (y, x, 2, 3)

    # Extract existence
    x_cls = Reshape(
        x_cls,
        shape=x_cls.shape[1:3] + (n_classes, reg_max),
    )  # (y, x, n_classes, 3)

    x_xst = x_cls[..., :1, :]  # (None, n, n, 1, 3)
    x_cls = x_cls[..., 1:, :]  # (None, n, n, 5, 3)

    # Activations
    x_xst = Sigmoid(x_xst)
    x_pos = ClampReLU(x_pos)  # clamp between 0 and 1
    # x_cls = Softmax(x_cls, axis=-2)  # determine class percentage-wise

    # Combining
    x = Concat([x_xst, x_pos, x_cls], axis=-2, name=out_name)  # (s, s, 8, 3)

    return x


# ------------------------------------------------------------------------
# Model Itself


def yolo_v8_model(
    classes=classes,
    variant="n",
) -> tf.keras.Model:
    """
    YOLOv8 implementation in TensorFlow (which is way cooler than PyTorch).

    Parameters
    ----------

    variant: str
        Variant of the model. Can be one of (n, s, m, l, x).
        The default is "n".

    Returns
    -------

    model: tf.keras.Model
        YOLOv8 model in TensorFlow.
        input_shape: (img_size, img_size, 3)
        output_shape: [
            (s, s, 2 + n_classes, 3),
            (m, m, 2 + n_classes, 3),
            (l, l, 2 + n_classes, 3),
        ]
        where:
            s = img_size // 32
            m = img_size // 16
            l = img_size // 8
    """
    inputs = layers.Input(shape=(img_size, img_size, 3), name="Input")

    # Implicit addition of nothing class as first index
    classes = list(set(classes))
    if "nothing" in classes:
        classes.remove("nothing")
    classes.insert(0, "nothing")

    variants = {  # d, w, r
        "n": (1 / 3, 0.25, 2),
        "s": (1 / 3, 0.5, 2),
        "m": (2 / 3, 0.75, 1.5),
        "l": (1, 1, 1),
        "x": (1, 1.25, 1),
    }

    d, w, r = variants[variant]
    reg_max = 3
    n_classes = len(classes)  # nothing / black / white / red / green / out

    # Backbone
    x = inputs
    dropout_backbone = 0.15

    # fmt: off
    x_0 = Conv(x, k=3, s=2, p=True, c=round(64 * w), dropout=0.0)  # P1
    x_1 = Conv(x_0, k=3, s=2, p=True, c=round(128 * w), dropout=0.0)  # P2
    x_2 = C2f(x_1, shortcut=True, n=round(3 * d), c=round(128 * w), dropout=0.0)
    x_3 = Conv(x_2, k=3, s=2, p=True, c=round(256 * w), dropout=0.0)  # P3
    x_4 = C2f(x_3, shortcut=True, n=round(6 * d), c=round(256 * w), dropout=dropout_backbone)
    x_5 = Conv(x_4, k=3, s=2, p=True, c=round(512 * w), dropout=dropout_backbone)  # P4
    x_6 = C2f(x_5, shortcut=True, n=round(6 * d), c=round(512 * w), dropout=dropout_backbone)
    x_7 = Conv(x_6, k=3, s=2, p=True, c=round(512 * w * r), dropout=dropout_backbone)  # P5
    x_8 = C2f(x_7, shortcut=True, n=round(3 * d), c=round(512 * w * r), dropout=dropout_backbone)
    x_9 = SPPF(x_8, pool_size=5)
    # fmt: on

    # Head
    dropout_head = 0.3
    x_10 = Upsample(x_9)
    x_11 = Concat([x_6, x_10])
    x_12 = C2f(
        x_11, shortcut=False, n=round(3 * d), c=round(512 * w), dropout=dropout_head
    )
    x_13 = Upsample(x_12)
    x_14 = Concat([x_4, x_13])
    x_15 = C2f(
        x_14, shortcut=False, n=round(3 * d), c=round(256 * w), dropout=dropout_head
    )  # P3

    x_16 = Conv(x_15, k=3, s=2, p=True, c=round(256 * w), dropout=dropout_head)  # P3
    x_17 = Concat([x_12, x_16])
    x_18 = C2f(
        x_17, shortcut=False, n=round(3 * d), c=round(512 * w), dropout=dropout_head
    )  # P4
    x_19 = Conv(x_18, k=3, s=2, p=True, c=round(512 * w), dropout=dropout_head)
    x_20 = Concat([x_9, x_19])
    x_21 = C2f(
        x_20, shortcut=False, n=round(3 * d), c=round(512 * w * r), dropout=dropout_head
    )  # P5

    # (n, n, 2)
    detect_s_pos, detect_s_cls = Detect(
        x_21, reg_max=reg_max, nc=n_classes, dropout=0.15
    )  # (None, 25, 25, 3*2), (None, 25, 25, 3*6)
    # Output Transformation
    detect = OutputTransformation(
        detect_s_pos,
        detect_s_cls,
        reg_max=reg_max,
        n_classes=n_classes,
        out_name="out_s",
    )  # (s, s, 2 + n_classes, reg_max)

    outputs = [detect]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.classes = classes
    return model


# =================================================================================================
# Data Loader


def positions_to_yolo(
    img_size: int,
    darts_info: list[tuple[tuple[int, int], str]],  # (y, x), score_str
) -> list[np.ndarray]:
    # Convert scores to classes
    darts_info = [(pos, score2class(s)) for pos, s in darts_info]
    # Filter out non-existing darts
    darts_info = [(pos, cls) for pos, cls in darts_info if not cls == 0]

    outputs = []

    for i in range(3):
        # print()
        # Calculate grid size
        grid_size = img_size // 2 ** (5 - i)
        cell_size = img_size / grid_size
        # print(f"{grid_size=}")

        scale_output = np.zeros(
            (grid_size, grid_size, 2 + len(classes), 3)
        )  # (s, s, 2+nc, 3)
        scale_output[:, :, 2, :] = 1

        for (y, x), color_cls in darts_info:

            # Bin position into grid
            grid_pos, local_pos = np.divmod([y, x], cell_size)
            grid_pos = np.int32(grid_pos)
            local_pos /= cell_size
            # print(grid_pos, local_pos, classes[color_cls])

            # get cell idx
            cell = scale_output[grid_pos[0], grid_pos[1]]  # (3, 3)
            # print("CELL", cell)
            cell_col = np.where(cell[2, :] != 0)[0][
                0
            ]  # smallest index with nothing class

            # Set position, class and un-set nothing class
            cell[:2, cell_col] = local_pos
            cell[2 + color_cls, cell_col] = 1
            cell[2, cell_col] = 0
            # print("CELL", cell)

        outputs.append(scale_output)

    return outputs


def yolo_to_positions_and_class(
    y: np.ndarray | tf.Tensor,  # (y, x, 8, 3)
) -> np.ndarray:  # (n, 3)

    s = tf.shape(y)[0]  # 25/50/100
    cell_indices = tf.stack(
        tf.meshgrid(tf.range(s), tf.range(s), indexing="ij"),
        axis=-1,
    )  # (s, s, 2)
    global_grid_pos = cell_indices * 800 / s  # (s, s, 2)
    global_grid_pos = tf.cast(global_grid_pos, tf.float32)

    xst = y[:, :, :1, :]  # (y, x, 1, 3)
    pos = y[:, :, 1:3, :]  # (y, x, 2, 3)
    cls = y[:, :, 3:, :]  # (y, x, 5, 3)

    pos_abs = (
        pos * tf.cast(800 / s, tf.float32) + global_grid_pos[:, :, :, None]
    )  # (s, s, 2, 3)

    pos_abs = tf.transpose(pos_abs, [0, 1, 3, 2])  # (y, x, 3, 2)
    pos_abs = tf.reshape(pos_abs, [-1, 2])  # (m, 2)

    cls = tf.transpose(cls, [0, 1, 3, 2])  # (y, x, 3, 3)
    cls = tf.reshape(cls, [-1, len(classes) - 1])  # (m, 5)

    xst = tf.transpose(xst, [0, 1, 3, 2])  # (y, x, 3, 1)
    xst = tf.reshape(xst, [-1, 1])  # (m, 1)

    existing = xst[:, 0] > 0.5  # (m,)
    darts_positions = pos_abs[existing]  # (n, 2), n <= m, n = amount of found points
    dart_classes = cls[existing]  # (n, 6)
    dart_classes = tf.argmax(dart_classes, axis=1)  # (n,)
    return darts_positions, dart_classes


def score2class(score: str):
    if score == "HIDDEN":
        return classes.index("nothing")
    if score == "OUT":
        return classes.index("out")

    # bull
    if score in ["DB", "DBull"]:
        return classes.index("red")
    if score in ["B", "Bull"]:
        return classes.index("green")

    dart_order = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    multiplier = score[0] in "DT"
    score = int(score[1:]) if multiplier else int(score)

    if score == 0:
        return classes.index("out")

    dark_color = dart_order.index(score) % 2 == 0

    if dark_color and not multiplier:
        return classes.index("black")
    if dark_color and multiplier:
        return classes.index("red")
    if not dark_color and not multiplier:
        return classes.index("white")
    if not dark_color and multiplier:
        return classes.index("green")


if __name__ == "__main__":
    model = yolo_v8_model(
        variant="n",
    )
    model.compile(loss="mse", optimizer="adam")
    model.build(input_shape=(None, 800, 800, 3))
    model.summary()
    tf.keras.utils.plot_model(model, "dump/model.png", show_shapes=True)
    exit()
    # model.compile(
    #     loss=lambda x, y: yolo_v8_loss(x, y, 50),
    #     optimizer="adam",
    # )

    model.load_weights("data/ai/darts/yolov8_train5.weights.h5")
    data_dir_paper = "data/paper/imgs/d1_02_04_2020/"
    data_dir_ma = "data/generation/out_val/"
    import os

    files_paper = [os.path.join(data_dir_paper, f) for f in os.listdir(data_dir_paper)]
    files_ma = [
        os.path.join(data_dir_ma, s, "undistort.png") for s in os.listdir(data_dir_ma)
    ]
    files = [f for tup in zip(files_paper, files_ma) for f in tup]
    files[0] = "dump/test.png"
    files[1] = "dump/test_2.png"

    for f in files:
        img = np.array([cv2.imread(f)], dtype=np.float32) / 255
        yolo_v8_predict(model, img)
    exit()

    y_true = positions_to_yolo(
        800,
        [
            ((410, 410), "DB"),
            ((490, 120), "D18"),
            ((400, 123), "12"),
            ((665, 584), "OUT"),
            ((0, 0), "HIDDEN"),
        ],
    )

    y_pred = positions_to_yolo(
        800,
        [
            ((400, 150), "11"),
            (
                (np.random.randint(0, 800), np.random.randint(0, 800)),
                str(np.random.randint(1, 21)),
            ),
            (
                (np.random.randint(0, 800), np.random.randint(0, 800)),
                str(np.random.randint(1, 21)),
            ),
            (
                (np.random.randint(0, 800), np.random.randint(0, 800)),
                str(np.random.randint(1, 21)),
            ),
            (
                (np.random.randint(0, 800), np.random.randint(0, 800)),
                str(np.random.randint(1, 21)),
            ),
            (
                (np.random.randint(0, 800), np.random.randint(0, 800)),
                str(np.random.randint(1, 21)),
            ),
            (
                (np.random.randint(0, 800), np.random.randint(0, 800)),
                str(np.random.randint(1, 21)),
            ),
            ((400, 150), "11"),
            ((493, 123), "12"),
            ((665, 584), "OUT"),
            ((0, 0), "HIDDEN"),
        ],
    )

    # y_pred = model.predict(np.zeros((1, 800, 800, 3), np.float32))

    # for i in range(3):
    #     y_pred[i][:, 14, 14, 3, 0] = 1
    #     y_pred[i][:, 14, 14, 2, 0] = 0

    y_true = [np.expand_dims(y, 0) for y in y_true]
    y_pred = [np.expand_dims(y, 0) for y in y_pred]

    from ma_darts.ai.callbacks import PredictionCallback
    from ma_darts.cv.utils import show_imgs
    import cv2

    X = cv2.imread("data/generation/out_test/12305/undistort.png")
    X = np.float32(X) / 255
    X = np.expand_dims(X, 0)
    pc = PredictionCallback(X=X, y=y_true, output_file="dump/pred.png")
    pc.set_model(model)
    pc.plot_prediction(y_pred)
    exit()

    loss = YOLOv8Loss(800, 50)
    l = loss(y_true, y_true)
    exit()

    from cProfile import Profile
    import pstats

    with Profile() as p:
        l = loss(y_true, y_true)
    pstats.Stats(p).dump_stats("dump/profile.prof")
    print(l)
