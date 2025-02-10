import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

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


def Split(
    x: tf.Tensor,
    name: str | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    split = layers.Lambda(lambda t: tf.split(t, num_or_size_splits=2, axis=-1))
    return split(x)


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
    pool_sizes: list[int] = [5, 5, 5],
    name: str | None = None,
) -> tf.Tensor:
    c = x.shape[-1]
    x = Conv(x, k=1, s=1, p=False, c=c)

    xs = []
    for i, p in enumerate(pool_sizes):
        x = MaxPool2d(x, p=p)
        xs.append(x)

    x = Concat(xs)
    x = Conv(x, k=1, s=1, p=False, c=c)

    return x


def Bottleck(
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

    x_cls = Conv(x_cls, k=3, s=1, p=True, c=c)
    x_cls = Conv(x_cls, k=3, s=1, p=True, c=c)
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
    x_0, x_1 = Split(x)

    concat_tensors = [x_0, x_1]
    for i in range(n):
        x_1 = Bottleck(x_1, shortcut=shortcut)
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


def OutputTansformation(
    x_pos: tf.Tensor,
    x_cls: tf.Tensor,
    reg_max: int,
    n_classes: int,
) -> tf.Tensor:

    # Reshaping
    x_pos = Reshape(
        x_pos,
        shape=x_pos.shape[1:3] + (2, reg_max),
    )  # (bs, y, x, 2, 3)
    x_cls = Reshape(
        x_cls,
        shape=x_cls.shape[1:3] + (n_classes, reg_max),
    )  # (bs, y, x, n, 3)

    # Activations
    x_pos = HardSigmoid(x_pos)  # clamp between 0 and 1
    x_cls = Softmax(x_cls, axis=-2)  # determine class percentage-wise

    # Combining
    x = Concat([x_pos, x_cls], axis=-2)

    return x


# ------------------------------------------------------------------------
# Model Itself


def yolo_v8_model(input_size: int, classes: list[str], variant="n") -> tf.keras.Model:
    """
    YOLOv8 implementation in TensorFlow (which is way cooler than PyTorch).

    Parameters
    ----------

    input_size: int
        Input image size for the model.
    classes: list[str]
        Classes to predict. Note that there's an implicit "nothing"-class
        as the first element of the list.
    variant: str
        Variant of the model. Can be one of (n, s, m, l, x).
        The default is "n".

    Returns
    -------

    model: tf.keras.Model
        YOLOv8 model in TensorFlow.
        input_shape: (input_size, input_size, 3)
        output_shape: [
            (s, s, 2 + n_classes, 3),
            (m, m, 2 + n_classes, 3),
            (l, l, 2 + n_classes, 3),
        ]
        where:
            s = input_size // 32
            m = input_size // 16
            l = input_size // 8
    """
    inputs = layers.Input(shape=(input_size, input_size, 3), name="Input")

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
    n_classes = len(classes)  # nothing / black / green / red / white / out

    # Backbone
    x = inputs

    x_0 = Conv(x, k=3, s=2, p=True, c=round(64 * w))  # P1
    x_1 = Conv(x_0, k=3, s=2, p=True, c=round(128 * w))  # P2
    x_2 = C2f(x_1, shortcut=True, n=round(3 * d), c=round(128 * w))
    x_3 = Conv(x_2, k=3, s=2, p=True, c=round(256 * w))  # P3
    x_4 = C2f(x_3, shortcut=True, n=round(6 * d), c=round(256 * w))
    x_5 = Conv(x_4, k=3, s=2, p=True, c=round(512 * w))  # P4
    x_6 = C2f(x_5, shortcut=True, n=round(6 * d), c=round(512 * w))
    x_7 = Conv(x_6, k=3, s=2, p=True, c=round(512 * w * r))  # P5
    x_8 = C2f(x_7, shortcut=True, n=round(3 * d), c=round(512 * w * r))
    x_9 = SPPF(x_8, pool_sizes=[5, 5, 5])

    # Head
    x_10 = Upsample(x_9)
    x_11 = Concat([x_6, x_10])
    x_12 = C2f(x_11, shortcut=False, n=round(3 * d), c=round(512 * w), dropout=0.2)
    x_13 = Upsample(x_12)
    x_14 = Concat([x_4, x_13])
    x_15 = C2f(
        x_14, shortcut=False, n=round(3 * d), c=round(256 * w), dropout=0.2
    )  # P3

    x_16 = Conv(x_15, k=3, s=2, p=True, c=round(256 * w), dropout=0.2)  # P3
    x_17 = Concat([x_12, x_16])
    x_18 = C2f(
        x_17, shortcut=False, n=round(3 * d), c=round(512 * w), dropout=0.2
    )  # P4
    x_19 = Conv(x_18, k=3, s=2, p=True, c=round(512 * w), dropout=0.2)
    x_20 = Concat([x_9, x_19])
    x_21 = C2f(
        x_20, shortcut=False, n=round(3 * d), c=round(512 * w * r), dropout=0.2
    )  # P5

    # (n, n, 2)
    detect_s_pos, detect_s_cls = Detect(
        x_21, reg_max=reg_max, nc=n_classes, dropout=0.2
    )
    detect_m_pos, detect_m_cls = Detect(
        x_18, reg_max=reg_max, nc=n_classes, dropout=0.2
    )
    detect_l_pos, detect_l_cls = Detect(
        x_15, reg_max=reg_max, nc=n_classes, dropout=0.2
    )

    # Output Transformation
    detect_s = OutputTansformation(
        detect_s_pos, detect_s_cls, reg_max=reg_max, n_classes=n_classes
    )  # (s, s, 2 + n_classes, reg_max)
    detect_m = OutputTansformation(
        detect_m_pos, detect_m_cls, reg_max=reg_max, n_classes=n_classes
    )  # (m, m, 2 + n_classes, reg_max)
    detect_l = OutputTansformation(
        detect_l_pos, detect_l_cls, reg_max=reg_max, n_classes=n_classes
    )  # (l, l, 2 + n_classes, reg_max)

    outputs = [
        detect_s,
        detect_m,
        detect_l,
    ]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.classes = classes
    return model


# =================================================================================================
# Loss

tf.config.set_soft_device_placement(True)


class YOLOv8Loss(tf.keras.Loss):
    def __init__(
        self,
        img_size: int,
        square_size: int = 50,
        class_introduction_threshold: int | float = np.Inf,
        position_introduction_threshold: int | float = np.Inf,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.square_size = square_size
        self.existence_threshold = 0.5
        self.xst_loss = tf.keras.losses.BinaryFocalCrossentropy()

        self.class_introduction_threshold = tf.constant(
            class_introduction_threshold, tf.float32
        )
        self.position_introduction_threshold = tf.constant(
            position_introduction_threshold, tf.float32
        )

    @tf.function(jit_compile=False)
    def call(
        self,
        y_true: tf.Tensor,  # (bs, y, x, 2+n, 3)
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        # Split tensors
        pos_true = y_true[..., :2, :]  # (bs, y, x, 2, 3)
        pos_pred = y_pred[..., :2, :]
        cls_true = y_true[..., 2:, :]  # (bs, y, x, n, 3)
        cls_pred = y_pred[..., 2:, :]

        # ---------------------------------------
        # 1. Existence per cell
        # Get existences
        xst_true = self.get_cell_existence(cls_true)  # (bs, y, x)
        xst_pred = self.get_cell_existence(cls_pred)

        # Calculate existence loss
        total_loss = self.xst_loss(xst_true, xst_pred)

        # ---------------------------------------
        # 2. Classes of existing cells

        cls_loss, pos_pred, cls_pred = tf.cond(
            tf.less(total_loss, self.class_introduction_threshold),
            true_fn=lambda: self.get_cls_loss(y_pred, cls_true, cls_pred),
            false_fn=lambda: (tf.constant(100, tf.float32), pos_pred, cls_pred),
        )

        total_loss = total_loss + cls_loss

        # ---------------------------------------
        # 3. Positions loss

        pos_loss = tf.cond(
            tf.less(total_loss, self.position_introduction_threshold),
            true_fn=lambda: self.get_positions_loss(
                pos_true, pos_pred, cls_true, cls_pred
            ),
            false_fn=lambda: tf.constant(100, tf.float32),
        )

        total_loss = total_loss + pos_loss

        return total_loss

    def get_cls_loss(
        self,
        y_pred,
        cls_true,
        cls_pred,
    ):

        # Get all output permutations
        permutations = tf.constant(
            [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)],
            dtype=tf.int32,
        )  # if you can figure out how to tf-vectorize this, go for it!

        # Gather permutations - we step into weird territory here
        # (bs, y, x, 18, 6)
        cls_pred_permuted = self.get_permutations(cls_pred, permutations)
        cls_true_permuted = self.get_permutations(cls_true, [[0, 1, 2]] * 6)

        # Get best loss per cell permutation and its permutation index
        cls_loss, cell_perm_idx = self.binary_focal_crossentropy_per_permutation_cell(
            cls_true_permuted, cls_pred_permuted
        )  # (1,), (bs, y, x)

        # Switch to the best class permutation orders
        y_pred = self.apply_cell_permutations(
            y_pred, permutations, cell_perm_idx
        )  # (bs, y, x, 2+n, 3)
        pos_pred = y_pred[..., :2, :]  # (bs, y, x, 2, 3)
        cls_pred = y_pred[..., 2:, :]  # (bs, y, x, n, 3)

        return cls_loss, pos_pred, cls_pred

    def get_positions_loss(
        self,
        pos_true: tf.Tensor,  # (bs, y, x, 2, 3)
        pos_pred: tf.Tensor,
        cls_true: tf.Tensor,  # (bs, y, x, 6, 3)
        cls_pred: tf.Tensor,
    ):
        # Convert to absolute coordinates
        pos_true, pos_pred = self.convert_to_absolute_coordinates(pos_true, pos_pred)

        # Find predicted classes
        cls_true = tf.transpose(cls_true, (0, 1, 2, 4, 3))  # (bs, y, x, 3, 6)
        cls_pred = tf.transpose(cls_pred, (0, 1, 2, 4, 3))
        cls_true = tf.argmax(cls_true, axis=-1)  # (bs, y, x, 3)
        cls_pred = tf.argmax(cls_pred, axis=-1)

        # Flatten grid dimension
        batch_size = tf.shape(cls_true)[0]
        cls_true = tf.reshape(cls_true, (batch_size, -1, 3))  # (bs, s, 3), s=y*x
        cls_pred = tf.reshape(cls_pred, (batch_size, -1, 3))
        pos_true = tf.reshape(pos_true, (batch_size, -1, 2, 3))  # (bs, s, 2, 3)
        pos_pred = tf.reshape(pos_pred, (batch_size, -1, 2, 3))
        pos_true = tf.transpose(pos_true, (0, 1, 3, 2))  # (bs, s, 3, 2)
        pos_pred = tf.transpose(pos_pred, (0, 1, 3, 2))

        # Mask positions
        mask_true = cls_true != 0  # (bs, s, 3)
        mask_pred = cls_pred != 0
        mask_true = tf.cast(tf.expand_dims(mask_true, -1), tf.float32)  # (bs, s, 3, 1)
        mask_pred = tf.cast(tf.expand_dims(mask_pred, -1), tf.float32)
        mask_true = tf.repeat(mask_true, repeats=2, axis=-1)  # (bs, s, 3, 2)
        mask_pred = tf.repeat(mask_pred, repeats=2, axis=-1)

        # batch_size = tf.shape(cls_true)[0]
        # pos_loss_batch = tf.map_fn(
        #     lambda batch: self.single_sample_iou(
        #         tf.reshape(tf.boolean_mask(pos_true[batch], mask_true[batch]), [-1, 2]),
        #         tf.reshape(tf.boolean_mask(pos_pred[batch], mask_pred[batch]), [-1, 2]),
        #     ),
        #     elems=tf.range(batch_size),
        #     dtype=tf.float32,
        # )
        # pos_loss = tf.reduce_mean(pos_loss_batch)
        # return tf.cast(pos_loss, tf.float32)
        pos_true_reduced = tf.boolean_mask(pos_true, mask_true)
        pos_pred_reduced = tf.boolean_mask(pos_pred, mask_true)

        mse = tf.reduce_mean(tf.square(pos_true_reduced - pos_pred_reduced))
        return tf.cast(mse, tf.float32)

    def single_sample_iou(
        self,
        # pos_true: tf.Tensor,  # (3, 2)
        # pos_pred: tf.Tensor,  # (n, 2)
        pos_true,  # (s, 3, 2)
        pos_pred,
        mask_true,  # (s, 3, 2)
        mask_pred,
        batch,
    ):
        """
        Okay, here we step into very deep abstraction levels:
        We want to compute IoU scores of squares of _equal_ sizes each -
        that turns out to be really handy.
        """
        pos_true = pos_pred
        pos_pred = pos_pred
        mask_true = mask_pred
        mask_pred = mask_pred
        return tf.constant(10, tf.float32)  # NOTE: does not work with this

        def iou_computation(pos_true, pos_pred, square_size):
            # Extending dimensions
            # return tf.constant(10, tf.float32)  # XXX

            pos_true = tf.expand_dims(pos_true, 1)  # (m, 1, 2)
            pos_pred = tf.expand_dims(pos_pred, 0)  # (1, n, 2)

            # Get distances to square centers
            dists = tf.abs(pos_true - pos_pred)  # (m, n, 2)

            # Compute intersection area
            # NOTE: we do not handle self-intersection within a mask
            side_lengths = tf.maximum(square_size - dists, 0)
            intersections = tf.reduce_prod(side_lengths, axis=-1)
            intersection_area = tf.cast(tf.reduce_sum(intersections), tf.float32)

            # Compute union area
            n_points = tf.shape(pos_true)[0] + tf.shape(pos_pred)[0]
            total_area = tf.cast(n_points * square_size * square_size, tf.float32)
            union_area = total_area - intersection_area

            # Compute IoU
            iou = tf.math.divide_no_nan(intersection_area, union_area)
            return iou

        def iou_guess(pos_true, square_size, img_size):
            intersection = tf.cast(
                tf.shape(pos_true)[0] * square_size * square_size, tf.float32
            )
            union = tf.cast(img_size * img_size, tf.float32)
            return tf.math.divide_no_nan(intersection, union)

        # cut-down lengths for graph compilation
        pos_true = pos_true[:251]
        pos_pred = pos_pred[:251]

        iou = tf.cond(
            tf.shape(pos_pred)[0] < 250,
            lambda: iou_computation(pos_true, pos_pred, self.square_size),
            lambda: iou_guess(pos_true, self.square_size, self.img_size),
        )

        return 1 - iou

    def apply_cell_permutations(
        self,
        y: tf.Tensor,  # (bs, y, x, n, 3)
        permutations: tf.Tensor,  # (6, 3)
        cell_perm_idx: tf.Tensor,  # (bs, y, x)
    ):
        # Get permutation per cell
        target_perms = tf.gather(permutations, cell_perm_idx)  # (bs, y, x, 3)

        # Apply permutations per cell
        y_perm = tf.gather(y, target_perms, axis=-1, batch_dims=3)  # (bs, y, x, n, 3)

        return y_perm

    def get_permutations(
        self,
        y: tf.Tensor,  # (bs)
        perms: tf.Tensor,
    ):
        batch_size = tf.shape(y)[0]
        s = tf.shape(y)[1]

        # Apply all given permutations
        gathered_perms = tf.gather(y, perms, axis=-1)  # (bs, y, x, 6, 6, 3)

        # Transpose to re-order axes
        transposed_perms = tf.transpose(
            gathered_perms, [0, 1, 2, 5, 3, 4]
        )  # (bs, y, x, 3, 6, 6)

        # Combine corresponding axes
        y_permuted = tf.reshape(
            transposed_perms, [batch_size, s, s, 18, 6]
        )  # (bs, y, x, 18, 6)

        return y_permuted

    def binary_focal_crossentropy_per_permutation_cell(
        self,
        cls_true: tf.Tensor,
        cls_pred: tf.Tensor,
        gamma: float = 2.0,
        alpha: float = 0.25,
    ):
        epsilon = tf.keras.backend.epsilon()
        cls_pred = tf.clip_by_value(cls_pred, epsilon, 1.0 - epsilon)

        # Compute binary focal crossentropy loss per element
        loss_pos = (
            -alpha * cls_true * tf.math.pow(1 - cls_pred, gamma) * tf.math.log(cls_pred)
        )
        loss_neg = (
            -(1 - alpha)
            * (1 - cls_true)
            * tf.math.pow(cls_pred, gamma)
            * tf.math.log(1 - cls_pred)
        )
        loss_per_element = loss_pos + loss_neg  # (bs, y, x, 18, 6)

        loss_per_permutation = tf.reduce_mean(loss_per_element, axis=3)  # (bs, y, x, 6)

        loss_per_cell = tf.reduce_min(loss_per_permutation, axis=-1)  # (bs, y, x)
        cls_loss = tf.reduce_mean(loss_per_cell)

        cell_perm_idx = tf.argmin(loss_per_permutation, axis=-1)  # (bs, y, x)

        return cls_loss, cell_perm_idx

    def get_cell_existence(
        self,
        y_cls: tf.Tensor,  # (bs, y, x, n, 3)
    ) -> tf.Tensor:
        # Flatten all predictions per cell
        y_cls = tf.reduce_max(y_cls, axis=-1)  # (bs, y, x, n)

        # Binarize existences
        some_class_found = tf.reduce_max(y_cls[..., 1:], axis=-1)  # (bs, y, x)

        y_xst = tf.where(some_class_found > 0.5, 1, 0)  # (bs, y, x)
        return y_xst

    def convert_to_absolute_coordinates(
        self,
        pos_true: tf.Tensor,  # (bs, y, x, 2, 3)
        pos_pred: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        s = tf.shape(pos_true)[1]
        cell_size = self.img_size / tf.cast(s, tf.float32)

        grid_indices = tf.stack(
            tf.meshgrid(tf.range(s), tf.range(s), indexing="ij"),
            axis=-1,
        )  # (y, x, 2)
        global_grid_pos = (
            tf.cast(grid_indices, tf.float32) * cell_size
        )  # top-left corners for each cell, (y, x, 2)

        global_pos_true = (
            global_grid_pos[None, :, :, :, None] + pos_true * cell_size
        )  # (bs, y, x, 2, 3)
        global_pos_pred = (
            global_grid_pos[None, :, :, :, None] + pos_pred * cell_size
        )  # (bs, y, x, 2, 3)

        return global_pos_true, global_pos_pred


# =================================================================================================
# Data Loader

CLASSES = ["nothing", "black", "white", "red", "green", "out"]


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
            (grid_size, grid_size, 2 + len(CLASSES), 3)
        )  # (s, s, 2+nc, 3)
        scale_output[:, :, 2, :] = 1

        for (y, x), color_cls in darts_info:

            # Bin position into grid
            grid_pos, local_pos = np.divmod([y, x], cell_size)
            grid_pos = np.int32(grid_pos)
            local_pos /= cell_size
            # print(grid_pos, local_pos, CLASSES[color_cls])

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


def yolo_to_positions(
    grid_positions: np.ndarray | tf.Tensor,  # (y, x, 3, 3)
) -> np.ndarray:  # (n, 3)
    s = tf.shape(grid_positions)[0]  # 25/50/100
    cell_indices = tf.stack(
        tf.meshgrid(tf.range(s), tf.range(s), indexing="ij"),
        axis=-1,
    )  # (s, s, 2)
    global_grid_pos = cell_indices * 800 / s  # (s, s, 2)

    xst = grid_positions[:, :, -1:, :]  # (y, x, 1, 3)
    pos = grid_positions[:, :, :-1, :]  # (y, x, 2, 3)

    pos_abs = pos * (800 / s) + global_grid_pos[:, :, :, None]  # (s, s, 2, 3)

    pos_abs = tf.transpose(pos_abs, [0, 1, 3, 2])  # (y, x, 3, 2)
    pos_abs = tf.reshape(pos_abs, [-1, 2])  # (m, 2)

    xst = tf.transpose(xst, [0, 1, 3, 2])  # (y, x, 3, 1)
    xst = tf.reshape(xst, [-1, 1])  # (m, 1)

    existing = xst[:, 0] > 0.5  # (m,)
    darts_positions = pos_abs[existing]  # (n, 2), n <= m, n = amount of found points
    darts_existence = xst[existing]  # (n, 1)
    res = np.concatenate(
        [darts_positions, darts_existence], axis=-1
    )  # (n, 3): y, x, existence

    return res


def score2class(score: str):
    if score == "HIDDEN":
        return CLASSES.index("nothing")
    if score == "OUT":
        return CLASSES.index("out")

    # bull
    if score in ["DB", "DBull"]:
        return CLASSES.index("red")
    if score in ["B", "Bull"]:
        return CLASSES.index("green")

    dart_order = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    multiplier = score[0] in "DT"
    score = int(score[1:]) if multiplier else int(score)

    if score == 0:
        return CLASSES.index("out")

    dark_color = dart_order.index(score) % 2 == 0

    if dark_color and not multiplier:
        return CLASSES.index("black")
    if dark_color and multiplier:
        return CLASSES.index("red")
    if not dark_color and not multiplier:
        return CLASSES.index("white")
    if not dark_color and multiplier:
        return CLASSES.index("green")


if __name__ == "__main__":
    exit()
    model = yolo_v8_model(
        input_size=800,
        classes=["black", "white", "red", "green", "out", "nothing"],
        variant="n",
    )
    model.compile(
        loss=lambda x, y: yolo_v8_loss(x, y, 50),
        optimizer="adam",
    )

    model.fit(
        np.zeros((4, 800, 800, 3), np.uint8),
        [np.zeros((4, s, s, 8, 3), np.uint8) for s in [25, 50, 100]],
        epochs=1,
        batch_size=4,
    )
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

    loss = YOLOv8Loss(800, 50)
    l = loss(y_true, y_true)
    exit()

    from cProfile import Profile
    import pstats

    with Profile() as p:
        l = loss(y_true, y_true)
    pstats.Stats(p).dump_stats("dump/profile.prof")
    print(l)
