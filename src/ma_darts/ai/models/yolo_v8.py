import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

tf.config.run_functions_eagerly(False)
print("Eager execution enabled:", tf.executing_eagerly())

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
    name: str | None = None,
) -> tf.Tensor:
    x = Conv2d(x, k, s, p, c)
    x = BatchNorm2d(x)
    x = SiLU(x)
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
    name: str | None = None,
) -> tf.Tensor:

    c = x.shape[-1]
    x_conv = x
    x_conv = Conv(x_conv, k=3, s=1, p=True, c=c // 2)
    x_conv = Conv(x_conv, k=3, s=1, p=True, c=c)

    if shortcut:
        x_conv = Add([x, x_conv])

    return x_conv


def Detect(
    x: tf.Tensor,
    reg_max: int,
    nc: int,
    name: str | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    x_pos = x
    x_cls = x
    c = x.shape[-1]

    x_pos = Conv(x_pos, k=3, s=1, p=True, c=c)
    x_pos = Conv(x_pos, k=3, s=1, p=True, c=c)
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
    name: str | None = None,
) -> tf.Tensor:

    x = Conv(x, k=1, s=1, p=False, c=c)
    x_0, x_1 = Split(x)

    xs = [x_0, x_1]
    for i in range(n):
        x_1 = Bottleck(x_1, shortcut=shortcut)
        xs.append(x_1)

    x = Concat(xs)

    x = Conv(x, k=1, s=1, p=False, c=c)

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
) -> tf.Tensor:
    sm = layers.Softmax()
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
    x_cls = Softmax(x_cls)  # determine class percentage-wise

    # Combining
    x = Concat([x_pos, x_cls], axis=-2)

    return x


# ------------------------------------------------------------------------
# Model Itself


# This loss only works for a single class - needs an update, but before that I wanted to commit my changes
def yolo_v8_model(input_size: int, classes: list[str], variant="n"):
    inputs = layers.Input(shape=(input_size, input_size, 3), name="Input")
    classes = list(set(classes))

    variants = {  # d, w, r
        "n": (1 / 3, 0.25, 2),
        "s": (1 / 3, 0.5, 2),
        "m": (2 / 3, 0.75, 1.5),
        "l": (1, 1, 1),
        "x": (1, 1.25, 1),
    }

    d, w, r = variants[variant]
    reg_max = 3
    n_classes = len(classes)  # black / green / red / white / out / nothing

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
    x_12 = C2f(x_11, shortcut=False, n=round(3 * d), c=round(512 * w))
    x_13 = Upsample(x_12)
    x_14 = Concat([x_4, x_13])
    x_15 = C2f(x_14, shortcut=False, n=round(3 * d), c=round(256 * w))  # P3

    x_16 = Conv(x_15, k=3, s=2, p=True, c=round(256 * w))  # P3
    x_17 = Concat([x_12, x_16])
    x_18 = C2f(x_17, shortcut=False, n=round(3 * d), c=round(512 * w))  # P4
    x_19 = Conv(x_18, k=3, s=2, p=True, c=round(512 * w))
    x_20 = Concat([x_9, x_19])
    x_21 = C2f(x_20, shortcut=False, n=round(3 * d), c=round(512 * w * r))  # P5

    # (n, n, 2)
    detect_s_pos, detect_s_cls = Detect(x_21, reg_max=reg_max, nc=n_classes)
    detect_m_pos, detect_m_cls = Detect(x_18, reg_max=reg_max, nc=n_classes)
    detect_l_pos, detect_l_cls = Detect(x_15, reg_max=reg_max, nc=n_classes)

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


class YOLOv8Loss(tf.keras.Loss):
    def __init__(
        self,
        img_size: int,
        square_size: int = 50,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.square_size = square_size
        self.existence_threshold = 0.5
        self.existence_loss = tf.keras.losses.BinaryCrossentropy()

    def call(
        self,
        y_true: tf.Tensor,  # 3x (bs, y, x, 2+n, 3)
        y_pred: tf.Tensor,
    ) -> tf.Tensor:

        # Calculate loss for each scale
        scale_losses = tf.map_fn(
            lambda scl: self.single_loss(y_true[scl], y_pred[scl], scl),
            tf.range(3),
            dtype=tf.float32,
        )

        return tf.reduce_max(scale_losses)

    def single_loss(
        self,
        y_true: tf.Tensor,  # (bs, y, x, 2+n, 3)
        y_pred: tf.Tensor,
        scl: tf.Tensor,  # (1,)
    ) -> tf.Tensor:
        # Split tensors
        pos_true = y_true[..., :2, :]  # (bs, y, x, 2, 3)
        pos_pred = y_pred[..., :2, :]
        cls_true = y_true[..., 2:, :]  # (bs, y, x, n, 3)
        cls_pred = y_pred[..., 2:, :]
        xst_true = self.get_existence(cls_true)  # (bs, y, x, 1, 3)
        xst_pred = self.get_existence(cls_pred)

        # Compute existence loss
        class_loss = self.get_class_loss(cls_true, cls_pred)

        # Compute position loss
        positions_loss = self.get_positions_loss(
            xst_true, pos_true, xst_pred, pos_pred, scl
        )
        return class_loss + positions_loss

    def get_existence(
        self,
        y_cls: tf.Tensor,  # (bs, y, x, n, 3)
    ) -> tf.Tensor:

        # Get best class
        found_classes = tf.argmax(y_cls, axis=-2, output_type=tf.int32)  # (bs, y, x, 3)

        # nothing-class is the last class by design
        nothing_class = tf.shape(y_cls)[-2] - 1
        is_present = tf.cast(
            tf.not_equal(found_classes, nothing_class),
            tf.int32,
        )  # (bs, y, x, 3)

        is_present = tf.expand_dims(is_present, axis=-2)  # (bs, y, x, 1, 3)

        return tf.cast(is_present, tf.float32)

    def get_class_loss(
        self,
        cls_true: tf.Tensor,  # (bs, y, x, nc, 3)
        cls_pred: tf.Tensor,
    ) -> tf.Tensor:
        # Flatten the grid
        bs, s, _, nc, r = tf.shape(cls_true)
        cls_true = tf.reshape(cls_true, (bs, -1, nc, r))  # (bs, y*X, nc, 3)
        cls_pred = tf.reshape(cls_pred, (bs, -1, nc, r))

        # Get all output permutations
        permutations = tf.constant(
            [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)],
            dtype=tf.int32,
        )  # if you can figure out how to tf-vectorize this, go for it!

        losses = []  # -> (r!, bs, y*x)

        perm_losses = tf.map_fn(
            lambda p: self.get_permuted_positions_loss(p, cls_true, cls_pred),
            permutations,
            dtype=tf.float32,
        )  # (r!, bs, y*x)

        # Take minimal loss to determine each correct permutation
        cell_loss = tf.reduce_min(perm_losses, axis=0)  # (b, y*x)

        # The final position loss is the mean of each cell loss
        total_loss = tf.reduce_mean(cell_loss)

        return total_loss

    def get_permuted_positions_loss(
        self,
        perm: tf.Tensor,  # (3,)
        cls_true: tf.Tensor,  # (bs, y*X, nc, 3)
        cls_pred: tf.Tensor,  # (bs, y*X, nc, 3)
    ) -> tf.Tensor:
        # Get permutation (3* = 3, but permuted)
        cls_true_perm = tf.gather(cls_true, perm, axis=-1)  # (bs, y*x, nc, 3*)
        cls_pred_perm = tf.gather(cls_pred, perm, axis=-1)

        # Calculate loss for permutation
        loss = tf.keras.backend.categorical_crossentropy(
            cls_true_perm, cls_pred_perm, axis=-2
        )  # (bs, y*x, 3*)

        # Take mean loss for permutation order
        loss = tf.reduce_mean(loss, axis=-1)  # (bs, y*x)
        return loss

    def get_positions_loss(
        self,
        xst_true: tf.Tensor,  # (bs, y, x, 2, 3)
        pos_true: tf.Tensor,
        xst_pred: tf.Tensor,  # (, bsy, x, 1, 3)
        pos_pred: tf.Tensor,
        scl: int,
    ) -> tf.Tensor:
        # Map to global positions
        pos_true, pos_pred = self.convert_to_absolute_coordinates(
            pos_true, pos_pred, scl
        )  # (bs, y, x, 1, 3)

        # Filter predicted points by confidence
        valid_pos_true_mask = tf.cast(
            xst_true > self.existence_threshold, tf.float32
        )  # (bs, y, x, 1, 3): 0/1
        valid_pos_pred_mask = tf.cast(xst_pred > self.existence_threshold, tf.float32)
        valid_pos_true_mask = tf.repeat(
            valid_pos_true_mask, repeats=2, axis=-2
        )  # (bs, y, x, 2, 3)
        valid_pos_pred_mask = tf.repeat(valid_pos_pred_mask, repeats=2, axis=-2)

        # Apply confidence threshold
        pos_true_filtered = valid_pos_true_mask * pos_true  # (bs, y, x, 2, 3)
        pos_pred_filtered = valid_pos_pred_mask * pos_pred

        # Generate masks
        batch_size = tf.shape(pos_pred_filtered)[0]
        mask_true = tf.map_fn(
            lambda b: self.get_mask(pos_true_filtered[b]),
            tf.range(batch_size),
            tf.bool,
        )
        mask_pred = tf.map_fn(
            lambda b: self.get_mask(pos_pred_filtered[b]),
            tf.range(batch_size),
            tf.bool,
        )

        # Get IoU score
        iou = self.get_iou(mask_true, mask_pred)

        return 1 - iou

    def convert_to_absolute_coordinates(
        self,
        pos_true: tf.Tensor,  # (bs, y, x, 2, 3)
        pos_pred: tf.Tensor,
        scl: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        s = self.img_size // 2 ** (5 - scl)
        grid_size = (s, s)
        cell_size = self.img_size / tf.cast(s, tf.float32)

        grid_indices = tf.stack(
            tf.meshgrid(tf.range(s), tf.range(s), indexing="ij"),
            axis=-1,
        )  # (y, x, 2)
        global_grid_pos = (
            tf.cast(grid_indices, tf.float32) * cell_size
        )  # top-left corners for each cell, (y, x, 2)

        global_pos_true = global_grid_pos[None, :, :, :, None] + pos_true * tf.cast(
            s, tf.float32
        )  # (bs, y, x, 2, 3)
        global_pos_pred = global_grid_pos[None, :, :, :, None] + pos_pred * tf.cast(
            s, tf.float32
        )  # (bs, y, x, 2, 3)

        return global_pos_true, global_pos_pred

    def get_mask(
        self,
        positions,  # (y, x, 2, 3)
    ) -> tf.Tensor:  # bool

        # Filter out masked positions
        positions_idx = tf.where(
            tf.logical_and(positions[:, :, 0, :] != 0, positions[:, :, 1, :] != 0)
        )  # (n, 3)

        # When there are too many points, we just skip the calculation
        # because "masks" might cause OOM error else
        if tf.shape(positions_idx)[0] > 50:
            # positions_idx = positions_idx[:50]
            return tf.ones((self.img_size, self.img_size), tf.bool)

        # Add x and y indices into dimension 2
        pad = tf.zeros_like(positions_idx[:, :1])
        x_ids = tf.concat(
            [positions_idx[:, :2], pad + 0, positions_idx[:, 2:]], axis=1
        )  # (n, 4)
        y_ids = tf.concat([positions_idx[:, :2], pad + 1, positions_idx[:, 2:]], axis=1)
        x_pos = tf.gather_nd(positions, x_ids)  # (n,)
        y_pos = tf.gather_nd(positions, y_ids)

        # Get min and max values
        half_size = self.square_size / 2
        y_min = tf.clip_by_value(y_pos - half_size, 0, self.img_size - 1)  # (n,)
        y_max = tf.clip_by_value(y_pos + half_size, 0, self.img_size - 1)
        x_min = tf.clip_by_value(x_pos - half_size, 0, self.img_size - 1)
        x_max = tf.clip_by_value(x_pos + half_size, 0, self.img_size - 1)

        # Get image positions
        y_indices, x_indices = tf.meshgrid(
            tf.range(self.img_size), tf.range(self.img_size), indexing="ij"
        )
        indices = tf.cast(
            tf.stack([y_indices, x_indices], axis=-1), tf.float32
        )  # (800, 800, 2)

        # Apply square conditions
        # (1, 800, 800) x (n, 1, 1) = (n, 800, 800)
        masks = tf.logical_and(
            tf.logical_and(  # square height constraints
                indices[None, ..., 0] >= y_min[:, None, None],
                indices[None, ..., 0] <= y_max[:, None, None],
            ),
            tf.logical_and(  # square width constraints
                indices[None, ..., 1] >= x_min[:, None, None],
                indices[None, ..., 1] <= x_max[:, None, None],
            ),
        )  # (n, 800, 800)

        final_mask = tf.reduce_any(masks, axis=0)  # (800, 800)
        return final_mask

    def get_iou(
        self,
        mask_true: tf.Tensor,  # bool
        mask_pred: tf.Tensor,  # bool
    ) -> tf.Tensor:

        intersection = tf.reduce_sum(
            tf.cast(tf.logical_and(mask_true, mask_pred), tf.float32)
        )
        union = tf.reduce_sum(tf.cast(tf.logical_or(mask_true, mask_pred), tf.float32))

        iou = tf.math.divide_no_nan(intersection, union + 1e-6)

        return iou
