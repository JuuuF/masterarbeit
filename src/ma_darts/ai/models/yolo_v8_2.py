import tensorflow as tf
from tensorflow.keras import layers, Model
from ma_darts import classes as default_classes
from ma_darts import img_size as default_img_size
from ma_darts.ai.data import dummy_ds

from keras.src.ops.operation_utils import compute_conv_output_shape


class Conv(layers.Layer):
    def __init__(
        self,
        k: int,
        s: int,
        p: bool,
        c: int,
        dropout: float = 0.0,
        dropout_type="regular",
    ):
        super(Conv, self).__init__()
        self.k = k
        self.c = c
        self.s = s
        self.p = p
        self.dropout = dropout
        self.dropout_type = dropout_type
        if dropout_type not in ["regular", "spatial"]:
            raise ValueError(
                f"Invalid dropout: {dropout_type}. "
                "Has to be one of ['regular', 'spatial']."
            )

    def build(self):
        self.conv2d = layers.Conv2D(
            filters=self.c,
            kernel_size=(self.k, self.k),
            strides=(self.s, self.s),
            padding="same" if self.p else "valid",
        )
        self.batchnorm2d = layers.BatchNormalization()
        self.silu = layers.Activation("silu")

        if self.dropout_type == "regular":
            self.dropout_layer = layers.Dropout(self.dropout)
        elif self.dropout_type == "spatial":
            self.dropout_layer = layers.SpatialDropout2D(self.dropout)
        else:
            raise RuntimeError("I've f#cked up.")

    def call(self, inputs, training=False):
        x = self.conv2d(inputs)
        x = self.batchnorm2d(x, training=training)
        x = self.silu(x)
        x = self.dropout_layer(x, training=training)

        return x

    def get_config(self):
        config = super(Conv, self).get_config()
        config.update(
            {
                "k": self.k,
                "c": self.c,
                "s": self.s,
                "p": self.p,
                "dropout": self.dropout,
                "dropout_type": self.dropout_type,
            }
        )
        return config

    # def compute_output_shape(self, input_shape):
    #     return compute_conv_output_shape(
    #         input_shape,
    #         filters=self.c,
    #         kernel_size=self.k,
    #         strides=self.s,
    #         padding="same" if self.p else "valid",
    #     )


class Bottleneck(layers.Layer):
    def __init__(
        self,
        c: int,
        shortcut: bool,
        expansion: float = 0.5,
        dropout: float = 0.0,
    ):
        super(Bottleneck, self).__init__()
        self.c = c
        self.shortcut = shortcut
        self.expansion = expansion
        self.dropout = dropout

    def build(self):
        hidden_dim = int(self.c * self.expansion)
        self.conv_1 = Conv(k=3, s=1, p=True, c=hidden_dim)
        self.conv_2 = Conv(k=3, s=1, p=True, c=self.c)
        if self.shortcut:
            self.add = layers.Add()

    def call(self, inputs, training=False):
        x = self.conv_1(inputs, training=training)
        x = self.conv_2(x, training=training)
        return self.add([x, inputs]) if self.shortcut else x

    def get_config(self):
        config = super(Bottleneck, self).get_config()
        config.update(
            {
                "c": self.c,
                "shortcut": self.shortcut,
                "expansion": self.expansion,
                "dropout": self.dropout,
            }
        )
        return config


class SPPF(layers.Layer):
    def __init__(
        self,
        c: int,
        pool_size: int,
    ):
        super(SPPF, self).__init__()
        self.c = c
        self.pool_size = pool_size

    def build(self):
        self.conv_1 = Conv(k=1, s=1, p=False, c=self.c)
        self.pool = layers.MaxPool2D(
            pool_size=(self.pool_size, self.pool_size),
            strides=(1, 1),
            padding="same",
        )
        self.conv_2 = Conv(k=1, s=1, p=False, c=self.c)

    def call(self, inputs, training=False):
        x = self.conv_1(inputs, training=training)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        x = tf.concat([x, p1, p2, p3], axis=-1)
        x = self.conv_2(x, training=training)
        return x

    def get_config(self):
        config = super(SPPF, self).get_config()
        config.update(
            {
                "c": self.c,
                "pool_size": self.pool_size,
            }
        )
        return config


class C2f(layers.Layer):
    def __init__(
        self,
        shortcut: bool,
        n: int,
        c: int,
        dropout: float = 0.0,
    ):
        super(C2f, self).__init__()
        self.shortcut = shortcut
        self.n = n
        self.c = c
        self.dropout = dropout

    def build(self):
        split_c = self.c // 2
        self.conv_1 = Conv(k=1, s=1, p=False, c=self.c)
        self.conv_2 = Conv(k=1, s=1, p=False, c=self.c)
        self.dropout = layers.Dropout(self.dropout)

        self.bottlenecks = [
            Bottleneck(c=split_c, shortcut=self.shortcut, dropout=self.dropout)
            for _ in range(self.n)
        ]

    def call(self, inputs, training=False):
        x = self.conv_1(inputs, training=training)
        x_1, x_2 = tf.split(x, num_or_size_splits=2, axis=-1)
        concats = [x_1, x_2]
        for bottleneck in self.bottlenecks:
            x_2 = bottleneck(x_2, training=training)
            concats.append(x_2)
        x = tf.concat(concats, axis=-1)
        x = self.conv_2(x, training=training)
        return x

    def get_config(self):
        config = super(C2f, self).get_config()
        config.update(
            {
                "shortcut": self.shortcut,
                "n": self.n,
                "c": self.c,
                "dropout": self.dropout,
            }
        )
        return config


class Detect(layers.Layer):
    def __init__(
        self,
        c: int,
        n_anchors: int,
        n_classes: int,
        dropout: float = 0.0,
    ):
        super(Detect, self).__init__()
        self.c = c
        self.n_anchors = n_anchors
        self.n_classes = n_classes
        self.dropout = dropout

    def build(self):
        self.conv_pos_1 = Conv(k=3, s=1, p=True, c=self.c)
        self.conv_pos_2 = Conv(k=3, s=1, p=True, c=self.c)
        self.conv_pos_3 = layers.Conv2D(
            filters=2 * self.n_anchors,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
        )
        self.hard_sigmoid = layers.Activation("hard_sigmoid")

        self.conv_cls_1 = Conv(k=3, s=1, p=True, c=self.c)
        self.conv_cls_2 = Conv(k=3, s=1, p=True, c=self.c)
        self.conv_cls_3 = layers.Conv2D(
            filters=self.n_anchors * self.n_classes,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
        )
        self.softmax = layers.Softmax(axis=-2)

    def call(self, inputs, training=False):
        # Position branch
        x_pos = self.conv_pos_1(inputs, training=training)
        x_pos = self.conv_pos_2(x_pos, training=training)
        x_pos = self.conv_pos_3(x_pos, training=training)
        x_pos = tf.reshape(
            x_pos, shape=x_pos.shape[:-1] + (2, self.n_anchors)
        )  # (s, s, 2, 3)
        x_pos = self.hard_sigmoid(x_pos)

        # Class branch
        x_cls = self.conv_cls_1(inputs, training=training)
        x_cls = self.conv_cls_2(x_cls, training=training)
        x_cls = self.conv_cls_3(x_cls, training=training)
        x_cls = tf.reshape(
            x_cls, shape=x_cls.shape[:-1] + (self.n_classes, self.n_anchors)
        )  # (s, s, n, 3)
        y_cls = self.softmax(x_cls)

        # Combine
        x = tf.concat([x_pos, x_cls], axis=-2)

        return x

    def get_config(self):
        config = super(Detect, self).get_config()
        config.update(
            {
                "c": self.c,
                "n_anchors": self.n_anchors,
                "n_classes": self.n_classes,
                "dropout": self.dropout,
            }
        )
        return config


# model = tf.keras.Sequential(
#     [
#         # layers.Input((800, 800, 3)),
#         # Conv(k=3, s=1, p=True, c=8, dropout=0.0, dropout_type="regular"),
#         # Conv(k=5, s=1, p=True, c=16, dropout=0.0, dropout_type="regular"),
#         # Bottleneck(c=16, shortcut=False, dropout=0.0),
#         # Bottleneck(c=16, shortcut=True, dropout=0.0),
#         # SPPF(c=16, pool_size=5),
#         # SPPF(c=16, pool_size=9),
#         # C2f(shortcut=True, n=2, c=16, dropout=0.0),
#         # C2f(shortcut=False, n=3, c=16, dropout=0.0),
#     ]
# )
# model.compile(optimizer="adam", loss="mse")
# model.summary()
# print(model.get_config())
# exit()


class YOLOv8(Model):
    def __init__(
        self,
        classes: list[str],
        variant: str = "n",
    ):
        super(YOLOv8, self).__init__()
        self.variant = variant

        self.classes = list(set(classes))
        if "nothing" in self.classes:
            self.classes.remove("nothing")
        self.classes.insert(0, "nothing")
        self.n_classes = len(self.classes)

        # def build(self, input_shape):

        d, w, r = self.get_variant()

        # Backbone
        dropout = 0.2
        c_out = round(64 * w)
        self.conv_0 = Conv(k=3, s=2, p=True, c=c_out, dropout=dropout)  # (400, 400, 16)
        c_out = round(128 * w)
        self.conv_1 = Conv(k=3, s=2, p=True, c=c_out, dropout=dropout)  # (200, 200, 32)
        self.c2f_2 = C2f(shortcut=True, n=int(3 * d), c=c_out, dropout=dropout)
        c_out = round(256 * w)
        self.conv_3 = Conv(k=3, s=2, p=True, c=c_out, dropout=dropout)  # (100, 100, 64)
        self.c2f_4 = C2f(shortcut=True, n=int(6 * d), c=c_out, dropout=dropout)
        c_out = round(512 * w)
        self.conv_5 = Conv(k=3, s=2, p=True, c=c_out, dropout=dropout)  # (50, 50, 128)
        self.c2f_6 = C2f(shortcut=True, n=int(6 * d), c=c_out, dropout=dropout)
        c_out = round(512 * w * r)
        self.conv_7 = Conv(k=3, s=2, p=True, c=c_out, dropout=dropout)  # (25, 25, 256)
        self.c2f_8 = C2f(shortcut=True, n=int(3 * d), c=c_out, dropout=dropout)

        # Neck
        self.sppf = SPPF(c=c_out, pool_size=5)  # (25, 25, 256)

        # Head
        dropout = 0.1
        self.upsampling = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.c2f_12 = C2f(
            shortcut=False, n=int(3 * d), c=round(512 * w), dropout=dropout
        )
        self.c2f_15 = C2f(
            shortcut=False, n=int(3 * d), c=round(256 * w), dropout=dropout
        )

        self.conv_16 = Conv(k=3, s=2, p=True, c=round(256 * w), dropout=dropout)
        self.c2f_18 = C2f(
            shortcut=False, n=int(3 * d), c=round(512 * w), dropout=dropout
        )
        self.conv_19 = Conv(k=3, s=2, p=True, c=round(512 * w), dropout=dropout)
        self.c2f_21 = C2f(
            shortcut=False, n=int(3 * d), c=round(512 * w * r), dropout=dropout
        )

        # Detect
        self.detect_l = Detect(c=round(256 * w), n_anchors=3, n_classes=self.n_classes)
        self.detect_m = Detect(c=round(512 * w), n_anchors=3, n_classes=self.n_classes)
        self.detect_s = Detect(
            c=round(512 * w * r), n_anchors=3, n_classes=self.n_classes
        )

    def get_variant(self):
        variants = {  # d, w, r
            "n": (1 / 3, 0.25, 2),
            "s": (1 / 3, 0.5, 2),
            "m": (2 / 3, 0.75, 1.5),
            "l": (1, 1, 1),
            "x": (1, 1.25, 1),
        }
        return variants[self.variant]

    @tf.function
    def call(self, inputs, training=False):
        # Backbone
        x_0 = self.conv_0(inputs, training=training)
        x_1 = self.conv_1(x_0, training=training)
        x_2 = self.c2f_2(x_1, training=training)
        x_3 = self.conv_3(x_2, training=training)
        x_4 = self.c2f_4(x_3, training=training)
        x_5 = self.conv_5(x_4, training=training)
        x_6 = self.c2f_6(x_5, training=training)
        x_7 = self.conv_7(x_6, training=training)
        x_8 = self.c2f_8(x_7, training=training)

        # Neck
        x_9 = self.sppf(x_8, training=training)

        # Head
        x_10 = self.upsampling(x_9)
        x_11 = tf.concat([x_6, x_10], axis=-1)
        x_12 = self.c2f_12(x_11, training=training)
        x_13 = self.upsampling(x_12)
        x_14 = tf.concat([x_4, x_13], axis=-1)
        x_15 = self.c2f_15(x_14, training=training)

        x_16 = self.conv_16(x_15, training=training)
        x_17 = tf.concat([x_12, x_16], axis=-1)
        x_18 = self.c2f_18(x_17, training=training)
        x_19 = self.conv_19(x_18, training=training)
        x_20 = tf.concat([x_9, x_19], axis=-1)
        x_21 = self.c2f_21(x_20, training=training)

        # Detect
        x_l = self.detect_l(x_15, training=training)  # (100, 100, 2+6, 3),
        x_m = self.detect_m(x_18, training=training)  # (50, 50, 2+6, 3)
        x_s = self.detect_s(x_21, training=training)  # (25, 25, 2+6, 3)

        return x_s, x_m, x_l

    def get_config(self):
        config = super(YOLOv8, self).get_callbacks()
        config.update(
            {
                "classes": self.classes,
                "variant": self.variant,
            }
        )
        return config


def get_yolo_v8(
    img_size: int = default_img_size,
    classes: list[str] = default_classes,
    variant: str = "n",
):
    inputs = layers.Input(shape=(img_size, img_size, 3))

    variants = {  # d, w, r
        "n": (1 / 3, 0.25, 2),
        "s": (1 / 3, 0.5, 2),
        "m": (2 / 3, 0.75, 1.5),
        "l": (1, 1, 1),
        "x": (1, 1.25, 1),
    }
    d, w, r = variants[variant]

    # Backbone
    dropout = 0.2
    c_out = round(64 * w)
    x_0 = Conv(k=3, s=2, p=True, c=c_out, dropout=dropout)(inputs)  # (400, 400, 16)

    c_out = round(128 * w)
    x_1 = Conv(k=3, s=2, p=True, c=c_out, dropout=dropout)(x_0)  # (200, 200, 32)
    x_2 = C2f(shortcut=True, n=int(3 * d), c=c_out, dropout=dropout)(x_1)

    c_out = round(256 * w)
    x_3 = Conv(k=3, s=2, p=True, c=c_out, dropout=dropout)(x_2)  # (100, 100, 64)
    x_4 = C2f(shortcut=True, n=int(6 * d), c=c_out, dropout=dropout)(x_3)

    c_out = round(512 * w)
    x_5 = Conv(k=3, s=2, p=True, c=c_out, dropout=dropout)(x_4)  # (50, 50, 128)
    x_6 = C2f(shortcut=True, n=int(6 * d), c=c_out, dropout=dropout)(x_5)

    c_out = round(512 * w * r)
    x_7 = Conv(k=3, s=2, p=True, c=c_out, dropout=dropout)(x_6)  # (25, 25, 256)
    x_8 = C2f(shortcut=True, n=int(3 * d), c=c_out, dropout=dropout)(x_7)

    # Neck
    x_9 = SPPF(c=c_out, pool_size=5)(x_8)  # (25, 25, 256)

    # Head
    dropout = 0.1
    x_10 = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x_9)
    x_11 = layers.Concatenate(axis=-1)([x_6, x_10])
    x_12 = C2f(shortcut=False, n=int(3 * d), c=round(512 * w), dropout=dropout)(x_11)
    x_13 = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x_12)
    x_14 = layers.Concatenate(axis=-1)([x_4, x_13])
    x_15 = C2f(shortcut=False, n=int(3 * d), c=round(256 * w), dropout=dropout)(x_14)

    x_16 = Conv(k=3, s=2, p=True, c=round(256 * w), dropout=dropout)(x_15)
    x_17 = layers.Concatenate(axis=-1)([x_12, x_16])
    x_18 = C2f(shortcut=False, n=int(3 * d), c=round(512 * w), dropout=dropout)(x_17)
    x_19 = Conv(k=3, s=2, p=True, c=round(512 * w), dropout=dropout)(x_18)
    x_20 = layers.Concatenate(axis=-1)([x_9, x_19])
    x_21 = C2f(shortcut=False, n=int(3 * d), c=round(512 * w * r), dropout=dropout)(
        x_20
    )

    # Detect
    n_classes = len(classes)
    out_l = Detect(c=round(256 * w), n_anchors=3, n_classes=n_classes)(
        x_15
    )  # (100, 100, 2+6, 3)
    out_M = Detect(c=round(512 * w), n_anchors=3, n_classes=n_classes)(
        x_18
    )  # (50, 50, 2+6, 3),
    out_s = Detect(c=round(512 * w * r), n_anchors=3, n_classes=n_classes)(
        x_21
    )  # (25, 25, 2+6, 3),

    yolo = Model(inputs=inputs, outputs=[out_s, out_m, out_l])
    return yolo


if __name__ == "__main__":
    yolo = get_yolo_v8(
        classes=["nothing", "black", "white", "red", "green", "out"],
        variant="n",
    )
    yolo.compile(optimizer="adam", loss="mse")
    yolo.build(input_shape=(800, 800, 3))
    yolo.summary()

    ds = dummy_ds()

    try:
        yolo.fit(ds, epochs=5, steps_per_epoch=256)
    except KeyboardInterrupt:
        print()
    yolo.summary()

    # import numpy as np

    # for _ in range(10):
    #     X = np.random.random((4, 800, 800, 3))
    #     y = yolo.predict(X)
    #     print(len(y))
    #     for y_ in y:
    #         print(y_.shape)
    # yolo.summary()
