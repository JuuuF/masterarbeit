import tensorflow as tf
from tensorflow.keras import layers


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
        super().__init__()
        self.conv2d = layers.Conv2D(
            filters=c,
            kernel_size=(k, k),
            strides=(s, s),
            padding="same" if p else "valid",
        )
        self.batchnorm2d = layers.BatchNormalization()
        self.silu = layers.Activation("silu")
        if dropout_type == "regular":
            self.dropout = layers.Dropout(dropout)
        elif dropout_type == "spatial":
            self.dropout = layers.SpatialDropout2D(dropout)
        else:
            raise ValueError(
                f"Invalid dropout: {dropout_type}. "
                "Has to be one of ['regular', 'spatial']."
            )

    def call(self, inputs, training=False):
        x = self.conv2d(inputs)
        x = self.batchnorm2d(x, training=training)
        if training:
            x = self.dropout(x, training=training)
        x = self.silu(x)

        return x


class Bottleneck(layers.Layer):
    def __init__(
        self,
        c: int,
        shortcut: bool,
        expansion: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = int(c * expansion)

        self.conv_1 = Conv(k=3, s=1, p=True, c=hidden_dim)
        self.conv_2 = Conv(k=3, s=1, p=True, c=c)
        self.shortcut = shortcut

    def call(self, inputs, training=False):
        x = self.conv_1(inputs, training=training)
        x = self.conv_2(x, training=training)
        return x + inputs if self.shortcut else x


class SPPF(layers.Layer):
    def __init__(
        self,
        c: int,
        pool_size: int,
    ):
        super().__init__()
        self.conv_1 = Conv(k=1, s=1, p=False, c=c)
        self.pool = layers.MaxPool2D(
            pool_size=(pool_size, pool_size), strides=(1, 1), padding="same"
        )
        self.conv_2 = Conv(k=1, s=1, p=False, c=c)

    def call(self, inputs, training=False):
        x = self.conv_1(inputs, training=training)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        x = tf.concat([x, p1, p2, p3], axis=-1)
        x = self.conv_2(x, training=training)
        return x


class C2f(layers.Layer):
    def __init__(
        self,
        shortcut: bool,
        n: int,
        c: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        split_c = c // 2
        self.conv_1 = Conv(k=1, s=1, p=False, c=c)
        self.conv_2 = Conv(k=1, s=1, p=False, c=c)
        self.dropout = layers.Dropout(dropout)

        self.bottlenecks = [
            Bottleneck(c=split_c, shortcut=shortcut, dropout=dropout) for _ in range(n)
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


class Detect(layers.Layer):
    def __init__(
        self,
        c: int,
        n_anchors: int,
        n_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_anchors = n_anchors
        self.n_classes = n_classes

        self.conv_pos_1 = Conv(k=3, s=1, p=True, c=c)
        self.conv_pos_2 = Conv(k=3, s=1, p=True, c=c)
        self.conv_pos_3 = layers.Conv2D(
            filters=2 * n_anchors, kernel_size=(1, 1), strides=(1, 1), padding="valid"
        )
        self.hard_sigmoid = layers.Activation("hard_sigmoid")

        self.conv_cls_1 = Conv(k=3, s=1, p=True, c=c)
        self.conv_cls_2 = Conv(k=3, s=1, p=True, c=c)
        self.conv_cls_3 = layers.Conv2D(
            filters=n_anchors * n_classes,
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


class YOLOv8(tf.keras.Model):
    def __init__(self, input_size: int, classes: list[str], variant: str = "n"):
        super().__init__()
        classes = list(set(classes))
        if "nothing" in classes:
            classes.remove("nothing")
        classes.insert(0, "nothing")
        self.classes = classes
        self.n_classes = len(self.classes)

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


def dummy_ds(batch_size=4):
    # Define a single sample
    input_sample = tf.random.uniform((800, 800, 3), dtype=tf.float32)
    output_sample = (
        tf.random.uniform((25, 25, 8, 3), dtype=tf.float32),
        tf.random.uniform((50, 50, 8, 3), dtype=tf.float32),
        tf.random.uniform((100, 100, 8, 3), dtype=tf.float32),
    )

    # Expand dims to match batch size
    def batch_fn():
        return (
            tf.expand_dims(input_sample, axis=0),  # (1, 800, 800, 3)
            tuple(
                tf.expand_dims(t, axis=0) for t in output_sample
            ),  # [(1, 25, 25, 8, 3), ...]
        )

    # Create dataset from tensors
    dataset = tf.data.Dataset.from_tensors(batch_fn()).repeat()

    # Batch the dataset
    dataset = dataset.map(
        lambda x, y: (
            tf.tile(x, [batch_size, 1, 1, 1]),
            tuple(tf.tile(t, [batch_size, 1, 1, 1, 1]) for t in y),
        )
    )

    return dataset


if __name__ == "__main__":
    yolo = YOLOv8(
        input_size=800,
        classes=["nothing", "black", "white", "red", "green", "out"],
        variant="n",
    )
    yolo.build(input_shape=(None, 800, 800, 3))
    yolo.compile(optimizer="adam", loss="mse")
    yolo.summary()

    ds = dummy_ds()
    yolo.fit(ds, epochs=5, steps_per_epoch=16)

    # import numpy as np

    # for _ in range(10):
    #     X = np.random.random((4, 800, 800, 3))
    #     y = yolo.predict(X)
    #     print(len(y))
    #     for y_ in y:
    #         print(y_.shape)
    # yolo.summary()
