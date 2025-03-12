import cv2
import numpy as np
from time import time
from tensorflow.keras.callbacks import Callback  # type: ignore
from tensorflow.keras.activations import softmax


class PredictionCallback(Callback):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        output_file: str = "outputs/training_outputs.png",
        update_on: str = "batches",
        update_frequency: int | float = 10,
    ):
        super().__init__()

        self.X = X  # (bs, 800, 800, 3)
        self.y = y  # (bs, 25, 25, 8, 3)

        self.output_file = output_file

        # Updating
        self.update_functions = {
            "seconds": self._update_on_time,
            "batches": self._update_on_batches,
        }
        if update_on not in self.update_functions.keys():
            raise ValueError(
                f"Invalid update type: {update_on}. Has to be one of {self.update_functions.keys()}."
            )
        self.update_fn = self.update_functions[update_on]
        self.update_frequency = update_frequency
        self.last_update = time()
        print(
            f"Predicting every {self.update_frequency} {update_on} to file: {self.output_file}"
        )

        self.epoch = 0

        # Add grid
        cs = self.X.shape[1] // self.y.shape[1]
        self.imgs = np.uint8(self.X * 255)
        self.imgs[:, 0::cs] //= 4
        self.imgs[:, cs - 1 :: cs] //= 4
        self.imgs[:, :, 0::cs] //= 4
        self.imgs[:, :, cs - 1 :: cs] //= 4

        xst_cell = np.zeros((32, 32, 3), np.uint8)
        n = 12
        xst_cell[0, :n] = 255
        xst_cell[0, -n:] = 255
        xst_cell[-1, :n] = 255
        xst_cell[-1, -n:] = 255
        xst_cell[:n, 0] = 255
        xst_cell[-n:, 0] = 255
        xst_cell[:n, -1] = 255
        xst_cell[-n:, -1] = 255

        xst_cell = np.concatenate([xst_cell for _ in range(25)], axis=0)  # (800, 25)
        xst_cell = np.concatenate([xst_cell for _ in range(25)], axis=1)  # (800, 800)
        self.xst_overlay = xst_cell

        # self.init_grid_imgs()

    # def init_grid_imgs(self):
    #     self.grid_imgs = {}
    #     for s in [y.shape[1] for y in self.y]:
    #         img = np.uint8(self.X * 255)
    #         cs = img.shape[1] // s
    #         img[:, 0::cs] //= 4
    #         img[:, cs - 1 :: cs] //= 4
    #         img[:, :, 0::cs] //= 4
    #         img[:, :, cs - 1 :: cs] //= 4
    #         self.grid_imgs[str(s)] = img

    # def get_grid_img(self, batch: int, grid_size: int):
    #     return self.grid_imgs[str(grid_size)][batch].copy()

    def _update_on_batches(self, batch, *args, **kwargs):
        return batch % self.update_frequency == 0

    def _update_on_time(self, batch, *args, **kwargs):
        if batch == 0 or time() - self.last_update > self.update_frequency:
            self.last_update = time()
            return True

        return False

    def get_xst_tile(
        self,
        img: np.ndarray,  # (800, 800, 3)
        xst_true: np.ndarray,  # (s, s, 1, 3)
        xst_pred: np.ndarray,  # (s, s, 1, 3)
    ):
        # Get existences
        mask_true = np.max(xst_true, axis=-1)  # (s, s, 1)
        mask_pred = np.max(xst_pred, axis=-1)
        mask_true = np.repeat(mask_true, 3, axis=-1)  # (s, s, 3)
        mask_pred = np.repeat(mask_pred, 3, axis=-1)
        mask_true = np.uint8(mask_true * 255)
        mask_pred = np.uint8(mask_pred * 255)
        mask_pred[..., 1:] = 0

        # Extend mask to image size
        mask_true = np.kron(mask_true, np.ones((32, 32, 1), np.uint8))  # (800, 800, 3)
        mask_pred = np.kron(mask_pred, np.ones((32, 32, 1), np.uint8))

        # Add prediction mask
        out = cv2.addWeighted(img, 0.5, mask_pred, 0.5, 1.0)

        # Add true outlines
        mask_true *= self.xst_overlay
        out = np.maximum(out, mask_true * self.xst_overlay)
        return out

    def get_cls_tile(
        self,
        img: np.ndarray,  # ()
        xst_true: np.ndarray,  # (s, s, 1, 3)
        xst_pred: np.ndarray,
        cls_true: np.ndarray,  # (s, s, 5, 3)
        cls_pred: np.ndarray,
    ):
        # Get existences
        mask_true = np.max(xst_true, axis=-1)  # (s, s, 1)
        mask_true = np.repeat(mask_true, 3, axis=-1)  # (s, s, 3)
        mask_true = np.uint8(mask_true * 255)
        mask_true = np.kron(mask_true, np.ones((32, 32, 1), np.uint8))  # (800, 800, 3)

        # Apply activation
        cls_pred = softmax(cls_pred, axis=-2)

        # Mask predictinos
        cls_true *= xst_true
        cls_pred *= xst_true

        colors = [
            (50, 50, 50),  # black
            (255, 255, 255),  # white
            (0, 0, 255),  # red
            (0, 255, 0),  # green
            (127, 127, 127),  # out
        ]

        mask = np.zeros((800, 800, 3), np.uint8)

        cls_true = np.round(30 * cls_true).astype(np.int32)
        cls_pred = np.round(30 * cls_pred).astype(np.int32)
        for y in range(25):
            row_true = cls_true[y]
            row_pred = cls_pred[y]
            y_pos = (y + 1) * 32 - 1
            for x in range(25):
                cell_true = row_true[x]
                cell_pred = row_pred[x]
                for i in range(3):
                    idx_true = cell_true[:, i]
                    idx_pred = cell_pred[:, i]
                    for cls in range(5):
                        bar_true = idx_true[cls]
                        bar_pred = idx_pred[cls]
                        x_pos = x * 32 + 1 + i * 5 * 2 + cls * 2
                        mask[y_pos - bar_true : y_pos, x_pos] = colors[cls]
                        mask[y_pos - bar_pred : y_pos, x_pos + 1] = colors[cls]

        out = cv2.addWeighted(img, 0.5, mask, 0.5, 1.0)

        mask_true *= self.xst_overlay
        out = np.maximum(out, mask_true * self.xst_overlay)
        return out

    #     grid_size = cls_true.shape[0]
    #     img = self.get_grid_img(batch, grid_size)

    #     overlay = np.zeros((grid_size * 8, grid_size * 8), np.float32)
    #     for y, (cls_true_row, cls_pred_row) in enumerate(zip(cls_true, cls_pred)):
    #         y *= 8
    #         for x, (cell_true, cell_pred) in enumerate(zip(cls_true_row, cls_pred_row)):
    #             x *= 8
    #             if np.all(cell_pred[0] > cell_pred[1:]) and np.all(cell_true[0] == 1):
    #                 continue
    #             # Nothing class
    #             overlay[y + 1, x + 1 : x + 4] = 0.5 + cell_true[0] / 2
    #             overlay[y + 1, x + 5 : x + 8] = 0.5 + cell_pred[0] / 2
    #             # Other classes
    #             overlay[y + 3 : y + 8, x + 1 : x + 4] = 0.5 + cell_true[1:] / 2
    #             overlay[y + 3 : y + 8, x + 5 : x + 8] = 0.5 + cell_pred[1:] / 2
    #     overlay = np.uint8(overlay * 255)
    #     overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_PLASMA)
    #     overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    #     # Light strips
    #     overlay[2::8, ..., -1] = 0
    #     overlay[:, 4::8, ..., -1] = 0
    #     # Dark strips
    #     overlay[::8, ..., -1] = 0
    #     overlay[:, ::8, ..., -1] = 0

    #     scl = img.shape[0] // overlay.shape[0]
    #     overlay = np.kron(overlay, np.ones((scl, scl, 1), np.uint8))

    #     # Combine images
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    #     img = cv2.addWeighted(img, 1.0, overlay, 0.75, 1.0)

    #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    #     return img

    # def get_pos_tile(self, pos_true, pos_pred):
    #     grid_size = pos_true.shape[0]
    #     img = self.get_grid_img(grid_size)
    #     return img

    # def get_scaled_output(
    #     self,
    #     batch: int,
    #     y_true: np.ndarray,  # (s, s, 8, 3)
    #     y_pred: np.ndarray,
    # ):
    #     tiles = []

    #     # Split into parts
    #     pos_true = y_true[:, :, :2, :]  # (s, s, 2, 3)
    #     pos_pred = y_true[:, :, :2, :]
    #     cls_true = y_true[:, :, 2:, :]  # (s, s, 6, 3)
    #     cls_pred = y_pred[:, :, 2:, :]
    #     xst_true = np.max(np.argmax(cls_true, axis=-2) != 0, axis=-1)  # (s, s) bool
    #     xst_pred = np.max(np.argmax(cls_pred, axis=-2) != 0, axis=-1)

    #     xst_tile = self.get_xst_tile(batch, xst_true, xst_pred)
    #     cls_tile = self.get_cls_tile(batch, cls_true, cls_pred)
    #     # pos_tile = self.get_pos_tile(batch, pos_true, pos_pred)

    #     out_tile = np.concatenate(
    #         [
    #             xst_tile,
    #             cls_tile,
    #             # pos_tile,
    #         ],
    #         axis=0,
    #     )

    #     return out_tile

    def plot_prediction(self, y_pred=None):
        y_pred = y_pred or self.model.predict(self.X, verbose=0)  # (bs, s, s, 8, 3)

        # Get output tiles
        batch_imgs = []
        for b in range(self.X.shape[0]):
            img = self.imgs[b]

            xst_true = self.y[b, ..., 0:1, :]  # (s, s, 1, 3)
            pos_true = self.y[b, ..., 1:3, :]  # (s, s, 2, 3)
            cls_true = self.y[b, ..., 3:, :]  # (s, s, 5, 3)

            xst_pred = y_pred[b, ..., 0:1, :]
            pos_pred = y_pred[b, ..., 1:3, :]
            cls_pred = y_pred[b, ..., 3:, :]

            batch_img = np.concatenate(
                [
                    self.get_xst_tile(img, xst_true, xst_pred),
                    # self.get_pos_tile(img, xst_true, xst_pred, pos_true, pos_pred),
                    self.get_cls_tile(img, xst_true, xst_pred, cls_true, cls_pred),
                    # self.get_diou_tile(self.X[b], self.y[b], y_pred[b]),
                ],
                axis=0,
            )
            batch_imgs.append(batch_img)
        out_img = np.concatenate(batch_imgs, axis=1)
        cv2.imwrite(self.output_file, out_img)

    # def on_train_batch_end(self, batch, logs={}):
    #     if self.update_fn(batch):
    #         self.plot_prediction()

    def on_epoch_end(self, epoch, logs=None):
        self.plot_prediction()
