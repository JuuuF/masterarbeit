import cv2
import numpy as np
from time import time
from tensorflow.keras.callbacks import Callback  # type: ignore


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
        self.y = y  # (bs, s, s, 8, 3) for s=25, 50, 100

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

        self.init_grid_imgs()

    def init_grid_imgs(self):
        self.grid_imgs = {}
        for s in [y.shape[1] for y in self.y]:
            img = np.uint8(self.X * 255)
            cs = img.shape[1] // s
            img[:, 0::cs] //= 4
            img[:, cs - 1 :: cs] //= 4
            img[:, :, 0::cs] //= 4
            img[:, :, cs - 1 :: cs] //= 4
            self.grid_imgs[str(s)] = img

    def get_grid_img(self, batch: int, grid_size: int):
        return self.grid_imgs[str(grid_size)][batch].copy()

    def _update_on_batches(self, batch, *args, **kwargs):
        return batch % self.update_frequency == 0

    def _update_on_time(self, batch, *args, **kwargs):
        if batch == 0 or time() - self.last_update > self.update_frequency:
            self.last_update = time()
            return True

        return False

    def get_xst_tile(
        self,
        batch: int,
        xst_true: np.ndarray,  # (s, s)
        xst_pred: np.ndarray,
    ):
        grid_size = xst_true.shape[0]
        img = self.get_grid_img(batch, grid_size)

        mask = np.zeros((grid_size, grid_size, 3))  # (s, s, 3)

        true_preds = np.logical_and(xst_true, xst_pred)
        combined = np.logical_xor(xst_true, xst_pred)
        missed_trues = np.logical_and(combined, xst_true)
        wrong_preds = np.logical_and(combined, xst_pred)

        mask[missed_trues, 0] = 1  # missed = blue
        mask[true_preds, 1] = 1  # correct = green
        mask[wrong_preds, 2] = 1  # wrong = red

        cell_size = img.shape[0] // grid_size
        mask = np.kron(mask, np.ones((cell_size, cell_size, 1)))  # (800, 800, 3)
        mask = np.uint8(mask) * 255

        # Apply mask to img
        out = cv2.addWeighted(img, 0.5, mask, 0.5, 1.0)
        return out

    def get_cls_tile(
        self,
        batch: int,
        cls_true,  # (s, s, 6, 3)
        cls_pred,  # (s, s, 6, 3)
    ):
        grid_size = cls_true.shape[0]
        img = self.get_grid_img(batch, grid_size)

        overlay = np.zeros((grid_size * 8, grid_size * 8), np.float32)
        for y, (cls_true_row, cls_pred_row) in enumerate(zip(cls_true, cls_pred)):
            y *= 8
            for x, (cell_true, cell_pred) in enumerate(zip(cls_true_row, cls_pred_row)):
                x *= 8
                if np.all(cell_pred[0] > cell_pred[1:]) and np.all(cell_true[0] == 1):
                    continue
                # Nothing class
                overlay[y + 1, x + 1 : x + 4] = 0.5 + cell_true[0] / 2
                overlay[y + 1, x + 5 : x + 8] = 0.5 + cell_pred[0] / 2
                # Other classes
                overlay[y + 3 : y + 8, x + 1 : x + 4] = 0.5 + cell_true[1:] / 2
                overlay[y + 3 : y + 8, x + 5 : x + 8] = 0.5 + cell_pred[1:] / 2
        overlay = np.uint8(overlay * 255)
        overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_PLASMA)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        # Light strips
        overlay[2::8, ..., -1] = 0
        overlay[:, 4::8, ..., -1] = 0
        # Dark strips
        overlay[::8, ..., -1] = 0
        overlay[:, ::8, ..., -1] = 0

        scl = img.shape[0] // overlay.shape[0]
        overlay = np.kron(overlay, np.ones((scl, scl, 1), np.uint8))

        # Combine images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img = cv2.addWeighted(img, 1.0, overlay, 0.75, 1.0)

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def get_pos_tile(self, pos_true, pos_pred):
        grid_size = pos_true.shape[0]
        img = self.get_grid_img(grid_size)
        return img

    def get_scaled_output(
        self,
        batch: int,
        y_true: np.ndarray,  # (s, s, 8, 3)
        y_pred: np.ndarray,
    ):
        tiles = []

        # Split into parts
        pos_true = y_true[:, :, :2, :]  # (s, s, 2, 3)
        pos_pred = y_true[:, :, :2, :]
        cls_true = y_true[:, :, 2:, :]  # (s, s, 6, 3)
        cls_pred = y_pred[:, :, 2:, :]
        xst_true = np.max(np.argmax(cls_true, axis=-2) != 0, axis=-1)  # (s, s) bool
        xst_pred = np.max(np.argmax(cls_pred, axis=-2) != 0, axis=-1)

        xst_tile = self.get_xst_tile(batch, xst_true, xst_pred)
        cls_tile = self.get_cls_tile(batch, cls_true, cls_pred)
        # pos_tile = self.get_pos_tile(batch, pos_true, pos_pred)

        out_tile = np.concatenate(
            [
                xst_tile,
                cls_tile,
                # pos_tile,
            ],
            axis=0,
        )

        return out_tile

    def plot_prediction(self, y_pred=None):
        y_pred = y_pred or self.model.predict(
            self.X, verbose=0
        )  # (bs, s, s, 8, 3) for s = 25, 50, 100

        # Get output tiles
        batch_imgs = []
        for b in range(self.X.shape[0]):
            out_tiles = [
                self.get_scaled_output(b, self.y[i][b], y_pred[i][b]) for i in range(3)
            ]
            batch_img = np.concatenate(out_tiles, axis=1)
            batch_imgs.append(batch_img)
        out_img = np.concatenate(batch_imgs, axis=0)
        cv2.imwrite(self.output_file, out_img)

    def on_train_batch_end(self, batch, logs={}):
        if self.update_fn(batch):
            self.plot_prediction()

    def on_epoch_end(self, epoch, logs=None):
        self.plot_prediction()
