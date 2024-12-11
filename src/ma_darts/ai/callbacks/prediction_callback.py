import cv2
import numpy as np
from tensorflow.keras import Model  # type: ignore
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.callbacks import Callback  # type: ignore


class PredictionCallback(Callback):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        output_file: str = "outputs/training_outputs.png",
        output_video_file: str | None = None,
        visualize_layers: list[str] | None = None,
        colormap: int = cv2.COLORMAP_PLASMA,
    ):
        super().__init__()
        self.X = X  # (bs, <shape>)
        self.y = y  # (bs, <shape>)

        self.pred_model = None

        self.output_file = output_file
        self.output_video_file = output_video_file
        self.out_vid = None

        self.colormap = colormap

        self.input_tile = None
        self.output_tile = None

        self.visualize_layers = visualize_layers

    def init_output_video(self, img):
        self.out_vid = cv2.VideoWriter(
            filename=self.output_file.replace("png", "mp4"),
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=10,
            frameSize=(img.shape[1], img.shape[0]),
        )

    def tile_img(self, img: np.ndarray) -> np.ndarray:
        res = np.zeros(
            (img.shape[0] * 4, img.shape[1] // 4, img.shape[2]), dtype=np.uint8
        )
        for i in range(4):
            src_x0 = i * img.shape[1] // 4
            src_x1 = src_x0 + img.shape[1] // 4
            dst_y0 = i * img.shape[0]
            dst_y1 = dst_y0 + img.shape[0]
            res[dst_y0:dst_y1, :] = img[:, src_x0:src_x1]
        return res

    def shrink_img(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(
            img,
            (img.shape[1] // 2, img.shape[0] // 2),
            interpolation=cv2.INTER_NEAREST,
        )
        return img

    def init_pred_model(self):
        def get_output_layers():

            # Default outupts
            if self.visualize_layers is None:
                return self.model.outputs

            # User-specified layers
            output_layers = []
            for layer_name in self.visualize_layers:
                layer = next(
                    (layer for layer in self.model.layers if layer.name == layer_name),
                    None,
                )
                if not layer:
                    continue
                output_layers.append(layer.output)
            output_layers = [
                self.model.layers[i].output
                for i in range(len(self.model.layers))
                if self.model.layers[i].name in self.visualize_layers
            ]
            # output_layers = [
            #     layer.output
            #     for layer in self.model.layers
            #     if layer.name in self.visualize_layers
            # ]
            return output_layers

        # create model
        output_layers = get_output_layers()

        self.pred_model = Model(
            inputs=self.model.inputs,
            outputs=output_layers,
        )
        self.pred_model.summary()
        print(self.model.outputs)
        print(self.pred_model.outputs)

    def prepare_image(
        self,
        inputs: np.ndarray,
        separate_channels: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        # Set correct shape
        if len(inputs.shape) == 1:  # (bs,) -> (bs, 1)
            inputs = np.expand_dims(inputs, -1)

        if len(inputs.shape) == 2:  # (bs, f) -> (bs, f, 1)
            inputs = np.expand_dims(inputs, -1)

        if len(inputs.shape) == 3:  # (bs, y, x) -> (bs, y, x, 1)
            inputs = np.expand_dims(inputs, -1)

        if len(inputs.shape) > 4:
            raise NotImplementedError("4D inputs are not supported (yet).")

        # Concatenate to image
        res = np.hstack(inputs)  # (bs, y, x, c) -> (y, x * bs, c)
        if separate_channels:
            res = np.vstack(
                [res[:, :, c] for c in range(res.shape[-1])]
            )  # (y, x_, c) -> (y * c, x_)

        # Normalize
        if normalize:
            res -= res.min()
            res /= res.max()

        return res  # (y * c, x * bs) or (y, x * bs, c) 0..1

    def create_output_img(self, tiles: list[np.ndarray]) -> np.ndarray:

        def single_img(img: np.ndarray) -> np.ndarray:
            img = np.uint8(img * 255)
            if len(img.shape) != 3:
                img = cv2.applyColorMap(img, self.colormap)
            return img

        def multiple_imgs(imgs: list[np.ndarray]) -> np.ndarray:
            # combine images at y level
            max_x = max(img.shape[1] for img in imgs)
            res_imgs = []
            for i, img in enumerate(imgs):
                # Add divider bar
                if i != 0:
                    res_imgs.append(np.zeros((10, max_x, 3)))

                # Convert to BGR image
                img = single_img(img)

                # Add image directly
                if img.shape[1] == max_x:
                    res_imgs.append(img)
                    continue

                # Add padded image
                expand_factor = max_x // img.shape[1]
                img = np.kron(img, np.ones((1, expand_factor, 1)))

                res_imgs.append(img)

            res_imgs = np.vstack(res_imgs)
            return res_imgs

        res_tiles = []
        for tile in tiles:
            if type(tile) in [list, tuple]:
                res_tiles.append(multiple_imgs(tile))
            else:
                res_tiles.append(single_img(tile))

        # Combine res tiles
        res_img = multiple_imgs(res_tiles)

        # Shrink image
        while max(*res_img.shape) > 3000:
            res_img = cv2.pyrDown(res_img)
        return res_img

    def on_epoch_end(self, epoch, logs=None):
        # Initialize prediction model
        if not self.pred_model:
            self.init_pred_model()

        # Predict on data
        pred_outputs = self.pred_model(self.X, training=False)  # (bs, <shape>) * n
        print(pred_outputs)
        exit()

        if type(pred_outputs) not in [list, tuple]:
            pred_outputs = [pred_outputs]

        # ------------------------
        # Create output image
        img_tiles = []

        # Add inputs to image
        if self.input_tile is None:
            separate_input_channels = not (
                len(self.X.shape) == 4 and self.X.shape[-1] == 3
            )
            self.input_tile = self.prepare_image(
                self.X, separate_channels=separate_input_channels
            )
        img_tiles.append(self.input_tile)

        # Add predictions to image
        pred_tiles = []
        for pred in pred_outputs:
            pred_tiles.append(self.prepare_image(pred))
        img_tiles.append(pred_tiles)

        # Add outputs to image
        if self.y is not None:
            if type(self.y) in [list, tuple]:
                out_tiles = []
                for y in self.y:
                    out_tiles.append(self.prepare_image(y))
                img_tiles.append(out_tiles)
            else:
                img_tiles.append(self.prepare_image(self.y))

        res = self.create_output_img(img_tiles)
        cv2.imwrite(self.output_file, res)
        # # encoder
        # pred_encoder = np.transpose(pred_encoder, (1, 0))  # (n, bs) 0..1
        # pred_encoder = np.kron(
        #     pred_encoder, np.ones((4, self.X.shape[2]))
        # )  # (y, w) 0..1
        # pred_encoder = (pred_encoder * 255).astype(np.uint8)  # (y, w) 0..255
        # pred_encoder = cv2.applyColorMap(
        #     pred_encoder, cv2.COLORMAP_PLASMA
        # )  # (y, w, 3) 0..255

        # # decoder
        # pred_decoder = np.concatenate(pred_decoder, axis=1)  # (y, w, 1)
        # pred_decoder = pred_decoder[:, :, 0]  # (y, w)
        # pred_decoder = np.clip(pred_decoder, 0, 1)
        # pred_decoder = (pred_decoder * 255).astype(np.uint8)  # (y, w) 0..255
        # pred_decoder = cv2.cvtColor(
        #     pred_decoder, cv2.COLOR_GRAY2BGR
        # )  # (y, w, 3) 0..255

        # # classifier
        # is_true_pred = np.argmax(pred_classifier, axis=1)  # (bs,) int
        # is_true_pred = is_true_pred == self.ys_true  # (bs,) bool
        # correct_pred = np.zeros(
        #     (is_true_pred.shape[0], 3), dtype=np.uint8
        # )  # (bs, 3) 0..255
        # correct_pred[is_true_pred] = [0, 255, 0]
        # correct_pred[~is_true_pred] = [0, 0, 255]
        # correct_pred = np.expand_dims(correct_pred, 0)  # (1, bs, 3)
        # correct_pred = np.kron(
        #     correct_pred, np.ones((8, self.X.shape[2], 1))
        # )  # (y, w, 3) 0..255

        # # encoder diff img
        # diff_img = self.Xs_true.astype(np.float32) - pred_decoder.astype(
        #     np.float32
        # )  # (y, w, 3) -255, 255
        # diff_img = ((diff_img + 255) / 2).astype(np.uint8)  # (y, w, 3) 0..255
        # diff_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)  # (y, w)
        # diff_img = cv2.applyColorMap(diff_img, cv2.COLORMAP_VIRIDIS)  # (y, w, 3)

        # # combine all pred imgs
        # img = np.concatenate(
        #     [
        #         self.Xs_true,
        #         correct_pred,
        #         pred_decoder,
        #         diff_img,
        #         pred_encoder,
        #     ],
        #     axis=0,
        # )  # (h, w, 3)
        # # reshape
        # img = self.tile_img(img)

        # # write frame
        # cv2.imwrite(self.output_file, img)

        # if img.shape[0] * img.shape[1] > 75_000_000:
        #     return

        # # Video output
        # if self.output_video_file is None:
        #     return

        # if self.output_video_file is not None:
        #     if self.out_vid is None:
        #         self._init_output_video(img)
        #     self.out_vid.write(img)

    def on_train_end(self, logs=None):
        if self.out_vid is not None:
            self.out_vid.release()