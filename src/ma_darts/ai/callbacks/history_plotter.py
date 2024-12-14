import numpy as np
from tensorflow.keras.callbacks import Callback  # type: ignore

import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt


class HistoryPlotter(Callback):
    def __init__(
        self,
        filepath,
        update_on_batches: bool = True,
        batch_update_frequency: int = 10,
        ease_curves: bool = False,
        smooth_curves: bool = True,
    ):
        super().__init__()
        self.train_logs = {}
        self.val_logs = {}
        self.filepath = filepath

        self.update_on_batches = update_on_batches
        self.batch_update_frequency = batch_update_frequency

        self.ease_curves = ease_curves
        self.smooth_curves = smooth_curves

        self.dividers = []
        self.epoch = 0

    def init_logs(self, logs: dict):
        # Get log keys
        log_keys = [k for k in logs.keys() if not k.startswith("val_")]
        if "loss" not in log_keys:
            log_keys.insert(0, "loss")

        # Initialize logs
        self.train_logs = {k: [] for k in log_keys}
        self.val_logs = {"val_" + k: [] for k in log_keys}

    def update_logs(self, logs: dict):
        for log_key, log_val in logs.items():
            if log_key.startswith("val_"):
                self.val_logs[log_key].append(log_val)
            else:
                self.train_logs[log_key].append(log_val)

    def smooth(self, y, window_size=5):

        if window_size <= 1:
            return y
        if window_size % 2 == 0:
            window_size += 1

        window = np.ones(window_size) / window_size
        # pad y
        pad_width = (window_size - 1) // 2
        y_pad = np.pad(y, pad_width, mode="edge")

        y_smooth = np.convolve(y_pad, window, mode="same")
        # remove padding
        y_smooth = y_smooth[pad_width:-pad_width]

        # fade start and end
        if self.ease_curves:
            for i in range(window_size):
                fac = (window_size - i) / window_size
                y_smooth[i] = fac * y[i] + (1 - fac) * y_smooth[i]
                y_smooth[-i - 1] = fac * y[-i - 1] + (1 - fac) * y_smooth[-i - 1]

        return y_smooth

    def plot_losses(self, extra_logs=None):
        def draw_single_loss(ax, log, label, color):
            x = np.arange(len(log)) + 1
            if self.smooth_curves:
                log_smoothed = self.smooth(log, window_size=round(len(x) / 5))
                ax.plot(x, log_smoothed, label=label, color=color)
                ax.scatter(x, log, label="_nolegend_", color=color, alpha=0.4, s=6)
            else:
                ax.plot(x, log, label=label, color=color)

        # Prepare logs
        logs = {k: v.copy() for k, v in self.train_logs.items()}
        for log_key, extra_val in (extra_logs or {}).items():
            if log_key not in logs:
                logs[log_key] = []
            logs[log_key].append(extra_val)

        if len(logs) == 0:
            return

        # Create plot
        n_rows = len([l for l in logs.keys() if not l.startswith("val_")])
        height = 6 * n_rows
        width = 8 + len(logs["loss"]) * 0.05
        width = min(width, round(height * 2.5))  # clamp to reasonable width
        fig, axs = plt.subplots(
            nrows=n_rows,
            figsize=(width, height),
        )
        if n_rows == 1:
            axs = [axs]

        fig.suptitle("Training logs")

        # Plot metrics
        for i, (log_key, train_log) in enumerate(logs.items()):
            # For "accuracy" logs, draw min and max lines
            if "accuracy" in log_key:
                kwargs = dict(
                    color="gray", alpha=0.5, linestyle="dashed", label="_nolegend_"
                )
                axs[i].axhline(y=0, **kwargs)
                axs[i].axhline(y=1, **kwargs)

            # Draw train log
            draw_single_loss(axs[i], train_log, "train", "blue")

            # Draw validation log
            if len(self.val_logs) != 0:
                val_log = self.val_logs["val_" + log_key]
                draw_single_loss(axs[i], val_log, "val", "orange")

            # Add dividers
            if len(self.dividers) != 0:
                y_max = np.max(train_log)
                if "accuracy" in log_key:
                    y_max = max(y_max, 1)
                if len(self.val_logs) != 0:
                    y_max = max(y_max, np.max(val_log))
                for divider_x, divider_label in self.dividers:
                    axs[i].axvline(x=divider_x, color="gray", alpha=0.2)
                    if divider_label is not None:
                        axs[i].text(
                            divider_x,
                            y_max,
                            divider_label,
                            color="gray",
                            horizontalalignment="left",
                            verticalalignment="top",
                            rotation=90,
                            fontsize="small",
                        )

            # figure setup
            axs[i].set_title(log_key)
            axs[i].set_xlabel("epoch")
            axs[i].legend()

        fig.savefig(self.filepath)
        plt.close()

    def add_divider(self, label=None):
        self.dividers.append((self.epoch, label))

    def on_train_batch_end(self, batch, logs={}):
        if not self.update_on_batches:
            return
        if batch % self.batch_update_frequency != 0:
            return
        self.plot_losses(extra_logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        if not self.train_logs:
            self.init_logs(logs)
        self.update_logs(logs)
        self.plot_losses()

        self.epoch += 1
