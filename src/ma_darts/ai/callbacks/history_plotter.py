import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.callbacks import Callback  # type: ignore

import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt, ticker
from time import time


class HistoryPlotter(Callback):
    def __init__(
        self,
        filepath,
        update_on: str = "batches",
        update_frequency: int | float = 10,
        smooth_curves: bool = True,
        dark_mode: bool = True,
        log_scale: bool = False,
    ):
        super().__init__()

        # Internal trackkeeping
        self.train_logs = {}
        self.val_logs = {}
        self.filepath = filepath

        if not os.path.exists(dirname := os.path.dirname(filepath)):
            os.makedirs(dirname)

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
            f"Plotting every {self.update_frequency} {update_on} to file: {self.filepath}"
        )

        # Drawing functions
        self.smooth_curves = smooth_curves

        # Utils
        self.dividers = []
        self.epoch = 0
        self.log_scale = log_scale

        if dark_mode:
            plt.style.use("dark_background")

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
        def gaussian_window(size, sigma=None):
            if sigma is None:
                sigma = size / 3
            """Create a Gaussian window."""
            x = np.linspace(-size // 2, size // 2, size)
            weights = np.exp(-0.5 * (x / sigma) ** 2)
            return weights / weights.sum()

        if window_size <= 1:
            return y

        if window_size % 2 == 0:
            window_size += 1

        y_smooth = []
        n = len(y)
        for i in range(n):
            dist_l = i
            dist_r = n - i - 1
            w = min(dist_l, dist_r)
            w = min(w, window_size)
            window = y[i - w : i + w + 1]

            gaussian_weights = gaussian_window(2 * w + 1)

            # Handle log scale
            if self.log_scale:
                # Take the log of the values
                window_log = np.log10(
                    np.clip(window, 1e-10, None)
                )  # Clip to avoid log(0)
                smoothed_value = np.dot(gaussian_weights, window_log)
                y_smooth.append(10**smoothed_value)  # Exponentiate back
            else:
                y_smooth.append(np.dot(gaussian_weights, window))

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
            elif self.log_scale:
                axs[i].set_yscale("log")
                axs[i].yaxis.set_major_formatter(ticker.ScalarFormatter())

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

        fig.savefig(self.filepath, dpi=200)
        plt.close()

        # Save logs as pickle
        with open(self.filepath.replace(".png", ".pkl"), "wb") as f:
            pickle.dump({"train_logs": self.train_logs, "val_logs": self.val_logs}, f)

    def _update_on_batches(self, batch, *args, **kwargs):
        return batch % self.update_frequency == 0

    def _update_on_time(self, batch, *args, **kwargs):
        if batch == 0 or time() - self.last_update > self.update_frequency:
            self.last_update = time()
            return True

        return False

    def add_divider(self, label=None):
        self.dividers.append((self.epoch, label))

    # def on_train_batch_end(self, batch, logs={}):
    #     if self.update_fn(batch):
    #         self.plot_losses(extra_logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        if not self.train_logs:
            self.init_logs(logs)
        self.update_logs(logs)
        self.plot_losses()

        self.epoch += 1
