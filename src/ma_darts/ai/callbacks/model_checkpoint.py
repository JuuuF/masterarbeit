import os
import numpy as np
import warnings

import tensorflow as tf
from keras.src import backend

# from keras.src.utils import file_utils
from keras.src.utils import io_utils


class ModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Mostly yoinked from the official ModelCheckpoint.
    I just added the ability for a max_saves option to only keep the last n checkpoints.
    That seems like a reasonable thing to have when you don't have infinite storage.
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        verbose: int = 0,
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: str = "auto",
        save_freq: str = "epoch",
        initial_value_threshold: float | None = None,
        max_saves: int | None = None,
    ):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self.best = initial_value_threshold
        self.max_saves = max_saves
        self.last_saved = []
        self._current_epoch = 0

        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                f"ModelCheckpoint mode '{mode}' is unknown, " "fallback to auto mode.",
                stacklevel=2,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
            if self.best is None:
                self.best = np.inf
        elif mode == "max":
            self.monitor_op = np.greater
            if self.best is None:
                self.best = -np.inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                if self.best is None:
                    self.best = -np.inf
            else:
                self.monitor_op = np.less
                if self.best is None:
                    self.best = np.inf

        if self.save_freq != "epoch" and not isinstance(self.save_freq, int):
            raise ValueError(
                f"Unrecognized save_freq: {self.save_freq}. "
                "Expected save_freq are 'epoch' or integer values"
            )

        if save_weights_only:
            if not self.filepath.endswith(".weights.h5"):
                raise ValueError(
                    "When using `save_weights_only=True` in `ModelCheckpoint`"
                    ", the filepath provided must end in `.weights.h5` "
                    "(Keras weights format). Received: "
                    f"filepath={self.filepath}"
                )
        else:
            if not self.filepath.endswith(".keras"):
                raise ValueError(
                    "The filepath provided must end in `.keras` "
                    "(Keras model format). Received: "
                    f"filepath={self.filepath}"
                )

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self._save_model(epoch=self._current_epoch, batch=batch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if self.save_freq == "epoch":
            self._save_model(epoch=epoch, batch=None, logs=logs)

    def _should_save_on_batch(self, batch):
        """Handles batch-level saving logic, supports steps_per_execution."""
        if self.save_freq == "epoch":
            return False
        if batch <= self._last_batch_seen:  # New epoch.
            add_batches = batch + 1  # batches are zero-indexed.
        else:
            add_batches = batch - self._last_batch_seen
        self._batches_seen_since_last_saving += add_batches
        self._last_batch_seen = batch

        if self._batches_seen_since_last_saving >= self.save_freq:
            self._batches_seen_since_last_saving = 0
            return True
        return False

    def _save_model(self, epoch, batch, logs):
        """Saves the model.

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
                is set to `"epoch"`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        filepath = self._get_file_path(epoch, batch, logs)
        # Create host directory if it doesn't exist.
        dirname = os.path.dirname(filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        try:
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        f"Can save best model only with {self.monitor} "
                        "available, skipping.",
                        stacklevel=2,
                    )
                elif (
                    isinstance(current, np.ndarray) or backend.is_tensor(current)
                ) and len(current.shape) > 0:
                    warnings.warn(
                        "Can save best model only when `monitor` is "
                        f"a scalar value. Received: {current}. "
                        "Falling back to `save_best_only=False`."
                    )
                    self._save_with_track(filepath)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            io_utils.print_msg(
                                f"\nEpoch {epoch + 1}: {self.monitor} "
                                "improved "
                                f"from {self.best:.5f} to {current:.5f}, "
                                f"saving model to {filepath}"
                            )
                        self.best = current
                        self._save_with_track(filepath)
                    else:
                        pass
                        # if self.verbose > 0:
                        #     io_utils.print_msg(
                        #         f"\nEpoch {epoch + 1}: "
                        #         f"{self.monitor} did not improve "
                        #         f"from {self.best:.5f}"
                        #     )
            else:  # not save_best_only
                if self.verbose > 0:
                    io_utils.print_msg(
                        f"\nEpoch {epoch + 1}: saving model to {filepath}"
                    )
                self._save_with_track(filepath)
        except IsADirectoryError:  # h5py 3.x
            raise IOError(
                "Please specify a non-directory filepath for "
                "ModelCheckpoint. Filepath used is an existing "
                f"directory: {filepath}"
            )
        except IOError as e:  # h5py 2.x
            # `e.errno` appears to be `None` so checking the content of
            # `e.args[0]`.
            if "is a directory" in str(e.args[0]).lower():
                raise IOError(
                    "Please specify a non-directory filepath for "
                    "ModelCheckpoint. Filepath used is an existing "
                    f"directory: f{filepath}"
                )
            # Re-throw the error for any other causes.
            raise e

    def _save_with_track(self, filepath):

        # Keep track of last saves
        self.last_saved.append(filepath)

        # Only keep specified amount
        if self.max_saves is not None and len(self.last_saved) > self.max_saves:
            os.remove(self.last_saved.pop(0))
        if self.save_weights_only:
            self.model.save_weights(filepath, overwrite=True)
        else:
            self.model.save(filepath, overwrite=True)

    def _get_file_path(self, epoch, batch, logs):
        """Returns the file path for checkpoint."""

        try:
            # `filepath` may contain placeholders such as
            # `{epoch:02d}`,`{batch:02d}` and `{mape:.2f}`. A mismatch between
            # logged metrics and the path's placeholders can cause formatting to
            # fail.
            if batch is None or "batch" in logs:
                file_path = self.filepath.format(epoch=epoch + 1, **logs)
            else:
                file_path = self.filepath.format(
                    epoch=epoch + 1, batch=batch + 1, **logs
                )
        except KeyError as e:
            raise KeyError(
                f'Failed to format this callback filepath: "{self.filepath}". '
                f"Reason: {e}"
            )
        return file_path
