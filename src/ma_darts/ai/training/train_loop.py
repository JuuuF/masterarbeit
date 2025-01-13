import numpy as np
from tensorflow import GradientTape
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import CallbackList


@staticmethod
def train_step(model, X, y) -> tuple[np.ndarray, dict]:
    # Forward Pass
    with GradientTape() as tape:
        y_pred = model(X, training=True)
        loss = model.compute_loss(X, y, y_pred)
        if model.optimizer is not None:
            loss = model.optimizer.scale_loss(loss)

    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_weights)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # Compute logs
    logs = model.compute_metrics(X, y, y_pred)
    logs["loss"] = loss.numpy()

    return y_pred, logs


@staticmethod
def update_epoch_logs(epoch_logs: dict, batch_logs: dict) -> None:
    for key, batch_log in batch_logs.items():
        if key not in epoch_logs:
            epoch_logs[key] = []
        epoch_logs[key].append(batch_log)


@staticmethod
def average_epoch_logs(epoch_logs: dict) -> None:
    for log_key, log_values in epoch_logs.items():
        epoch_logs[log_key] = np.mean(log_values)


@staticmethod
def add_val_logs(epoch_logs: dict, val_epoch_logs: dict) -> None:
    for k, val_log in val_epoch_logs.items():
        epoch_logs["val_" + k] = val_log


@staticmethod
def val_step(model, X, y) -> tuple[np.ndarray, dict]:
    y_pred = model(X, training=False)
    loss = model.compute_loss(X, y, y_pred)
    model._loss_tracker.update_state(loss)

    logs = model.compute_metrics(X, y, y_pred)

    return y_pred, logs


def train_loop(
    model,
    train_data,
    *,
    epochs=1,
    val_data=None,
    callbacks: list = [],
    verbose: int = 1,
):
    if not isinstance(callbacks, CallbackList):
        callbacks = CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            verbose=verbose,
            model=model,
            epochs=epochs,
            steps=None,
        )

    model.stop_training = False
    try:
        callbacks.on_train_begin()
        for epoch in range(epochs):
            callbacks.on_epoch_begin(epoch)

            # TRAINING
            epoch_logs = {}
            for train_batch, (X_batch, y_batch) in enumerate(train_data):

                callbacks.on_train_batch_begin(train_batch, logs=epoch_logs)

                y_pred, batch_logs = train_step(model, X_batch, y_batch)
                update_epoch_logs(epoch_logs, batch_logs)

                callbacks.on_train_batch_end(train_batch, logs=batch_logs)

            average_epoch_logs(epoch_logs)

            # VALIDATION
            val_epoch_logs = {}
            for val_batch, (X_batch, y_batch) in enumerate(val_data):
                y_pred, batch_logs = val_step(model, X_batch, y_batch)
                update_epoch_logs(val_epoch_logs, batch_logs)

            average_epoch_logs(val_epoch_logs)
            add_val_logs(epoch_logs, val_epoch_logs)

            callbacks.on_epoch_end(epoch, epoch_logs)

            clear_session()
            if model.stop_training:
                print("\nTraining stopped.")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    callbacks.on_train_end(epoch_logs)

    return model.history
