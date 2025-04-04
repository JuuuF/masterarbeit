import tensorflow as tf


class WarmupLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs, initial_lr):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr

    def on_epoch_begin(self, epoch, logs=None):
        # Only apply warmup during the initial warmup_epochs
        if epoch < self.warmup_epochs:
            # Linear warmup: gradually increase LR from a small value up to the initial_lr
            warmup_lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
            self.model.optimizer.learning_rate = warmup_lr
            tf.print(f"Epoch {epoch+1}: Warmup LR set to {warmup_lr:.6f}")
