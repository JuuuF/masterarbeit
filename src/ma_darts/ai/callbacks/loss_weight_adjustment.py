import tensorflow as tf


class LossWeightAdjustmentCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        loss_names=["decoder", "classifier"],
        start_epoch=50,
        total_epochs=450,
        initial_weights=[0.95, 0.05],
        final_weights=[0.35, 0.65],
        model_metrics: dict | list | None = None,
    ):
        super(LossWeightAdjustmentCallback, self).__init__()
        self.loss_names = loss_names
        self.start_epoch = start_epoch
        self.total_epochs = total_epochs
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.model_metrics = model_metrics

        # initial loss weights

    def get_loss_weights(self, epoch) -> dict[str, float]:
        epoch -= self.start_epoch
        if epoch < 0:
            # pre-start
            return {l: w for l, w in zip(self.loss_names, self.initial_weights)}

        if epoch < self.total_epochs:
            # loss weight interpolation
            fac = epoch / self.total_epochs
            loss_weights = {
                l: (f - i) * fac + i
                for l, f, i, in zip(
                    self.loss_names, self.final_weights, self.initial_weights
                )
            }
            if epoch % 10 == 0:
                print("\nCurrent loss weights:", loss_weights)
            return loss_weights

        if epoch == self.total_epochs:
            # final weights
            loss_weights = {l: w for l, w in zip(self.loss_names, self.final_weights)}
            print("\nFinal loss weights:", loss_weights)
            return loss_weights

        # epoch > total epochs: nothing to do, use last weights
        return {l: w for l, w in zip(self.loss_names, self.final_weights)}

    def on_epoch_begin(self, epoch, logs=None):
        opt = self.model.optimizer
        loss = self.model.loss
        loss_weights = self.get_loss_weights(epoch)
        # metrics = {
        #     "decoder": [],
        #     "classifier": [tf.keras.metrics.CategoricalAccuracy()],
        # }
        metrics = self.metrics

        # update loss weights
        self.model.compile(
            optimizer=opt,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
        )
