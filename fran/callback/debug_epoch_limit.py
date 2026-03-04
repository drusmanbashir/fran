from lightning.pytorch.callbacks import BatchSizeFinder, Callback, TQDMProgressBar


class DebugEpochBatchLimit(Callback):
    def __init__(self, n: int = 10):
        self.n = n

    def on_fit_start(self, trainer, pl_module):
        trainer.callbacks = [
            cb
            for cb in trainer.callbacks
            if not isinstance(cb, (BatchSizeFinder, TQDMProgressBar))
        ]

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx >= self.n:
            raise StopIteration

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx >= self.n:
            raise StopIteration
