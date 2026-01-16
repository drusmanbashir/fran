from lightning.pytorch.callbacks import Callback


class PeriodicTest(Callback):
    def __init__(self, every_n_epochs: int, limit_batches:int = None):
        self.every_n_epochs = every_n_epochs
        self.limit_batches = limit_batches

    def on_validation_epoch_start(self, trainer, pl_module):
        self.skip = (trainer.current_epoch % self.every_n_epochs != 0)
            

    def on_validation_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx == 1 :
            if self.skip==True:
                raise StopIteration
            elif self.limit_batches is not None and batch_idx > self.limit_batches:
                    raise StopIteration


