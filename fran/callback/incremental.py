from lightning.pytorch.callbacks import Callback


class LRFloorStop(Callback):
    """
    Lightning-native stop criterion: stop when optimizer LR reaches a floor.
    """

    def __init__(self, min_lr: float):
        super().__init__()
        self.min_lr = float(min_lr)
        self.triggered = False
        self.observed_lr = None

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.optimizers:
            return
        lrs = []
        for opt in trainer.optimizers:
            for group in opt.param_groups:
                lrs.append(float(group.get("lr", 0.0)))
        if not lrs:
            return
        self.observed_lr = min(lrs)
        if self.observed_lr <= self.min_lr:
            self.triggered = True
            trainer.should_stop = True
            setattr(trainer, "_incremental_stop_reason", f"lr_floor_reached:{self.observed_lr}")
