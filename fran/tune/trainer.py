from ray.tune.integration.pytorch_lightning import TuneReportCallback
from fran.trainers.trainer import Trainer


class RayTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_cbs(self, neptune, profiler, tags, description):
        cbs, logger, profiler = super().init_cbs(neptune, profiler, tags, description)
        T = TuneReportCallback(metrics={"loss": "val_loss_dice"}, on="validation_end")
        cbs.append (T)
        return cbs, logger, profiler
