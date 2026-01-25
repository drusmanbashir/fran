from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from fran.trainers.trainer import Trainer


class RayTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_cbs(self, cbs, neptune, profiler, tags, description):
        cbs, logger, profiler = super().init_cbs(cbs=cbs, neptune=neptune, profiler=profiler, tags=tags, description=description)
        T2 = TuneReportCheckpointCallback(metrics={"loss": "val1_loss_dice"}, on="validation_end")
        cbs.append (T2)
        return cbs, logger, profiler
