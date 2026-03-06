from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from fran.trainers.trainer import Trainer


class RayTrainer(Trainer):
    def __init__(self, project_title, configs, run_name=None):
        super().__init__(project_title, configs, run_name)

    def init_cbs(self, cbs, neptune, batchsize_finder=True, test_every_n_epochs=True, profiler=False, tags=[], description=""):
        cbs, logger, profiler = super().init_cbs(cbs=cbs, neptune=neptune, batchsize_finder=batchsize_finder, test_every_n_epochs=test_every_n_epochs, profiler=profiler, tags=tags, description=description)
        T2 = TuneReportCheckpointCallback(metrics={"loss": "val1_loss_dice"}, on="validation_end")
        cbs.append (T2)
        return cbs, logger, profiler
