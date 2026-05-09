from pathlib import Path
from typing import Optional

from fran.callback.case_recorder import CaseIDRecorder
from fran.callback.wandb.wandb import WandbLogBestCkpt
from fran.trainers.trainer import Trainer


class CaseIDRecorderRT(CaseIDRecorder):
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch + 1
        if (epoch > 0 and epoch % self.freq == 0) or self.incrementing is True:
            self.dfs["epoch"] = epoch
            if self.incrementing is False:
                self._store(trainer, "train", self.loss_dicts_train, epoch)
            else:
                self._store(trainer, "train2", self.loss_dicts_train2, epoch)
            trainer.dfs = self.dfs
            self.reset()


class WandbLogBestCkptRT(WandbLogBestCkpt):
    def on_train_epoch_end(self, trainer, pl_module):
        ckpt_best = trainer.checkpoint_callback.best_model_path
        ckpt_last = trainer.checkpoint_callback.last_model_path
        run = trainer.logger.experiment
        run.summary.update(
            {
                "training/last_model_path": ckpt_last,
                "training/best_model_path": ckpt_best,
            }
        )


class TrainerRT(Trainer):
    """Compatibility shim for run-through training on top of the base Trainer."""

    def __init__(
        self,
        project_title,
        configs,
        run_name=None,
        ckpt: Optional[str | Path] = None,
    ):
        """Initialize the base Trainer in run-through mode."""
        super().__init__(
            project_title=project_title,
            configs=configs,
            run_name=run_name,
            ckpt=ckpt,
            run_through=True,
        )
