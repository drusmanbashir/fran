from __future__ import annotations

from typing import Literal, Optional

import torch
from lightning.pytorch.callbacks import Callback


class MinimalLearningRateMonitor(Callback):
    """Slim callback mirroring Lightning's LR monitor behavior for Fabric trainer."""

    def __init__(
        self,
        logging_interval: Optional[Literal["step", "epoch"]] = None,
        log_momentum: bool = False,
        log_weight_decay: bool = False,
    ) -> None:
        if logging_interval not in (None, "step", "epoch"):
            raise ValueError("logging_interval should be 'step', 'epoch', or None")
        self.logging_interval = logging_interval
        self.log_momentum = bool(log_momentum)
        self.log_weight_decay = bool(log_weight_decay)

    @staticmethod
    def _group_name(opt_name: str, param_groups: list[dict], idx: int) -> str:
        if len(param_groups) > 1:
            pg_name = param_groups[idx].get("name", f"pg{idx + 1}")
            return f"{opt_name}/{pg_name}"
        return opt_name

    def _collect(self, trainer) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for opt_idx, opt in enumerate(getattr(trainer, "optimizers", [])):
            opt_name = f"lr-{opt.__class__.__name__}"
            if opt_idx > 0:
                opt_name = f"{opt_name}-{opt_idx}"
            param_groups = list(opt.param_groups)
            use_betas = "betas" in getattr(opt, "defaults", {})
            for pg_idx, pg in enumerate(param_groups):
                base = self._group_name(opt_name, param_groups, pg_idx)
                metrics[base] = float(pg.get("lr", 0.0))
                if self.log_momentum:
                    if use_betas:
                        mom = float(pg.get("betas", (0.0, 0.0))[0])
                    else:
                        mom = float(pg.get("momentum", 0.0))
                    metrics[f"{base}-momentum"] = mom
                if self.log_weight_decay:
                    metrics[f"{base}-weight_decay"] = float(pg.get("weight_decay", 0.0))
        return metrics

    def _emit(self, trainer) -> None:
        if not getattr(trainer, "loggers", None):
            return
        metrics = self._collect(trainer)
        if not metrics:
            return
        root_device = getattr(getattr(trainer, "strategy", None), "root_device", "cpu")
        trainer.callback_metrics.update({k: torch.tensor(v, device=root_device) for k, v in metrics.items()})
        step = getattr(getattr(getattr(trainer, "fit_loop", None), "epoch_loop", None), "_batches_that_stepped", None)
        if step is None:
            step = getattr(trainer, "global_step", 0)
        for logger in trainer.loggers:
            if hasattr(logger, "log_metrics"):
                logger.log_metrics(metrics, step=step)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.logging_interval == "epoch":
            return
        self._emit(trainer)

    def on_train_epoch_start(self, trainer, pl_module):
        if self.logging_interval == "step":
            return
        self._emit(trainer)

