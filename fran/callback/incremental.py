from __future__ import annotations

from lightning.pytorch import Callback
import torch


def _log_wandb_metrics(trainer, metrics: dict, step: int | None = None):
    if not getattr(trainer, "is_global_zero", True):
        return

    loggers = getattr(trainer, "loggers", None)
    if not loggers:
        logger = getattr(trainer, "logger", None)
        loggers = [logger] if logger is not None else []

    for logger in loggers:
        if logger is None or logger.__class__.__name__ != "WandbLogger":
            continue
        exp = getattr(logger, "experiment", None)
        if exp is None or not hasattr(exp, "log"):
            continue
        try:
            exp.log(metrics, step=step)
        except TypeError:
            exp.log(metrics)


class UpdateDatasetOnPlateau(Callback):

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
        patience: int = 3,
        grace: int = 0,
        n_samples_to_add: int = 1,
        datamodule_fn: str = "add_new_cases",
        verbose: bool = True,
        log_to_wandb: bool = False,
        wandb_prefix: str = "incremental/plateau",
    ):
        self.monitor = monitor
        self.mode = mode
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.grace = int(grace)
        self.n_samples_to_add = int(n_samples_to_add)
        self.datamodule_fn = datamodule_fn
        self.verbose = verbose
        self.log_to_wandb = bool(log_to_wandb)
        self.wandb_prefix = wandb_prefix

        self.best_score = None
        self.wait_count = 0

        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")


    def _is_improvement(self, current: float) -> bool:
        if self.best_score is None:
            return True

        if self.mode == "min":
            return current < self.best_score - self.min_delta
        else:
            return current > self.best_score + self.min_delta

    def on_validation_epoch_end(self, trainer, pl_module):

        if trainer.current_epoch < self.grace:
            return

        metrics = trainer.callback_metrics

        if self.monitor not in metrics:
            return

        current = metrics[self.monitor]
        if isinstance(current, torch.Tensor):
            current = current.detach().cpu().item()

        did_add_samples = False
        if self._is_improvement(current):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1

            if self.wait_count >= self.patience:
                self.wait_count = 0
                self.best_score = None  # reset plateau tracking
                did_add_samples = True

                dm = trainer.datamodule
                train_manager = dm.train_manager
                fn = getattr(train_manager, self.datamodule_fn)
                fn(self.n_samples_to_add)

                if self.verbose:
                    print(
                        f"UpdateDatasetOnPlateau: added "
                        f"{self.n_samples_to_add} samples at epoch "
                        f"{trainer.current_epoch}"
                    )

        if self.log_to_wandb:
            prefix = self.wandb_prefix
            _log_wandb_metrics(
                trainer,
                {
                    f"{prefix}/current": float(current),
                    f"{prefix}/best_score": float("nan") if self.best_score is None else float(self.best_score),
                    f"{prefix}/min_delta": float(self.min_delta),
                    f"{prefix}/bad": int(self.wait_count),
                    f"{prefix}/did_add_samples": int(did_add_samples),
                },
                step=int(trainer.global_step),
            )


class UpdateDatasetOnEMAMomentum(Callback):
    def __init__(
        self,
        monitor="train/loss",
        beta=0.99,
        min_mom=1e-4,
        patience_steps=500,
        grace_steps=1000,
        cooldown_steps=1000,
        n_samples_to_add=10,
        datamodule_fn="add_new_cases",
        verbose: bool = True,
        log_to_wandb: bool = False,
        wandb_prefix: str = "incremental/ema_momentum",
    ):
        self.monitor = monitor
        self.beta = float(beta)
        self.min_mom = float(min_mom)
        self.patience_steps = int(patience_steps)
        self.grace_steps = int(grace_steps)
        self.cooldown_steps = int(cooldown_steps)
        self.n_samples_to_add = int(n_samples_to_add)
        self.datamodule_fn = datamodule_fn
        self.verbose = verbose
        self.log_to_wandb = bool(log_to_wandb)
        self.wandb_prefix = wandb_prefix

        self.ema = None
        self.mom = 0.0
        self.bad = 0
        self.last_update = -10**18


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step < self.grace_steps or step - self.last_update < self.cooldown_steps:
            return

        x = trainer.callback_metrics.get(self.monitor)
        if x is None:
            return
        if isinstance(x, torch.Tensor):
            x = float(x.detach().cpu())

        if self.ema is None:
            self.ema = x
            if self.log_to_wandb:
                prefix = self.wandb_prefix
                _log_wandb_metrics(
                    trainer,
                    {
                        f"{prefix}/ema": float(self.ema),
                        f"{prefix}/mom": float(self.mom),
                        f"{prefix}/bad": int(self.bad),
                        f"{prefix}/min_mom": float(self.min_mom),
                        f"{prefix}/did_add_samples": 0,
                    },
                    step=int(step),
                )
            return

        prev = self.ema
        self.ema = self.beta * self.ema + (1 - self.beta) * x
        self.mom = self.beta * self.mom + (1 - self.beta) * (prev - self.ema)  # >0 improving

        self.bad = self.bad + 1 if self.mom < self.min_mom else 0
        did_add_samples = False
        if self.bad >= self.patience_steps:
            self.bad = 0
            self.last_update = step
            did_add_samples = True

            dm = trainer.datamodule
            train_manager = dm.train_manager
            fn = getattr(train_manager, self.datamodule_fn)
            fn(self.n_samples_to_add)
            if self.verbose:
                print(
                    f"UpdateDatasetOnPlateau: added "
                    f"{self.n_samples_to_add} samples at epoch "
                    f"{trainer.current_epoch}"
                )

        if self.log_to_wandb:
            prefix = self.wandb_prefix
            _log_wandb_metrics(
                trainer,
                {
                    f"{prefix}/ema": float(self.ema),
                    f"{prefix}/mom": float(self.mom),
                    f"{prefix}/bad": int(self.bad),
                    f"{prefix}/min_mom": float(self.min_mom),
                    f"{prefix}/did_add_samples": int(did_add_samples),
                },
                step=int(step),
            )


# Usage:
# trainer = Trainer(
#     ...,
#     reload_dataloaders_every_n_epochs=1,
#     callbacks=[UpdateDatasetOnPlateau(monitor="train/loss_epoch", mode="min", patience=3, min_delta=1e-3, grace=2)],
# )

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
