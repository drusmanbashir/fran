# %%
import ipdb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fastcore.basics import listify, store_attr
from lightning.pytorch.callbacks import Callback
from fran.utils.common import PAD_VALUE

tr = ipdb.set_trace
import ray

tr2 = ray.util.pdb.set_trace
# %%



class BatchSizeSafetyMargin(Callback):
    def __init__(self,  min_bs: int = 1, min_buffer= 2):
        self.has_run = False
        self.min_bs = min_bs
        self.min_buffer= min_buffer

    def _log_final_batch_size(self, trainer, final_bs: int):
        logger = getattr(trainer, "logger", None)
        experiment = getattr(logger, "experiment", None)
        config = getattr(experiment, "config", None)
        if config is not None and hasattr(config, "update"):
            config.update(
                {"configs/datamodule/dataset_params/batch_size": final_bs},
                allow_val_change=True,
            )

    def on_fit_start(self, trainer, pl_module):
        # BatchSizeFinder has already run at this point
        if self.has_run:
            return
        dm = trainer.datamodule
        bs = int(dm.batch_size)
        buffer =max(self.min_buffer, int(bs/8))
        safe_bs = max(self.min_bs, bs - buffer)

        if safe_bs != bs:
            dm.batch_size = safe_bs
            trainer.print(
                f"[BatchSizeSafetyMargin] Reducing batch_size {bs} → {safe_bs}"
            )
        # Keep config batch size synced to the runtime value chosen after finder/safety margin.
        final_bs = int(dm.batch_size)
        dm.configs["dataset_params"]["batch_size"] = final_bs
        self._log_final_batch_size(trainer, final_bs)
        self.has_run = True


class TargetLabelSanitizer(Callback):
    def __init__(self, configs: dict, target_key: str = "lm"):
        self.configs = configs
        self.target_key = target_key
        self.has_run = False

    def _iter_dataloaders(self, dm):
        yield dm.train_dataloader()
        val_dl = dm.val_dataloader()
        if isinstance(val_dl, (list, tuple)):
            yield from val_dl
        else:
            yield val_dl

    def _scan_labels(self, dm):
        labels = set()
        for dl in self._iter_dataloaders(dm):
            for batch in dl:
                target = batch[self.target_key]
                uniq = torch.unique(target)
                labels.update(int(v) for v in uniq.cpu().tolist())
        return sorted(labels)

    def _identity_sanitizer(self, target):
        return target.long()

    def _binary_sanitizer(self, target):
        return (target > 0).long()

    def on_fit_start(self, trainer, pl_module):
        if self.has_run:
            return

        labels_seen = self._scan_labels(trainer.datamodule)
        labels_all = [lab for lab in labels_seen if lab != PAD_VALUE]
        fg_classes = max(int(self.configs["model_params"]["out_channels"]) - 1, 1)

        if fg_classes == 1:
            sanitizer = self._binary_sanitizer
            labels_sanitized = [0, 1]
        else:
            sanitizer = self._identity_sanitizer
            labels_sanitized = labels_all
            allowed = set(range(fg_classes + 1))
            unexpected = sorted(lab for lab in labels_all if lab not in allowed)
            if unexpected:
                raise ValueError(
                    f"Unexpected labels {unexpected} for fg_classes={fg_classes}. "
                    f"Seen labels: {labels_all}"
                )

        if hasattr(pl_module, "loss_fnc") and hasattr(pl_module.loss_fnc, "set_target_label_sanitizer"):
            pl_module.loss_fnc.set_target_label_sanitizer(sanitizer)

        self.configs["plan_train"]["labels_all"] = labels_all
        if "plan_valid" in self.configs:
            self.configs["plan_valid"]["labels_all"] = labels_all
        self.configs["loss_params"]["labels_all_sanitized"] = labels_sanitized
        trainer.print(
            f"[TargetLabelSanitizer] labels={labels_all} fg_classes={fg_classes} "
            f"sanitized={labels_sanitized}"
        )
        self.has_run = True

class PredAsList(Callback):
    def after_pred(self):
        self.learn.pred = listify(self.learn.pred)

    def after_loss(self):
        self.learn.pred = self.learn.pred[0]


class DownsampleMaskForDS(Callback):
    order = 3  # after DropBBox

    def __init__(self, ds_scales):
        self.ds_scales = ds_scales

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        mask = self.learn.y
        output = []
        for sc in self.ds_scales:
            if all([i == 1 for i in sc]):
                output.append(mask)
            else:
                size = [round(ss * aa) for ss, aa in zip(sc, mask.shape[2:])]
                mask_downsampled = F.interpolate(mask, size=size, mode="nearest")
                output.append(mask_downsampled)
        self.learn.yb = [output]


class FixPredNan(Callback):
    "A `Callback` that terminates training if loss is NaN."
    order = -9

    def after_pred(self):
        self.learn.pred = torch.nan_to_num(self.learn.pred, nan=0.5)
        "Test if `last_loss` is NaN and interrupts training."


def make_grid_5d_input(a: torch.Tensor, batch_size_to_plot=16):
    """
    this function takes in a 5d tensor (BxCxDxWXH) e.g., shape 4,1,64,128,128)
    and creates a grid image for tensorboard
    """
    middle_point = int(a.shape[2] / 2)
    middle = slice(
        int(middle_point - batch_size_to_plot / 2),
        int(middle_point + batch_size_to_plot / 2),
    )
    slc = [0, slice(None), middle, slice(None), slice(None)]
    img_to_save = a[slc]
    # BxCxHxW
    img_to_save2 = img_to_save.permute(
        1, 0, 2, 3
    )  # re-arrange so that CxBxHxW (D is now minibatch)
    img_grid = torchvision.utils.make_grid(img_to_save2, nrow=int(batch_size_to_plot))
    return img_grid


def make_grid_5d_input_numpy_version(a: torch.Tensor, batch_size_to_plot=16):
    img_grid = make_grid_5d_input(a)
    img_grid_np = img_grid.cpu().detach().permute(1, 2, 0).numpy()
    plt.imshow(img_grid_np)
