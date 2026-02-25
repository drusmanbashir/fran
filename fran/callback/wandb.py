from __future__ import annotations

import random
from typing import Any

import lightning as pl
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid

from fran.transforms.spatialtransforms import one_hot
from fran.utils.colour_palette import colour_palette
class WandbImageGridCallback(Callback):
    def __init__(
        self,
        classes,
        patch_size,
        grid_rows=6,
        imgs_per_batch=4,
        publish_deep_preds=False,
        apply_activation=True,
        epoch_freq=5,
    ):
        if not isinstance(patch_size, torch.Size):
            patch_size = torch.Size(patch_size)
        self.stride = int(patch_size[0] / imgs_per_batch)
        self.classes = classes
        self.patch_size = patch_size
        self.grid_rows = grid_rows
        self.imgs_per_batch = imgs_per_batch
        self.publish_deep_preds = publish_deep_preds
        self.apply_activation = apply_activation
        self.epoch_freq = epoch_freq
        self.grid_imgs = []
        self.grid_preds = []
        self.grid_labels = []

    def on_train_start(self, trainer, pl_module):
        trainer.store_preds = False
        len_dl = int(len(trainer.train_dataloader) / trainer.accumulate_grad_batches)
        self.freq = max(2, int(len_dl / self.grid_rows))

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.epoch_freq == 0:
            self.grid_imgs = []
            self.grid_preds = []
            self.grid_labels = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self.validation_grid_created = False

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.current_epoch % self.epoch_freq == 0:
            trainer.store_preds = trainer.global_step % self.freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.store_preds:
            self.populate_grid(pl_module, batch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.store_preds and not self.validation_grid_created:
            self.populate_grid(pl_module, batch)
            self.validation_grid_created = True

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.epoch_freq != 0 or len(getattr(self, "grid_imgs", [])) == 0:
            return

        merged = []
        for grd in [self.grid_imgs, self.grid_preds, self.grid_labels]:
            merged.append(torch.cat(grd))
        grd = torch.stack(merged)
        grd2 = grd.permute(1, 0, 2, 3, 4).contiguous().view(-1, 3, grd.shape[-2], grd.shape[-1])
        grd3 = make_grid(grd2, nrow=self.imgs_per_batch * 3, scale_each=True)
        img = grd3.permute(1, 2, 0).cpu().numpy().astype("uint8")

        run = trainer.logger.experiment
        run.log({"images/grid": wandb.Image(img)})

    def populate_grid(self, pl_module, batch):
        def _randomize():
            n_slices = img.shape[-1]
            batch_size = img.shape[0]
            self.slices = [random.randrange(0, n_slices) for _ in range(self.imgs_per_batch)]
            self.batches = [random.randrange(0, batch_size) for _ in range(self.imgs_per_batch)]

        img = batch["image"].cpu()
        label = batch["lm"].cpu().squeeze(1)
        label = one_hot(label, self.classes, axis=1)

        pred = pl_module.pred
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        elif pred.dim() == img.dim() + 1:
            pred = pred[:, 0, :]

        pred = F.softmax(pred.to(torch.float32), dim=1)

        _randomize()
        img, label, pred = self.img_to_grd(img), self.img_to_grd(label), self.img_to_grd(pred)
        img, label, pred = self.scale_tensor(img), self.assign_colour(label), self.assign_colour(pred)

        self.grid_imgs.append(img)
        self.grid_preds.append(pred)
        self.grid_labels.append(label)

    def img_to_grd(self, batch):
        return batch[self.batches, :, :, :, self.slices].clone()

    def assign_colour(self, tnsr):
        argmax_tensor = torch.argmax(tnsr, dim=1)
        bsz, h, w = argmax_tensor.shape
        rgb_tensor = torch.zeros((bsz, 3, h, w), dtype=torch.uint8)

        for key, color in colour_palette.items():
            mask = argmax_tensor == key
            for channel in range(3):
                rgb_tensor[:, channel, :, :][mask] = color[channel]
        return rgb_tensor

    def scale_tensor(self, tnsr):
        min_v, max_v = tnsr.min(), tnsr.max()
        rng = max_v - min_v
        tnsr = tnsr.repeat(1, 3, 1, 1)
        if rng == 0:
            return torch.zeros_like(tnsr, dtype=torch.uint8)
        out = (tnsr - min_v) / rng
        out = torch.clamp(out * 255, min=0, max=255)
        return out.to(torch.uint8)


class WandbLogBestCkpt(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        ckpt_best = trainer.checkpoint_callback.best_model_path
        ckpt_last = trainer.checkpoint_callback.last_model_path
        run = trainer.logger.experiment
        run.summary.update(
            {
                "training/last_model_path": ckpt_last,
                "training/best_model_path": ckpt_best,
            }
        )
