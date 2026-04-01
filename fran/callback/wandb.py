import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from fran.transforms.spatialtransforms import one_hot
from fran.utils.colour_palette import colour_palette
from fran.utils.string_works import info_from_filename
from lightning.pytorch.callbacks import Callback
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid


class WandbImageGridCallback(Callback):
    def __init__(self, classes, patch_size, grid_rows=6, imgs_per_batch=4, epoch_freq=5):
        self.patch_size = torch.Size(patch_size)
        self.stride = int(self.patch_size[0] / imgs_per_batch)
        self.classes = classes
        self.grid_rows = grid_rows
        self.imgs_per_batch = imgs_per_batch
        self.epoch_freq = epoch_freq
        self.reset_grid()

    def reset_grid(self):
        self.grid_imgs = []
        self.grid_preds = []
        self.grid_labels = []
        self.grid_case_ids = []
        self.val_start_idx = None

    def on_train_start(self, trainer, pl_module):
        trainer.store_preds = False
        len_dl = int(len(trainer.train_dataloader) / trainer.accumulate_grad_batches)
        self.freq = max(2, int(len_dl / self.grid_rows))

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_freq == 0:
            self.reset_grid()

    def on_validation_epoch_start(self, trainer, pl_module):
        self.validation_grid_created = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_freq == 0:
            trainer.store_preds = trainer.global_step % self.freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.store_preds:
            self.populate_grid(pl_module, batch)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.store_preds and not self.validation_grid_created:
            self.pad_to_row()
            self.val_start_idx = self.grid_item_count()
            self.populate_grid_val(pl_module, batch)
            self.validation_grid_created = True

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_freq != 0 or len(self.grid_imgs) == 0:
            return

        grid_stack = self.stack_grids()
        max_items = self.imgs_per_batch * 10
        grid_stack = grid_stack[:, :max_items]
        grid_tiles = self.to_grid_tiles(grid_stack)
        padding = 1
        rendered_grid = make_grid(
            grid_tiles, nrow=self.imgs_per_batch * 3, scale_each=True, padding=padding
        )
        rendered_image = rendered_grid.permute(1, 2, 0).cpu().numpy().astype("uint8")
        case_ids = self.flatten_case_ids(max_items)
        val_start_idx = self.val_start_idx
        if val_start_idx is not None and val_start_idx >= max_items:
            val_start_idx = None
        rendered_image = self.annotate_grid(
            rendered_image,
            case_ids,
            grid_stack.shape[-1],
            grid_stack.shape[-2],
            self.imgs_per_batch,
            padding,
            val_start_idx,
        )

        run = trainer.logger.experiment
        run.log({"images/grid": wandb.Image(rendered_image)})

    def populate_grid_val(self, pl_module, batch):
        img = batch["image"].cpu()
        label = batch["lm"].cpu().squeeze(1)
        label = one_hot(label, self.classes, axis=1)
        pred = batch["pred"]

        assert pred.dim() == img.dim(), "pred dim does not match img dim"
        pred = F.softmax(pred.to(torch.float32), dim=1)

        self.append_grid_batch(img, label, pred, batch)

    def populate_grid(self, pl_module, batch):
        img = batch["image"].cpu()
        label = batch["lm"].cpu().squeeze(1)
        label = one_hot(label, self.classes, axis=1)
        pred = batch["pred"].cpu()
        pred = F.softmax(pred.to(torch.float32), dim=1)

        self.append_grid_batch(img, label, pred, batch)

    def append_grid_batch(self, img, label, pred, batch):
        self._randomize(img)
        img = self.img_to_grd(img)
        label = self.img_to_grd(label)
        pred = self.img_to_grd(pred)
        img = self.scale_tensor(img)
        label = self.assign_colour(label)
        pred = self.assign_colour(pred)
        self.grid_imgs.append(img)
        self.grid_preds.append(pred)
        self.grid_labels.append(label)
        self.grid_case_ids.append(self.case_ids_for_selected_batches(batch))

    def case_ids_for_selected_batches(self, batch):
        case_ids = self.case_ids_from_batch(batch)
        case_id_row = []
        for batch_index in self.batches:
            if batch_index < len(case_ids):
                case_id_row.append(case_ids[batch_index])
            else:
                case_id_row.append("")
        return case_id_row

    def _randomize(self, img):
        n_slices = img.shape[-1]
        batch_size = img.shape[0]
        self.slices = []
        self.batches = []
        for _ in range(self.imgs_per_batch):
            self.slices.append(random.randrange(0, n_slices))
            self.batches.append(random.randrange(0, batch_size))

    def img_to_grd(self, tnsr):
        return tnsr[self.batches, :, :, :, self.slices].clone()

    def stack_grids(self):
        merged_grids = []
        merged_grids.append(torch.cat(self.grid_imgs))
        merged_grids.append(torch.cat(self.grid_preds))
        merged_grids.append(torch.cat(self.grid_labels))
        return torch.stack(merged_grids)

    def to_grid_tiles(self, grid_stack):
        grid_tiles = grid_stack.permute(1, 0, 2, 3, 4).contiguous()
        return grid_tiles.view(-1, 3, grid_stack.shape[-2], grid_stack.shape[-1])

    def flatten_case_ids(self, max_items):
        case_ids = []
        for case_row in self.grid_case_ids:
            for case_id in case_row:
                case_ids.append(case_id)
        return case_ids[:max_items]

    def grid_item_count(self):
        item_count = 0
        for case_row in self.grid_case_ids:
            item_count += len(case_row)
        return item_count

    def pad_to_row(self):
        if len(self.grid_case_ids) == 0:
            return
        remainder = self.grid_item_count() % self.imgs_per_batch
        if remainder == 0:
            return
        pad_n = self.imgs_per_batch - remainder
        pad_shape = (pad_n, 3, *self.grid_imgs[0].shape[-2:])
        self.grid_imgs.append(torch.zeros(pad_shape, dtype=torch.uint8))
        self.grid_preds.append(torch.zeros(pad_shape, dtype=torch.uint8))
        self.grid_labels.append(torch.zeros(pad_shape, dtype=torch.uint8))
        self.grid_case_ids.append([""] * pad_n)

    def assign_colour(self, tnsr):
        argmax_tensor = torch.argmax(tnsr, dim=1)
        batch_size, height, width = argmax_tensor.shape
        rgb_tensor = torch.zeros((batch_size, 3, height, width), dtype=torch.uint8)

        for key, color in colour_palette.items():
            mask = argmax_tensor == key
            for channel in range(3):
                rgb_tensor[:, channel, :, :][mask] = color[channel]
        return rgb_tensor

    def scale_tensor(self, tnsr):
        min_value = tnsr.min()
        max_value = tnsr.max()
        value_range = max_value - min_value
        repeated_tensor = tnsr.repeat(1, 3, 1, 1)
        if value_range == 0:
            return torch.zeros_like(repeated_tensor, dtype=torch.uint8)
        scaled_tensor = (repeated_tensor - min_value) / value_range
        scaled_tensor = torch.clamp(scaled_tensor * 255, min=0, max=255)
        return scaled_tensor.to(torch.uint8)

    def case_ids_from_batch(self, batch) -> list[str]:
        fns = batch["image"].meta["filename_or_obj"]
        if isinstance(fns, str):
            fns = [fns]
        out = []
        for fn in fns:
            name = Path(str(fn)).name
            out.append(info_from_filename(name, full_caseid=True)["case_id"])
        return out

    def draw_validation_separator(self, draw, canvas, tile_h, nrow, padding, val_start_idx):
        if val_start_idx is None:
            return
        row = val_start_idx // nrow
        y = max(0, padding + row * (tile_h + padding) - 1)
        draw.line((0, y, canvas.size[0], y), fill=(255, 255, 0), width=6)

    def annotate_grid(self, img, case_ids, tile_w, tile_h, nrow, padding, val_start_idx=None):
        canvas = Image.fromarray(img)
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for idx, case_id in enumerate(case_ids):
            row = idx // nrow
            col = idx % nrow
            x0 = padding + (col * 3) * (tile_w + padding)
            y0 = padding + row * (tile_h + padding)
            group_w = (tile_w * 3) + (padding * 2)
            x1 = x0 + group_w
            y1 = y0 + tile_h
            text = str(case_id)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            band_h = text_h + 4
            band_y0 = max(y0, y1 - band_h)
            text_x = max(x0 + 2, x0 + (group_w - text_w) // 2)
            text_y = band_y0 + max(1, (band_h - text_h) // 2 - 1)
            draw.rectangle(
                [(x0, band_y0), (x1, y1)],
                fill=(0, 0, 0),
            )
            draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
        self.draw_validation_separator(draw, canvas, tile_h, nrow, padding, val_start_idx)
        return np.asarray(canvas)


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
