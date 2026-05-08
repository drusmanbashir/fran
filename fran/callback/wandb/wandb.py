import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from fran.configs.mnemonics import Mnemonics
from fran.transforms.spatialtransforms import one_hot
from fran.utils.colour_palette import colour_palette
from utilz.stringz import info_from_filename
from lightning.pytorch.callbacks import Callback
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid
from utilz.helpers import is_hpc


def _candidate_wandb_projects(project) -> list[str]:
    names = []
    direct_name = project.project_title
    names.append(str(direct_name))
    mnemonic = project.global_properties["mnemonic"]
    mapped_name = Mnemonics()[mnemonic].wandb
    if mapped_name and str(mapped_name) != str(direct_name):
        names.append(str(mapped_name))
    return names


def _resolve_run_by_name(api: wandb.Api, project_name: str, run_name: str):
    for run in api.runs(project_name):
        if str(run.id) == str(run_name) or str(run.name) == str(run_name):
            return run
    raise FileNotFoundError(
        f"Run '{run_name}' not found in wandb project '{project_name}'"
    )


def _artifact_basename(artifact) -> str:
    raw = str(artifact.name)
    base = raw.rsplit("/", 1)[-1]
    return base.split(":", 1)[0]


def _artifact_version_number(artifact) -> int:
    version = str(artifact.version)
    if version.startswith("v") and version[1:].isdigit():
        return int(version[1:])
    return -1


def download_run_artifact_by_name(
    *,
    project,
    run_name: str,
    artifact_name: str,
    destination_folder: Optional[str | Path] = None,
    alias: str = "latest",
    api: Optional[wandb.Api] = None,
) -> Path:
    """
    Download artifact logged by a run (resolved by run id/name) into a local folder.
    If `destination_folder` is omitted, defaults to `project.log_folder`.
    """
    api = api or wandb.Api()
    for candidate in _candidate_wandb_projects(project):
        try:
            run = _resolve_run_by_name(api=api, project_name=candidate, run_name=run_name)
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError(f"Unable to resolve run '{run_name}'.")

    matching = [a for a in run.logged_artifacts() if _artifact_basename(a) == artifact_name]
    if not matching:
        raise FileNotFoundError(
            f"No logged artifact named '{artifact_name}' in run '{run_name}'"
        )

    if alias == "latest":
        artifact = max(matching, key=_artifact_version_number)
    else:
        alias_text = str(alias)
        artifact = next(
            (
                a
                for a in matching
                if str(getattr(a, "version", "")) == alias_text
                or str(a.name).endswith(f":{alias_text}")
            ),
            None,
        )
        if artifact is None:
            raise FileNotFoundError(
                f"No artifact '{artifact_name}' with alias/version '{alias_text}' in run '{run_name}'"
            )

    if destination_folder is None:
        dest_root = Path(project.log_folder)
    else:
        dest_root = Path(destination_folder)
    dest_root.mkdir(parents=True, exist_ok=True)
    downloaded_path = Path(artifact.download(root=str(dest_root)))
    return downloaded_path


def _draw_validation_separator_on_canvas(
    draw,
    canvas,
    tile_h: int,
    nrow: int,
    padding: int,
    val_start_idx: int | None,
) -> None:
    if val_start_idx is None:
        return
    row = val_start_idx // nrow
    y = max(0, padding + row * (tile_h + padding) - 1)
    draw.line((0, y, canvas.size[0], y), fill=(255, 255, 0), width=6)


def _annotate_wandb_grid_image(
    img: np.ndarray,
    case_ids: list[str],
    tile_w: int,
    tile_h: int,
    nrow: int,
    padding: int,
    val_start_idx: int | None = None,
) -> np.ndarray:
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
    _draw_validation_separator_on_canvas(
        draw=draw,
        canvas=canvas,
        tile_h=tile_h,
        nrow=nrow,
        padding=padding,
        val_start_idx=val_start_idx,
    )
    return np.asarray(canvas)


def render_wandb_grid_worker(job: dict) -> dict:
    Path(job["local_folder"]).mkdir(parents=True, exist_ok=True)
    grid_stack = torch.from_numpy(job["grid_stack"])
    grid_tiles = grid_stack.permute(1, 0, 2, 3, 4).contiguous()
    grid_tiles = grid_tiles.view(-1, 3, grid_stack.shape[-2], grid_stack.shape[-1])
    rendered_grid = make_grid(
        grid_tiles,
        nrow=job["imgs_per_batch"] * 3,
        scale_each=True,
        padding=job["padding"],
    )
    rendered_image = rendered_grid.permute(1, 2, 0).cpu().numpy().astype("uint8")
    rendered_image = _annotate_wandb_grid_image(
        rendered_image,
        job["case_ids"],
        int(grid_stack.shape[-1]),
        int(grid_stack.shape[-2]),
        job["imgs_per_batch"],
        job["padding"],
        job["val_start_idx"],
    )
    image_path = Path(job["image_path"])
    Image.fromarray(rendered_image).save(image_path)
    return {"image_path": str(image_path), "key": "images/grid"}


class WandbImageGridCallback(Callback):
    def __init__(
        self,
        classes,
        patch_size,
        grid_rows=6,
        imgs_per_batch=4,
        epoch_freq=5,
        local_folder="/tmp/fran_wandb_grids",
    ):
        self.patch_size = torch.Size(patch_size)
        self.stride = int(self.patch_size[0] / imgs_per_batch)
        self.classes = classes
        self.grid_rows = grid_rows
        self.imgs_per_batch = imgs_per_batch
        self.epoch_freq = epoch_freq
        self.local_folder = Path(local_folder)
        self.local_folder.mkdir(parents=True, exist_ok=True)
        self._skip_async_grid_render = is_hpc()
        self.reset_grid()
        self._reset_async_grid_render_state()

    def _reset_async_grid_render_state(self) -> None:
        self._grid_render_executor = None
        self._pending_grid_renders = []
        self._max_pending_grid_renders = 2
        self._warned_grid_render_backlog = False

    def reset_grid(self):
        self.grid_imgs = []
        self.grid_preds = []
        self.grid_labels = []
        self.grid_case_ids = []
        self.val_start_idx = None

    def on_train_start(self, trainer, pl_module):
        self._shutdown_grid_render_executor()
        trainer.store_preds = False
        len_dl = int(len(trainer.train_dataloader) / trainer.accumulate_grad_batches)
        self.freq = max(2, int(len_dl / self.grid_rows))
        self._reset_async_grid_render_state()

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_freq == 0:
            self.reset_grid()

    def on_validation_epoch_start(self, trainer, pl_module):
        self.validation_grid_created = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._drain_completed_grid_renders(trainer)
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_freq == 0:
            trainer.store_preds = trainer.global_step % self.freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._drain_completed_grid_renders(trainer)
        if trainer.store_preds:
            self.populate_grid(pl_module, batch)

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._drain_completed_grid_renders(trainer)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._drain_completed_grid_renders(trainer)
        if trainer.store_preds and not self.validation_grid_created:
            self.pad_to_row()
            self.val_start_idx = self.grid_item_count()
            self.populate_grid_val(pl_module, batch)
            self.validation_grid_created = True

    def on_train_epoch_end(self, trainer, pl_module):
        self._drain_completed_grid_renders(trainer)
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_freq != 0 or len(self.grid_imgs) == 0:
            return
        if self._skip_async_grid_render:
            self._render_and_log_grid_sync(trainer)
        else:
            self._submit_async_grid_render(trainer)

    def on_fit_end(self, trainer, pl_module):
        self._drain_completed_grid_renders(trainer, wait=True)
        self._shutdown_grid_render_executor()

    def _ensure_grid_render_executor(self) -> ProcessPoolExecutor:
        if self._grid_render_executor is None:
            self._grid_render_executor = ProcessPoolExecutor(max_workers=1)
        return self._grid_render_executor

    def _build_async_grid_job(self, trainer) -> dict:
        max_items = self.imgs_per_batch * 10
        grid_stack = self.stack_grids()[:, :max_items].contiguous().cpu().numpy()
        case_ids = self.flatten_case_ids(max_items)
        val_start_idx = self.val_start_idx
        if val_start_idx is not None and val_start_idx >= max_items:
            val_start_idx = None
        epoch = trainer.current_epoch + 1
        image_path = self.local_folder / f"grid_epoch_{epoch}_step_{trainer.global_step}.png"
        return {
            "case_ids": case_ids,
            "grid_stack": grid_stack,
            "image_path": str(image_path),
            "imgs_per_batch": self.imgs_per_batch,
            "local_folder": str(self.local_folder),
            "padding": 1,
            "val_start_idx": val_start_idx,
        }

    def _submit_async_grid_render(self, trainer) -> None:
        self._drain_completed_grid_renders(trainer)
        if len(self._pending_grid_renders) >= self._max_pending_grid_renders:
            if not self._warned_grid_render_backlog:
                print(
                    "WandbImageGridCallback async render backlog is full. Skipping grid render for this epoch."
                )
                self._warned_grid_render_backlog = True
            return

        future = self._ensure_grid_render_executor().submit(
            render_wandb_grid_worker, self._build_async_grid_job(trainer)
        )
        self._pending_grid_renders.append(future)

    def _drain_completed_grid_renders(self, trainer, wait: bool = False) -> None:
        if len(self._pending_grid_renders) == 0:
            return

        pending_futures = []
        for future in self._pending_grid_renders:
            if not wait and not future.done():
                pending_futures.append(future)
                continue

            try:
                result = future.result()
            except Exception as e:
                print(f"WandbImageGridCallback async render failed: {e}")
                continue

            image_path = Path(result["image_path"])
            try:
                run = trainer.logger.experiment
                with Image.open(image_path) as rendered_image:
                    run.log({result["key"]: wandb.Image(rendered_image.copy())})
            except Exception as e:
                print(e)
            finally:
                if image_path.exists():
                    image_path.unlink()

        self._pending_grid_renders = pending_futures
        if len(self._pending_grid_renders) < self._max_pending_grid_renders:
            self._warned_grid_render_backlog = False

    def _shutdown_grid_render_executor(self) -> None:
        if self._grid_render_executor is None:
            return
        self._grid_render_executor.shutdown(wait=True)
        self._grid_render_executor = None

    def _render_and_log_grid_sync(self, trainer) -> None:
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
        pred = batch["pred"]
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        elif pred.dim() == img.dim() + 1:
            pred = pred[:, 0, :]
        pred = pred.cpu()
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
        _draw_validation_separator_on_canvas(
            draw=draw,
            canvas=canvas,
            tile_h=tile_h,
            nrow=nrow,
            padding=padding,
            val_start_idx=val_start_idx,
        )

    def annotate_grid(self, img, case_ids, tile_w, tile_h, nrow, padding, val_start_idx=None):
        return _annotate_wandb_grid_image(
            img=img,
            case_ids=case_ids,
            tile_w=tile_w,
            tile_h=tile_h,
            nrow=nrow,
            padding=padding,
            val_start_idx=val_start_idx,
        )


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


# %%
if __name__ == "__main__":
    from fran.managers.project import Project

    project_title = "kits23"
    run_name = "KITS23-SIRIG"
    artifact_name = "case_recorder"
    destination_folder = None  # None -> project.log_folder
    alias = "latest"

    project = Project(project_title)
# %%
    downloaded = download_run_artifact_by_name(
        project=project,
        run_name=run_name,
        artifact_name=artifact_name,
        destination_folder=destination_folder,
        alias=alias,
    )
    print(f"Downloaded artifact to: {downloaded}")
# %%
