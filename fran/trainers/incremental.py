from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import random
import re
from pathlib import Path
from typing import Optional

from lightning.pytorch.callbacks import Callback
from utilz.stringz import info_from_filename

from fran.managers.project import Project
from fran.trainers.trainer import Trainer


def _case_ids_from_batch(batch) -> list[str]:
    image = batch.get("image")
    if image is None or not hasattr(image, "meta"):
        return []
    meta = image.meta
    fns = meta.get("filename_or_obj", None) or meta.get("src_filename", None)
    if fns is None:
        return []
    if isinstance(fns, (str, Path)):
        fns = [fns]
    out = []
    for fn in fns:
        name = Path(str(fn)).name
        try:
            cid = info_from_filename(name, full_caseid=True)["case_id"]
        except Exception:
            cid = Path(name).stem
        out.append(cid)
    return out


def _per_case_dice_from_loss_dict(loss_dict: dict) -> dict[int, float]:
    grouped = {}
    for key, val in loss_dict.items():
        matched = re.match(r"loss_dice_batch(\d+)_label(\d+)", str(key))
        if not matched:
            continue
        batch_idx = int(matched.group(1))
        dice = max(0.0, min(1.0, 1.0 - float(val)))
        grouped.setdefault(batch_idx, []).append(dice)
    return {
        idx: float(sum(vals) / max(1, len(vals))) for idx, vals in grouped.items()
    }


def _to_logits_for_metrics(pred, target):
    if isinstance(pred, (list, tuple)):
        logits = pred[0]
    else:
        logits = pred
    if isinstance(logits, torch.Tensor) and logits.dim() == target.dim() + 2:
        logits = logits[:, 0]
    return logits


def _minmax_norm(vals: dict[str, float]) -> dict[str, float]:
    if not vals:
        return {}
    xs = list(vals.values())
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in vals.keys()}
    return {k: (v - lo) / (hi - lo) for k, v in vals.items()}


def _l2(a, b) -> float:
    diff = a - b
    return float((diff * diff).sum() ** 0.5)


class CaseDiceRecorder(Callback):
    def __init__(self, dataloader_idx: int = 0):
        super().__init__()
        self.dataloader_idx = int(dataloader_idx)
        self.latest_epoch_scores = {}

    def on_validation_epoch_start(self, trainer, pl_module):
        self._acc = {}

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if int(dataloader_idx) != self.dataloader_idx:
            return
        loss_dict = getattr(getattr(pl_module, "loss_fnc", None), "loss_dict", {})
        if not loss_dict:
            return
        case_ids = _case_ids_from_batch(batch)
        scores = _per_case_dice_from_loss_dict(loss_dict)
        if not case_ids or not scores:
            return
        for idx, case_id in enumerate(case_ids):
            if idx not in scores:
                continue
            self._acc.setdefault(case_id, []).append(scores[idx])

    def on_validation_epoch_end(self, trainer, pl_module):
        self.latest_epoch_scores = {
            cid: float(sum(vals) / max(1, len(vals))) for cid, vals in self._acc.items()
        }


class IncrementalTrainer:
    def __init__(self, project_title: str, configs: dict, random_seed: int = 42):
        self.project = Project(project_title=project_title)
        self.base_configs = deepcopy(configs)
        self.random_seed = int(random_seed)

    def _score_candidate_pool(
        self, tm: Trainer, candidate_case_ids: list[str], ckpt_path: Optional[str | Path] = None
    ) -> dict[str, dict]:
        if len(candidate_case_ids) == 0:
            return {}

        cfg_eval = deepcopy(tm.configs)
        cfg_eval["dataset_params"]["valid_case_ids"] = list(candidate_case_ids)
        cfg_eval["dataset_params"]["case_override_source_valid"] = "train"
        cfg_eval["dataset_params"]["cache_rate"] = 0.0

        dm_cls = tm.resolve_datamanager(cfg_eval["plan_valid"]["mode"])
        eval_manager = dm_cls(
            project=tm.project,
            configs=cfg_eval,
            batch_size=cfg_eval["dataset_params"]["batch_size"],
            cache_rate=0.0,
            split="valid",
            device=cfg_eval["dataset_params"].get("device", "cuda"),
            ds_type=cfg_eval["dataset_params"].get("ds_type"),
            keys=getattr(tm.D, "keys_val", "L,Remap,Ld,E,N,Rva, ResizePC"),
            data_folder=getattr(tm.D.valid_manager, "data_folder", None),
        )
        eval_manager.prepare_data()
        eval_manager.setup(stage="fit")

        preds = tm.trainer.predict(
            model=tm.N,
            dataloaders=eval_manager.dl,
            ckpt_path=str(ckpt_path) if ckpt_path is not None else None,
            return_predictions=True,
        )
        out = {}
        for batch_rows in preds:
            for row in batch_rows:
                cid = row["case_id"]
                if cid not in out:
                    out[cid] = row
                else:
                    # average scalar metrics if a case is seen multiple times
                    out[cid]["dice"] = 0.5 * (out[cid]["dice"] + row["dice"])
                    out[cid]["difficulty_dice"] = 0.5 * (
                        out[cid]["difficulty_dice"] + row["difficulty_dice"]
                    )
                    out[cid]["uncertainty"] = 0.5 * (
                        out[cid]["uncertainty"] + row["uncertainty"]
                    )
        return out

    def _select_with_combined_metric(
        self,
        metrics_by_case: dict[str, dict],
        max_add: int,
        dice_threshold: float,
        w_dice: float,
        w_uncertainty: float,
        w_diversity: float,
    ) -> list[str]:
        eligible = {
            cid: met
            for cid, met in metrics_by_case.items()
            if met["dice"] <= float(dice_threshold)
        }
        if not eligible:
            return []

        hard = {cid: met["difficulty_dice"] for cid, met in eligible.items()}
        unc = {cid: met["uncertainty"] for cid, met in eligible.items()}
        hard_n = _minmax_norm(hard)
        unc_n = _minmax_norm(unc)

        selected = []
        remaining = set(eligible.keys())
        while remaining and len(selected) < int(max_add):
            if len(selected) == 0:
                div = {cid: 1.0 for cid in remaining}
            else:
                div = {}
                for cid in remaining:
                    e = eligible[cid]["embedding"]
                    min_d = min(_l2(e, eligible[sid]["embedding"]) for sid in selected)
                    div[cid] = min_d
            div_n = _minmax_norm(div)

            scores = {}
            for cid in remaining:
                scores[cid] = (
                    float(w_dice) * hard_n.get(cid, 0.0)
                    + float(w_uncertainty) * unc_n.get(cid, 0.0)
                    + float(w_diversity) * div_n.get(cid, 0.0)
                )
            next_cid = max(scores.items(), key=lambda kv: kv[1])[0]
            selected.append(next_cid)
            remaining.remove(next_cid)
        return selected

    def run(
        self,
        initial_samples_n: int,
        add_samples_y: int,
        epochs_per_stage: int = 150,
        max_stages: int = 10,
        selection_threshold: float = 0.7,
        monitor: str = "dice",
        neptune: bool = True,
        devices=1,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        min_lr_to_continue: Optional[float] = None,
        w_dice: float = 0.5,
        w_uncertainty: float = 0.3,
        w_diversity: float = 0.2,
        run_id: Optional[str] = None,
        periodic_test: int = 0,
        early_stopping_patience: int = 20,
    ):
        assert monitor == "dice", "Only dice monitor is implemented currently"
        fold = self.base_configs["dataset_params"]["fold"]
        dss = self.base_configs["plan_train"]["datasources"]
        train_pool, _ = self.project.get_train_val_case_ids(fold=fold, ds=dss)
        if len(train_pool) == 0:
            raise RuntimeError("No training pool cases found for incremental training.")

        rng = random.Random(self.random_seed)
        train_pool = sorted(train_pool)
        initial_n = max(1, min(int(initial_samples_n), len(train_pool)))
        active_cases = set(rng.sample(train_pool, k=initial_n))

        if run_id is None:
            run_id = f"inc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.project.init_incremental_run_tracking(run_id, overwrite=True)

        resume_ckpt = None
        for stage_idx in range(int(max_stages)):
            cfg_stage = deepcopy(self.base_configs)
            if batch_size is not None:
                cfg_stage["dataset_params"]["batch_size"] = int(batch_size)
            cfg_stage["dataset_params"]["train_case_ids"] = sorted(active_cases)

            case_cb = CaseDiceRecorder(dataloader_idx=0)
            tm = Trainer(
                project_title=self.project.project_title,
                configs=cfg_stage,
                run_name=None,
                ckpt_path=resume_ckpt,
            )
            tm.setup(
                batch_size=batch_size,
                lr=lr,
                devices=devices,
                neptune=neptune,
                periodic_test=periodic_test,
                epochs=epochs_per_stage,
                cbs=[case_cb],
                early_stopping=True,
                early_stopping_monitor="val0_loss_dice",
                early_stopping_mode="min",
                early_stopping_patience=early_stopping_patience,
                lr_floor=min_lr_to_continue,
            )
            tm.fit()
            best_ckpt = tm.best_available_checkpoint()
            if best_ckpt is None:
                raise RuntimeError("No checkpoint produced in incremental stage.")

            stop_reason = getattr(tm.trainer, "_incremental_stop_reason", "")
            if stop_reason.startswith("lr_floor_reached"):
                self.project.append_incremental_stage(
                    run_id=run_id,
                    stage_idx=stage_idx,
                    active_cases=len(active_cases),
                    added_cases=0,
                    candidate_pool=len(
                        [cid for cid in train_pool if cid not in active_cases]
                    ),
                    monitor=monitor,
                    threshold=selection_threshold,
                    best_ckpt=str(best_ckpt),
                    stop_reason=stop_reason,
                )
                resume_ckpt = best_ckpt
                break

            remaining = [cid for cid in train_pool if cid not in active_cases]
            if len(remaining) == 0:
                self.project.append_incremental_stage(
                    run_id=run_id,
                    stage_idx=stage_idx,
                    active_cases=len(active_cases),
                    added_cases=0,
                    candidate_pool=0,
                    monitor=monitor,
                    threshold=selection_threshold,
                    best_ckpt=str(best_ckpt),
                    stop_reason="no_remaining_cases",
                )
                break

            candidate_metrics = self._score_candidate_pool(
                tm, remaining, ckpt_path=best_ckpt
            )
            selected = self._select_with_combined_metric(
                metrics_by_case=candidate_metrics,
                max_add=add_samples_y,
                dice_threshold=selection_threshold,
                w_dice=w_dice,
                w_uncertainty=w_uncertainty,
                w_diversity=w_diversity,
            )
            for cid in selected:
                active_cases.add(cid)

            stop_reason = ""
            if len(selected) == 0:
                stop_reason = "threshold_not_met"

            self.project.append_incremental_stage(
                run_id=run_id,
                stage_idx=stage_idx,
                active_cases=len(active_cases),
                added_cases=len(selected),
                candidate_pool=len(remaining),
                monitor=monitor,
                threshold=selection_threshold,
                best_ckpt=str(best_ckpt),
                stop_reason=stop_reason,
            )

            resume_ckpt = best_ckpt
            if stop_reason:
                break

        tracking_folder = self.project.curriculum_folder / run_id
        return {
            "run_id": run_id,
            "final_ckpt": str(resume_ckpt) if resume_ckpt else None,
            "active_cases": len(active_cases),
            "tracking_csv": str(tracking_folder / "stages.csv"),
            "tracking_state": str(tracking_folder / "state.json"),
        }
