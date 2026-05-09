#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from fran.configs.parser import ConfigMaker
from fran.managers import Project
from fran.profilers.paths import profiler_folder
from fran.trainers.trainer import Trainer
from fran.utils.misc import parse_devices


def str2bool(v):
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def build_parser():
    p = argparse.ArgumentParser(
        description="Benchmark full-pass train dataloader iteration without training."
    )
    p.add_argument("-t", "--project-title", "--project", default="kits23", dest="project_title")
    p.add_argument("-p", "--plan", "--plan-num", type=int, default=3, dest="plan_num")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--devices", default="[0]")
    p.add_argument("--bs", "--batch-size", type=int, default=6, dest="batch_size")
    p.add_argument("--compiled", type=str2bool, default=False)
    p.add_argument("--cache-rate", type=float, default=0.0)
    p.add_argument("--train-indices", type=int, default=None)
    p.add_argument("--val-indices", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=24)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--pin-memory", type=str2bool, default=True)
    p.add_argument("--persistent-workers", type=str2bool, default=True)
    p.add_argument("--dual-ssd", type=str2bool, default=False)
    p.add_argument("--batch-tfms", type=str2bool, default=False)
    p.add_argument("--compare-batch-tfms", type=str2bool, default=False)
    p.add_argument("--out-dir", type=Path, default=None)
    return p


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _iterate_full_pass(dl, datamodule):
    _sync()
    t0 = time.perf_counter()
    batch_count = 0
    datamodule.trainer = SimpleNamespace(
        training=True,
        testing=False,
        validating=False,
        sanity_checking=False,
    )
    for batch in dl:
        batch = datamodule.on_after_batch_transfer(batch, 0)
        _sync()
        batch_count += 1
        del batch
    elapsed_s = time.perf_counter() - t0
    return batch_count, elapsed_s


def _build_trainer(project, conf, args, batch_tfms: bool):
    trainer = Trainer(
        project_title=project.project_title,
        configs=deepcopy(conf),
        run_name=None,
    )
    trainer.setup(
        compiled=bool(args.compiled),
        batch_size=int(args.batch_size),
        devices=parse_devices(args.devices),
        epochs=1,
        wandb=False,
        profiler=False,
        batchsize_finder=False,
        train_indices=args.train_indices,
        val_indices=args.val_indices,
        val_every_n_epochs=1000,
        cbs=[],
        logging_freq=1000,
        early_stopping=False,
        dual_ssd=bool(args.dual_ssd),
        batch_tfms=batch_tfms,
    )
    return trainer


def main():
    args = build_parser().parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or profiler_folder(args.project_title)
    out_dir.mkdir(parents=True, exist_ok=True)

    project = Project(args.project_title)
    cm = ConfigMaker(project)
    cm.setup(int(args.plan_num))
    conf = cm.configs
    conf["dataset_params"]["batch_size"] = int(args.batch_size)
    conf["dataset_params"]["cache_rate"] = float(args.cache_rate)
    conf["dataset_params"]["fold"] = int(args.fold)
    conf["dataset_params"]["num_workers"] = int(args.num_workers)
    conf["dataset_params"]["prefetch_factor"] = int(args.prefetch_factor)
    conf["dataset_params"]["pin_memory"] = bool(args.pin_memory)
    conf["dataset_params"]["persistent_workers"] = bool(args.persistent_workers)

    batch_tfms_modes = [bool(args.batch_tfms)]
    if bool(args.compare_batch_tfms):
        batch_tfms_modes = [False, True]

    pass_rows = []
    mode_summaries = []
    total_t0 = time.perf_counter()

    for batch_tfms_mode in batch_tfms_modes:
        trainer = _build_trainer(project, conf, args, batch_tfms=batch_tfms_mode)
        dl = trainer.D.train_dataloader()
        expected_batches = len(dl)
        mode_rows = []

        for pass_idx in range(2):
            batch_count, elapsed_s = _iterate_full_pass(dl, trainer.D)
            row = {
                "batch_tfms": batch_tfms_mode,
                "pass_idx": pass_idx,
                "batch_count": batch_count,
                "elapsed_s": round(elapsed_s, 6),
            }
            pass_rows.append(row)
            mode_rows.append(row)
            print(
                f"PASS\tbatch_tfms={batch_tfms_mode}\tidx={pass_idx}\tbatches={batch_count}\telapsed_s={elapsed_s:.6f}",
                flush=True,
            )

        mode_summaries.append(
            {
                "batch_tfms": batch_tfms_mode,
                "expected_batches_per_pass": int(expected_batches),
                "passes": mode_rows,
            }
        )
        del trainer
        gc.collect()

    total_elapsed_s = time.perf_counter() - total_t0

    csv_path = out_dir / f"dataloader_passes_{stamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["batch_tfms", "pass_idx", "batch_count", "elapsed_s"]
        )
        writer.writeheader()
        writer.writerows(pass_rows)

    summary = {
        "project_title": project.project_title,
        "plan_num": int(args.plan_num),
        "compare_batch_tfms": bool(args.compare_batch_tfms),
        "batch_tfms": bool(args.batch_tfms),
        "results": mode_summaries,
        "total_elapsed_s": round(total_elapsed_s, 6),
    }
    json_path = out_dir / f"dataloader_passes_{stamp}.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    gc.collect()
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
