#!/usr/bin/env python3
import argparse
import csv
import gc
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from fran.configs.parser import ConfigMaker
from fran.managers import Project
from fran.profilers.paths import profiler_folder
from fran.trainers.trainer_perf import Trainer
from fran.utils.misc import parse_devices


def str2bool(v):
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


class TimedTransform:
    def __init__(self, transform, split, name, out_dir, stamp):
        self.transform = transform
        self.split = split
        self.name = name
        self.out_dir = str(out_dir)
        self.stamp = stamp
        self.count = 0
        self.fh = None

    def _open(self):
        pid = os.getpid()
        path = Path(self.out_dir) / f"transform_times_{self.stamp}_{self.split}_pid{pid}.csv"
        self.fh = path.open("a", buffering=1)
        if path.stat().st_size == 0:
            self.fh.write("pid,split,idx,transform,ms\n")

    def __call__(self, data):
        if self.fh is None:
            self._open()
        t0 = time.perf_counter_ns()
        out = self.transform(data)
        ms = (time.perf_counter_ns() - t0) / 1e6
        self.fh.write(f"{os.getpid()},{self.split},{self.count},{self.name},{ms:.6f}\n")
        self.count += 1
        return out


def wrap_manager(manager, out_dir, stamp):
    transforms = manager.transforms.transforms
    wrapped = []
    for i, transform in enumerate(transforms):
        name = f"{i:02d}_{transform.__class__.__name__}"
        wrapped.append(TimedTransform(transform, manager.split, name, out_dir, stamp))
    manager.transforms.transforms = wrapped


def summarize_transform_files(out_dir, stamp):
    rows = defaultdict(lambda: [0, 0.0])
    for path in sorted(Path(out_dir).glob(f"transform_times_{stamp}_*_pid*.csv")):
        with path.open() as f:
            for row in csv.DictReader(f):
                key = (row["split"], row["transform"])
                rows[key][0] += 1
                rows[key][1] += float(row["ms"])

    summary = Path(out_dir) / f"transform_summary_{stamp}.csv"
    with summary.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "transform", "count", "total_ms", "mean_ms"])
        for (split, transform), (count, total_ms) in sorted(
            rows.items(), key=lambda kv: kv[1][1], reverse=True
        ):
            writer.writerow([split, transform, count, round(total_ms, 3), round(total_ms / count, 3)])
    return summary


def build_parser():
    p = argparse.ArgumentParser(description="Profile FRAN dataloader transform timings.")
    p.add_argument("-t", "--project-title", "--project", default="kits23", dest="project_title")
    p.add_argument("-p", "--plan-num", "--plan", type=int, default=2, dest="plan_num")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--devices", default="[0]")
    p.add_argument("--bs", "--batch-size", type=int, default=6, dest="batch_size")
    p.add_argument("--compiled", type=str2bool, default=False)
    p.add_argument("--cache-rate", type=float, default=0.0)
    p.add_argument("--train-indices", type=int, default=192)
    p.add_argument("--val-indices", type=int, default=1)
    p.add_argument("--limit-batches", type=int, default=24)
    p.add_argument("--num-workers", type=int, default=24)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--pin-memory", type=str2bool, default=True)
    p.add_argument("--persistent-workers", type=str2bool, default=True)
    p.add_argument("--batch-affine", type=str2bool, default=False)
    p.add_argument("--out-dir", type=Path, default=None)
    return p


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
    conf["dataset_params"]["batch_affine"] = bool(args.batch_affine)

    trainer = Trainer(project_title=project.project_title, configs=conf, run_name=None)
    trainer.setup(
        compiled=bool(args.compiled),
        batch_size=int(args.batch_size),
        devices=parse_devices(args.devices),
        epochs=1,
        wandb=False,
        profiler=False,
        batchsize_finder=False,
        train_indices=int(args.train_indices),
        val_indices=int(args.val_indices),
        val_every_n_epochs=1000,
        cbs=[],
        logging_freq=1000,
    )

    wrap_manager(trainer.D.train_manager, out_dir, stamp)
    dl = trainer.D.train_dataloader()
    batch_file = out_dir / f"dataloader_next_{stamp}.csv"
    with batch_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_idx", "next_ms"])
        it = iter(dl)
        for batch_idx in range(int(args.limit_batches)):
            t0 = time.perf_counter_ns()
            batch = next(it)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            next_ms = (time.perf_counter_ns() - t0) / 1e6
            writer.writerow([batch_idx, round(next_ms, 3)])
            print(f"BATCH\tidx={batch_idx}\tnext_ms={next_ms:.3f}", flush=True)
            del batch

    summary = summarize_transform_files(out_dir, stamp)
    del trainer
    gc.collect()
    print(f"Wrote {batch_file}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
