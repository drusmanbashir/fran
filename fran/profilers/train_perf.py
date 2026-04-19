#!/usr/bin/env python3
import argparse
import csv
import gc
import statistics
import subprocess
import time
from pathlib import Path

import torch
from fran.configs.parser import ConfigMaker
from fran.managers import Project
from fran.profilers.paths import profiler_folder
from fran.trainers.trainer_perf import Trainer
from fran.utils.misc import parse_devices


def str2bool(v):
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def split_ints(v):
    return [int(x) for x in str(v).split(",") if x.strip()]


def summarize_gpu(path, gpu):
    vals = []
    if not path.exists():
        return {}
    with path.open() as f:
        for row in csv.reader(f):
            if len(row) < 3 or int(row[0].strip()) != int(gpu):
                continue
            vals.append(int(row[1].strip()))
    if not vals:
        return {}
    low = sum(v <= 10 for v in vals)
    high = sum(v >= 80 for v in vals)
    return {
        "gpu_samples": len(vals),
        "gpu_util_mean": round(statistics.mean(vals), 1),
        "gpu_util_median": round(statistics.median(vals), 1),
        "gpu_idle_pct_le10": round(low / len(vals) * 100, 1),
        "gpu_busy_pct_ge80": round(high / len(vals) * 100, 1),
    }


def run_once(args, workers, prefetch, rep):
    project = Project(args.project_title)
    cm = ConfigMaker(project)
    cm.setup(int(args.plan_num))
    conf = cm.configs
    conf["dataset_params"]["batch_size"] = int(args.batch_size)
    conf["dataset_params"]["cache_rate"] = float(args.cache_rate)
    conf["dataset_params"]["fold"] = int(args.fold)
    conf["dataset_params"]["num_workers"] = int(workers)
    conf["dataset_params"]["prefetch_factor"] = int(prefetch)
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
    train_batches = len(trainer.D.train_dataloader())
    timed_batches = min(int(args.limit_train_batches), int(train_batches))
    trainer.trainer.limit_train_batches = timed_batches
    trainer.trainer.limit_val_batches = 0
    trainer.N.compiled = bool(args.compiled)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    gpu_file = args.out.parent / f"gpu_fit_w{workers}_p{prefetch}_r{rep}_{stamp}.csv"
    gpu_fh = gpu_file.open("w")
    sampler = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            "-lms",
            str(args.gpu_interval_ms),
        ],
        stdout=gpu_fh,
        stderr=subprocess.DEVNULL,
    )
    start = time.perf_counter()
    trainer.fit()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    sampler.terminate()
    try:
        sampler.wait(timeout=2)
    except subprocess.TimeoutExpired:
        sampler.kill()
        sampler.wait()
    gpu_fh.close()

    row = {
        "workers": int(workers),
        "prefetch": int(prefetch),
        "batch_affine": bool(args.batch_affine),
        "rep": int(rep),
        "batch_size": int(args.batch_size),
        "train_indices": int(args.train_indices),
        "dataloader_batches": int(train_batches),
        "timed_batches": int(timed_batches),
        "seconds": round(seconds, 3),
        "sec_per_batch": round(seconds / timed_batches, 4),
        "gpu_file": str(gpu_file),
    }
    row.update(summarize_gpu(gpu_file, args.gpu))
    print("RESULT\t" + "\t".join(f"{k}={v}" for k, v in row.items()), flush=True)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def build_parser():
    p = argparse.ArgumentParser(description="Short FRAN training timing sweep.")
    p.add_argument("-t", "--project-title", "--project", default="kits23", dest="project_title")
    p.add_argument("-p", "--plan-num", "--plan", type=int, default=2, dest="plan_num")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--devices", default="[0]")
    p.add_argument("--bs", "--batch-size", type=int, default=6, dest="batch_size")
    p.add_argument("--compiled", type=str2bool, default=False)
    p.add_argument("--cache-rate", type=float, default=0.0)
    p.add_argument("--train-indices", type=int, default=24)
    p.add_argument("--val-indices", type=int, default=1)
    p.add_argument("--limit-train-batches", type=int, default=20)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--num-workers", default="8,12,16")
    p.add_argument("--prefetch-factor", default="2")
    p.add_argument("--pin-memory", type=str2bool, default=True)
    p.add_argument("--persistent-workers", type=str2bool, default=True)
    p.add_argument("--batch-affine", type=str2bool, default=False)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--gpu-interval-ms", type=int, default=500)
    p.add_argument("--out", type=Path, default=None)
    return p


def main():
    args = build_parser().parse_args()
    if args.out is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.out = profiler_folder(args.project_title) / f"train_perf_{stamp}.csv"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for workers in split_ints(args.num_workers):
        for prefetch in split_ints(args.prefetch_factor):
            for rep in range(int(args.repeat)):
                rows.append(run_once(args, workers, prefetch, rep))

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
