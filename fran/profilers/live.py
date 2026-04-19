#!/usr/bin/env python3
import argparse
import csv
import statistics
import subprocess
import time
from pathlib import Path

from fran.profilers.paths import profiler_folder


def build_parser():
    p = argparse.ArgumentParser(description="Sample GPU utilization with nvidia-smi.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("-t", "--project-title", "--project", default="kits23", dest="project_title")
    p.add_argument("--seconds", type=int, default=105)
    p.add_argument("--interval-ms", type=int, default=500)
    p.add_argument("--out", type=Path, default=None)
    return p


def summarize(path, gpu):
    vals = []
    with path.open() as f:
        for row in csv.reader(f):
            if len(row) < 4 or int(row[1].strip()) != int(gpu):
                continue
            vals.append(int(row[2].strip()))
    low = sum(v <= 10 for v in vals)
    high = sum(v >= 80 for v in vals)
    print(f"samples={len(vals)} gpu={gpu}")
    print(f"util_mean={statistics.mean(vals):.1f} util_median={statistics.median(vals):.1f}")
    print(f"pct_le_10={low / len(vals) * 100:.1f} pct_ge_80={high / len(vals) * 100:.1f}")


def main():
    args = build_parser().parse_args()
    if args.out is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.out = profiler_folder(args.project_title) / f"live_gpu_util_{args.gpu}_{stamp}.csv"
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,memory.used,memory.total",
        "--format=csv,noheader,nounits",
        "-lms",
        str(args.interval_ms),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        try:
            subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL, timeout=int(args.seconds))
        except subprocess.TimeoutExpired:
            pass
    summarize(args.out, args.gpu)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
