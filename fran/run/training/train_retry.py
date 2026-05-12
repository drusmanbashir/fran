#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

OOM_MARKERS = ("CUDA out of memory", "torch.OutOfMemoryError")
VAL_MARKERS = (
    "Validation ",
    "validation_step",
    "val_loop",
    "evaluation_loop.py",
    "swi_on_val_batch",
)


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def nullable_int(v):
    s = str(v).strip()
    if s == "" or s.lower() in {"none", "null"}:
        return None
    return int(s)


def run_stream(cmd):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    rc = proc.wait()
    return rc, "".join(lines)


def is_oom(output: str) -> bool:
    return any(marker in output for marker in OOM_MARKERS)


def is_val_oom(output: str) -> bool:
    return is_oom(output) and any(marker in output for marker in VAL_MARKERS)


def main():
    p = argparse.ArgumentParser(
        description="Run train.py and retry on CUDA OOM with lower batch size."
    )
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--min-bs", type=int, default=1)
    p.add_argument("--python", default=sys.executable)

    p.add_argument("-t", "--project-title", "--project", dest="project_title")
    p.add_argument("-p", "--plan", "--plan-num", type=int, default=7)
    p.add_argument("-d", "--devices", default="1")
    p.add_argument("-lr", "--learning-rate", dest="lr", type=float, default=None)
    p.add_argument("--bs", "--batch-size", dest="batch_size", type=int, default=4)
    p.add_argument("-f", "--fold", type=int, default=None)
    p.add_argument("-e", "--epochs", type=int, default=600)
    p.add_argument("--compiled", type=str2bool, default=False)
    p.add_argument("--profiler", type=str2bool, default=False)
    p.add_argument("--wandb", type=str2bool, default=True)
    p.add_argument("--val-device", default="cuda")
    p.add_argument("--batch-tfms", type=str2bool, default=False)
    p.add_argument("-r", "--run-name", dest="run_name", default=None)
    p.add_argument("--description", default=None)
    p.add_argument("--cache-rate", type=float, default=0.0)
    p.add_argument("--ds-type", default=None)
    p.add_argument("--all", type=str2bool, default=False)
    p.add_argument(
        "--val-every-n-epochs", dest="val_every_n_epochs", type=int, default=5
    )
    p.add_argument("--train-indices", type=nullable_int, default=None)
    p.add_argument(
        "--bsf",
        "--batchsize-finder",
        "--batch-finder",
        dest="batchsize_finder",
        type=str2bool,
        default=False,
    )
    p.add_argument("--dual-ssd", type=str2bool, default=False)
    args = p.parse_args()

    train_script = Path(__file__).with_name("train.py")
    bs = int(args.batch_size)
    val_device = str(args.val_device)
    used_val_cpu_retry = False
    bsf = bool(args.batchsize_finder)
    last_rc = 1

    for attempt in range(1, args.max_retries + 1):
        cmd = [args.python, "-u", str(train_script)]
        if args.project_title is not None:
            cmd += ["--project", str(args.project_title)]
        cmd += ["--plan-num", str(args.plan)]
        cmd += ["--devices", str(args.devices)]
        if args.lr is not None:
            cmd += ["--learning-rate", str(args.lr)]
        cmd += ["--bs", str(bs)]
        if args.fold is not None:
            cmd += ["--fold", str(args.fold)]
        cmd += [
            "--epochs",
            str(args.epochs),
            "--compiled",
            str(args.compiled).lower(),
            "--profiler",
            str(args.profiler).lower(),
            "--wandb",
            str(args.wandb).lower(),
            "--cache-rate",
            str(args.cache_rate),
            "--batch-tfms",
            str(args.batch_tfms).lower(),
            "--val-device",
            val_device,
            "--val-every-n-epochs",
            str(args.val_every_n_epochs),
            "--bsf",
            str(bsf).lower(),
            "--all",
            str(args.all).lower(),
            "--dual-ssd",
            str(args.dual_ssd).lower(),
        ]
        if args.run_name is not None:
            cmd += ["--run-name", str(args.run_name)]
        if args.description is not None:
            cmd += ["--description", str(args.description)]
        if args.ds_type is not None:
            cmd += ["--ds-type", str(args.ds_type)]
        if args.train_indices is not None:
            cmd += ["--train-indices", str(args.train_indices)]

        print(
            f"[train_retry] attempt={attempt}/{args.max_retries} "
            f"bs={bs} bsf={str(bsf).lower()} val_device={val_device} "
            f"batch_tfms={str(args.batch_tfms).lower()}"
        )
        last_rc, output = run_stream(cmd)
        if last_rc == 0:
            break

        if is_val_oom(output) and val_device != "cpu" and not used_val_cpu_retry:
            val_device = "cpu"
            used_val_cpu_retry = True
            print("[train_retry] validation OOM detected; retrying with val_device=cpu")
            continue

        if not is_oom(output):
            break

        next_bs = max(args.min_bs, bs - args.step)
        if next_bs == bs:
            break
        bs = next_bs

    return 0 if last_rc == 0 else last_rc


if __name__ == "__main__":
    raise SystemExit(main())
