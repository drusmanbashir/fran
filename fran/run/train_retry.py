#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


OOM_MARKERS = ("CUDA out of memory", "torch.OutOfMemoryError")


def main():
    parser = argparse.ArgumentParser(
        description="Retry fran/run/train.py in a fresh process when CUDA OOM occurs."
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--step", type=int, default=1, help="Batch-size decrement per OOM")
    parser.add_argument("--min-bs", type=int, default=1)
    parser.add_argument("--start-bs", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    args, train_args = parser.parse_known_args()

    train_script = Path(__file__).with_name("train.py")
    bs = args.start_bs

    clean_args = []
    i = 0
    while i < len(train_args):
        arg = train_args[i]
        if arg.startswith("--bs=") or arg.startswith("--batch-size="):
            if bs is None:
                bs = int(arg.split("=", 1)[1])
            i += 1
            continue
        if arg in {"--bs", "--batch-size"}:
            if bs is None and i + 1 < len(train_args):
                bs = int(train_args[i + 1])
            i += 2
            continue
        if arg.startswith("--bsf=") or arg.startswith("--batchsize-finder="):
            i += 1
            continue
        if arg in {"--bsf", "--batchsize-finder"}:
            i += 2
            continue
        clean_args.append(arg)
        i += 1
    if bs is None:
        bs = 4

    last_rc = 1
    for attempt in range(1, args.max_retries + 1):
        cmd = [args.python, str(train_script), *clean_args, "--bs", str(bs), "--bsf", "false"]
        print(f"[train_retry] attempt={attempt}/{args.max_retries} bs={bs} bsf=false")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        last_rc = proc.returncode
        if proc.returncode == 0:
            break
        output = f"{proc.stdout or ''}{proc.stderr or ''}"
        if not any(marker in output for marker in OOM_MARKERS):
            print("[train_retry] non-OOM failure; stopping retries.")
            break
        next_bs = max(args.min_bs, bs - args.step)
        if next_bs == bs:
            print("[train_retry] reached minimum batch size; stopping retries.")
            break
        bs = next_bs

    return 0 if last_rc == 0 else last_rc


if __name__ == "__main__":
    raise SystemExit(main())
