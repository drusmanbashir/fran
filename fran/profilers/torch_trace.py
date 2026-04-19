#!/usr/bin/env python3
from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import torch
from fran.configs.parser import ConfigMaker
from fran.managers import Project
from fran.profilers.paths import profiler_folder
from fran.trainers.trainer_perf import Trainer
from fran.utils.misc import parse_devices
from torch.profiler import ProfilerActivity, profile


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def _iter_dm_managers(dm):
    for name in ("train_manager", "valid_manager", "test_manager"):
        m = getattr(dm, name, None)
        if m is not None:
            yield name, m


def _patch_project_split_for_mini_subset(
    project: Project, n_samples: int, skip_val: bool
):
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("--n-samples must be > 0")

    original = project.get_train_val_files

    def _limited_get_train_val_files(fold, datasources):
        train_cases, valid_cases = original(fold, datasources)
        train_cases = list(train_cases)[:n_samples]
        valid_cases = [] if skip_val else list(valid_cases)[:n_samples]
        return train_cases, valid_cases

    project.get_train_val_files = _limited_get_train_val_files
    return original


def _limit_dm_samples(dm, n_samples: int) -> None:
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("--n-samples must be > 0")

    for name, m in _iter_dm_managers(dm):
        old_cases = len(getattr(m, "cases", []) or [])
        old_data = len(getattr(m, "data", []) or [])

        m.cases = m.cases[:n_samples]
        m.data = m.data[:n_samples]

        m.create_dataset()
        m.create_train_dataloader() if m.split == "train" else m.create_valid_dataloader()

        new_cases = len(getattr(m, "cases", []) or [])
        new_data = len(getattr(m, "data", []) or [])
        print(
            f"[{name}] cases {old_cases} -> {new_cases}, data {old_data} -> {new_data}"
        )


def _configure_profiling_dataloaders(
    dm, enabled: bool, num_workers: int, persistent_workers: bool
) -> None:
    for name, m in _iter_dm_managers(dm):
        if name == "valid_manager" and enabled is False:
            continue

        m._num_workers = lambda: (int(num_workers), bool(persistent_workers))
        m.create_train_dataloader() if m.split == "train" else m.create_valid_dataloader()
        dl = getattr(m, "dl", None)
        print(
            f"[{name}] profiling dataloader: "
            f"num_workers={getattr(dl, 'num_workers', None)}, "
            f"persistent_workers={getattr(dl, 'persistent_workers', None)}"
        )


def _print_mem_summary() -> None:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_kb / 1024.0
    print(f"CPU max RSS: {rss_mb:.1f} MB")
    if torch.cuda.is_available():
        max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        max_resv = torch.cuda.max_memory_reserved() / (1024**3)
        print(f"GPU max allocated: {max_alloc:.3f} GB")
        print(f"GPU max reserved : {max_resv:.3f} GB")


def _maybe_experimental_config(enabled: bool):
    if not enabled:
        return None
    try:
        return torch._C._profiler._ExperimentalConfig(verbose=True)
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stack-aware FRAN profiler runner (writes separate stack artifacts)."
    )

    p.add_argument("-t", "--project-title", required=True, dest="project_title")
    p.add_argument("-p", "--plan", type=int, default=3)
    p.add_argument("--devices", type=parse_devices, default=[1])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--compiled", type=str2bool, default=False)
    p.add_argument("--val-every-n-epochs", type=int, default=1000)
    p.add_argument("--n-samples", type=int, default=2)
    p.add_argument("--skip-val", type=str2bool, default=True)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--persistent-workers", type=str2bool, default=False)
    p.add_argument("--cache-rate", type=float, default=0.0)

    p.add_argument("--limit-train-batches", type=int, default=10)
    p.add_argument("--cpu-profiling", type=str2bool, default=False)
    p.add_argument("--profile-record-shapes", type=str2bool, default=False)
    p.add_argument("--profile-with-stack", type=str2bool, default=False)
    p.add_argument("--stack-depth", type=int, default=2)
    p.add_argument("--profile-experimental-verbose", type=str2bool, default=False)
    p.add_argument("--export-stacks", type=str2bool, default=False)
    p.add_argument("--export-chrome-trace", type=str2bool, default=False)
    p.add_argument("--row-limit", type=int, default=80)

    return p


def _write_tables(
    prof, out_dir: Path, stamp: str, stack_depth: int, row_limit: int, stack_tables: bool
) -> None:
    ops_cpu_file = out_dir / f"ops_cpu_{stamp}.txt"
    ops_cuda_file = out_dir / f"ops_cuda_{stamp}.txt"

    table_cpu = prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=int(row_limit)
    )
    ops_cpu_file.write_text(table_cpu)

    table_cuda = prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=int(row_limit)
    )
    ops_cuda_file.write_text(table_cuda)

    print(f"Top ops (CPU): {ops_cpu_file}")
    print(f"Top ops (CUDA): {ops_cuda_file}")
    if stack_tables:
        stack_cpu_file = out_dir / f"ops_stack_cpu_{stamp}.txt"
        stack_cuda_file = out_dir / f"ops_stack_cuda_{stamp}.txt"
        table_stack_cpu = prof.key_averages(group_by_stack_n=int(stack_depth)).table(
            sort_by="self_cpu_time_total", row_limit=int(row_limit)
        )
        stack_cpu_file.write_text(table_stack_cpu)
        table_stack_cuda = prof.key_averages(group_by_stack_n=int(stack_depth)).table(
            sort_by="self_cuda_time_total", row_limit=int(row_limit)
        )
        stack_cuda_file.write_text(table_stack_cuda)
        print(f"Stack-grouped ops (CPU): {stack_cpu_file}")
        print(f"Stack-grouped ops (CUDA): {stack_cuda_file}")


def _export_stacks_if_available(prof, out_dir: Path, stamp: str, enabled: bool) -> None:
    if not enabled:
        return
    if not hasattr(prof, "export_stacks"):
        print("export_stacks not available in this torch build.")
        return

    stacks_cpu = out_dir / f"stacks_cpu_{stamp}.txt"
    stacks_cuda = out_dir / f"stacks_cuda_{stamp}.txt"

    try:
        prof.export_stacks(str(stacks_cpu), metric="self_cpu_time_total")
        print(f"Exported stacks (CPU): {stacks_cpu}")
    except Exception as e:
        print(f"Failed to export CPU stacks: {e}")

    try:
        prof.export_stacks(str(stacks_cuda), metric="self_cuda_time_total")
        print(f"Exported stacks (CUDA): {stacks_cuda}")
    except Exception as e:
        print(f"Failed to export CUDA stacks: {e}")


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this runner.")

    project = Project(args.project_title)
    original_split_fn = _patch_project_split_for_mini_subset(
        project, args.n_samples, bool(args.skip_val)
    )
    cm = ConfigMaker(project)
    cm.setup(int(args.plan))
    conf = cm.configs
    conf["dataset_params"]["cache_rate"] = float(args.cache_rate)

    tm = Trainer(project_title=project.project_title, configs=conf, run_name=None)
    tm.setup(
        compiled=bool(args.compiled),
        debug=True,
        batch_size=int(args.batch_size),
        devices=args.devices,
        epochs=int(args.epochs),
        lr=None,
        wandb=False,
        batchsize_finder=False,
        profiler=False,
        val_every_n_epochs=int(args.val_every_n_epochs),
        tags=["profile", "stacks"],
        description="Profiler run with stack exports",
    )
    if bool(args.skip_val):
        tm.trainer.limit_val_batches = 0
    tm.trainer.limit_train_batches = int(args.limit_train_batches)
    project.get_train_val_files = original_split_fn
    tm.N.compiled = bool(args.compiled)

    _limit_dm_samples(tm.D, n_samples=int(args.n_samples))
    _configure_profiling_dataloaders(
        tm.D,
        enabled=not bool(args.skip_val),
        num_workers=int(args.num_workers),
        persistent_workers=bool(args.persistent_workers),
    )

    torch.cuda.reset_peak_memory_stats()

    out_dir = profiler_folder(project.project_title) / "profile_traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    trace_file = out_dir / f"trace_{stamp}.json"

    activities = [ProfilerActivity.CUDA]
    if bool(args.cpu_profiling):
        activities.insert(0, ProfilerActivity.CPU)

    experimental_config = _maybe_experimental_config(
        bool(args.profile_experimental_verbose)
    )

    with profile(
        activities=activities,
        record_shapes=bool(args.profile_record_shapes),
        profile_memory=False,
        with_stack=bool(args.profile_with_stack),
        experimental_config=experimental_config,
    ) as prof:
        tm.ckpt = None
        tm.fit()

    if bool(args.export_chrome_trace):
        prof.export_chrome_trace(str(trace_file))
        print(f"Chrome trace: {trace_file}")
    else:
        print("Chrome trace: skipped. Use --export-chrome-trace true for a Perfetto timeline.")
    _write_tables(
        prof=prof,
        out_dir=out_dir,
        stamp=stamp,
        stack_depth=int(args.stack_depth),
        row_limit=int(args.row_limit),
        stack_tables=bool(args.profile_with_stack),
    )
    _export_stacks_if_available(
        prof=prof, out_dir=out_dir, stamp=stamp, enabled=bool(args.export_stacks)
    )
    _print_mem_summary()


if __name__ == "__main__":
    main(build_parser().parse_args())
