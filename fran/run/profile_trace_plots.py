#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from fran.managers import Project


def _time_to_ms(token: str) -> float:
    s = token.strip()
    if s in {"", "--"}:
        return 0.0
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)(us|ms|s)$", s)
    if not m:
        return 0.0
    value = float(m.group(1))
    unit = m.group(2)
    if unit == "us":
        return value / 1000.0
    if unit == "ms":
        return value
    return value * 1000.0


def _parse_ops_table(path: Path) -> list[dict]:
    lines = path.read_text(errors="ignore").splitlines()
    header = None
    rows: list[dict] = []

    for i, line in enumerate(lines):
        if "Name" in line and "Self CPU %" in line and "Self CUDA" in line:
            header = [c.strip() for c in re.split(r"\s{2,}", line.strip()) if c.strip()]
            start = i + 1
            break
    else:
        return rows

    for line in lines[start:]:
        if not line.strip():
            continue
        if set(line.strip()) == {"-"}:
            continue
        parts = [c.strip() for c in re.split(r"\s{2,}", line.strip()) if c.strip()]
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts[: len(header)]))
        rows.append(row)
    return rows


def _top(rows: list[dict], time_key: str, k: int) -> list[tuple[str, float]]:
    items = []
    for r in rows:
        name = r.get("Name", "<unknown>")
        ms = _time_to_ms(r.get(time_key, "0us"))
        if ms > 0:
            items.append((name, ms))
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:k]


def _plot_barh(
    items: list[tuple[str, float]], title: str, xlabel: str, out_file: Path
) -> None:
    if not items:
        return
    names = [n for n, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    max_name = 58
    names = [n if len(n) <= max_name else n[: max_name - 3] + "..." for n in names]

    fig_h = max(5.0, min(14.0, 0.4 * len(items) + 2.2))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(range(len(vals)), vals, color="#1f77b4")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_file, dpi=170)
    plt.close(fig)


def _latest_existing(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate PNG plots from profiler ops tables in profile_traces."
    )
    p.add_argument("-t", "--project-title", default="lidc", dest="project_title")
    p.add_argument("--trace-dir", type=Path, default=None)
    p.add_argument(
        "--stamp", default="", help="Specific timestamp suffix, e.g. 20260304_041258"
    )
    p.add_argument("--top-k", type=int, default=20)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    if args.trace_dir is None:
        project = Project(args.project_title)
        trace_dir = Path(project.log_folder) / "profile_traces"
    else:
        trace_dir = Path(args.trace_dir)

    if not trace_dir.exists():
        raise FileNotFoundError(f"Trace directory not found: {trace_dir}")

    stamp = args.stamp.strip()
    if stamp:
        candidates = [
            trace_dir / f"ops_{stamp}.txt",
            trace_dir / f"ops_cpu_{stamp}.txt",
            trace_dir / f"ops_cuda_{stamp}.txt",
        ]
        ops_file = _latest_existing(candidates)
    else:
        candidates = sorted(trace_dir.glob("ops_*.txt")) + sorted(
            trace_dir.glob("ops_cpu_*.txt")
        )
        ops_file = _latest_existing(candidates)

    if ops_file is None:
        raise FileNotFoundError(f"No ops table found in {trace_dir}")

    rows = _parse_ops_table(ops_file)
    if not rows:
        raise RuntimeError(f"Could not parse ops table: {ops_file}")

    stem = ops_file.stem
    stamp_guess = stem.split("_", 1)[1] if "_" in stem else stem

    top_cpu = _top(rows, time_key="Self CPU", k=int(args.top_k))
    top_cuda = _top(rows, time_key="Self CUDA", k=int(args.top_k))

    cpu_png = trace_dir / f"plot_self_cpu_{stamp_guess}.png"
    cuda_png = trace_dir / f"plot_self_cuda_{stamp_guess}.png"

    _plot_barh(
        top_cpu,
        title=f"Top {len(top_cpu)} Ops by Self CPU ({stamp_guess})",
        xlabel="Self CPU time (ms)",
        out_file=cpu_png,
    )
    _plot_barh(
        top_cuda,
        title=f"Top {len(top_cuda)} Ops by Self CUDA ({stamp_guess})",
        xlabel="Self CUDA time (ms)",
        out_file=cuda_png,
    )

    print(f"Ops source: {ops_file}")
    if top_cpu:
        print(f"Wrote CPU plot: {cpu_png}")
    else:
        print("No CPU data found for plot.")
    if top_cuda:
        print(f"Wrote CUDA plot: {cuda_png}")
    else:
        print("No CUDA data found for plot.")


if __name__ == "__main__":
    main()
