#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from fran.managers import Project
from fran.profilers.paths import profiler_folder


def _is_python_frame(frame: str) -> bool:
    f = frame.strip()
    return ".py" in f or f.endswith(".py")


def _clean_frame(frame: str) -> str:
    return frame.strip().replace("|", "/")


def _extract_file_hint(frame: str) -> str:
    m = re.search(r"([A-Za-z0-9_./\\-]+\.py)", frame)
    if m:
        return m.group(1)
    return frame


def _parse_stacks_file(path: Path):
    by_leaf = defaultdict(float)
    by_path = defaultdict(float)
    by_file = defaultdict(float)
    total_lines = 0
    parsed_lines = 0
    python_lines = 0

    for raw in path.read_text(errors="ignore").splitlines():
        total_lines += 1
        line = raw.strip()
        if not line:
            continue

        # Expected collapsed format: frame1;frame2;... <value>
        m = re.match(r"^(.*)\s+([+-]?\d+(?:\.\d+)?)$", line)
        if not m:
            continue
        stack_blob = m.group(1).strip()
        value = float(m.group(2))
        parsed_lines += 1

        frames = [_clean_frame(x) for x in stack_blob.split(";") if x.strip()]
        py_frames = [f for f in frames if _is_python_frame(f)]
        if not py_frames:
            continue
        python_lines += 1

        leaf = py_frames[-1]
        by_leaf[leaf] += value
        by_path[";".join(py_frames)] += value
        for f in py_frames:
            by_file[_extract_file_hint(f)] += value

    return {
        "by_leaf": by_leaf,
        "by_path": by_path,
        "by_file": by_file,
        "total_lines": total_lines,
        "parsed_lines": parsed_lines,
        "python_lines": python_lines,
    }


def _top_items(d: dict[str, float], top_k: int):
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


def _fmt(items, title: str, unit: str) -> str:
    out = [title, "-" * len(title)]
    for i, (name, val) in enumerate(items, start=1):
        out.append(f"{i:>2}. {val:>14.3f} {unit}  {name}")
    if len(items) == 0:
        out.append("(no entries)")
    return "\n".join(out)


def _find_stacks_file(trace_dir: Path, stamp: str) -> Path:
    if stamp:
        p = trace_dir / f"stacks_cpu_{stamp}.txt"
        if not p.exists():
            raise FileNotFoundError(f"Expected stacks file not found: {p}")
        return p

    files = sorted(trace_dir.glob("stacks_cpu_*.txt"))
    if not files:
        raise FileNotFoundError(f"No stacks_cpu_*.txt files found in: {trace_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Summarize top Python hotspots from profiler export_stacks CPU output."
    )
    p.add_argument("-t", "--project-title", default="lidc", dest="project_title")
    p.add_argument("--trace-dir", type=Path, default=None)
    p.add_argument("--stamp", default="", help="Timestamp suffix, e.g. 20260304_041258")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--unit-label", default="self_cpu_time_total")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    if args.trace_dir is None:
        project = Project(args.project_title)
        trace_dir = profiler_folder(project.project_title) / "profile_traces"
    else:
        trace_dir = Path(args.trace_dir)

    stacks_file = _find_stacks_file(trace_dir=trace_dir, stamp=args.stamp.strip())
    stamp = stacks_file.stem.replace("stacks_cpu_", "")

    parsed = _parse_stacks_file(stacks_file)

    top_leaf = _top_items(parsed["by_leaf"], int(args.top_k))
    top_file = _top_items(parsed["by_file"], int(args.top_k))
    top_path = _top_items(parsed["by_path"], int(args.top_k))

    report = []
    report.append(f"Source stacks file: {stacks_file}")
    report.append(f"Total lines: {parsed['total_lines']}")
    report.append(f"Parsed stack lines: {parsed['parsed_lines']}")
    report.append(f"Lines with Python frames: {parsed['python_lines']}")
    report.append("")
    report.append(_fmt(top_leaf, "Top Python Leaf Frames", args.unit_label))
    report.append("")
    report.append(_fmt(top_file, "Top Python Files (Aggregate)", args.unit_label))
    report.append("")
    report.append(_fmt(top_path, "Top Python Stack Paths", args.unit_label))
    text = "\n".join(report).rstrip() + "\n"

    out_file = trace_dir / f"python_hotspots_{stamp}.txt"
    out_file.write_text(text)
    print(text, end="")
    print(f"Wrote report: {out_file}")


if __name__ == "__main__":
    main()
