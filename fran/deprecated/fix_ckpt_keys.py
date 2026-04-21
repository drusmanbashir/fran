# %%
import argparse
import ast
import shutil
from datetime import datetime
from pathlib import Path

import torch


CONFIG_PATHS = (
    ("hyper_parameters", "configs"),
    ("datamodule_hyper_parameters", "configs"),
)


def parse_value(value):
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def backup_ckpt(ckpt_fn):
    ckpt_fn = Path(ckpt_fn)
    backup_fn = ckpt_fn.with_name(
        f"{ckpt_fn.stem}.{datetime.now().strftime('%Y%m%d_%H%M%S')}{ckpt_fn.suffix}.bak"
    )
    shutil.copy2(ckpt_fn, backup_fn)
    return backup_fn


def get_config_roots(ckpt):
    roots = []
    for path in CONFIG_PATHS:
        node = ckpt
        for key in path:
            node = node.get(key, {})
        if node:
            roots.append((path, node))
    return roots


def resolve_value(config, src_parent, src_key, fixed_value):
    if fixed_value is not None:
        return parse_value(fixed_value)
    return config[src_parent][src_key]


def fix_ckpt_keys(
    ckpt_fn,
    src_key,
    dest_key=None,
    src_parent="dataset_params",
    dest_parent=("plan_train", "plan_valid", "plan_test"),
    value=None,
    dry_run=False,
):
    ckpt_fn = Path(ckpt_fn)
    ckpt = torch.load(ckpt_fn, map_location="cpu", weights_only=False)
    changes = []

    for root_path, config in get_config_roots(ckpt):
        val = resolve_value(config, src_parent, src_key, value)
        for parent in dest_parent:
            if parent in config:
                key = dest_key or src_key
                old = config[parent].get(key)
                config[parent][key] = val
                changes.append((".".join(root_path + (parent, key)), old, val))

    for path, old, new in changes:
        print(f"{path}: {old} -> {new}")

    if changes and not dry_run:
        backup_fn = backup_ckpt(ckpt_fn)
        torch.save(ckpt, ckpt_fn)
        print(f"backup: {backup_fn}")
        print(f"saved: {ckpt_fn}")

    return changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("--src-key", required=True)
    parser.add_argument("--dest-key")
    parser.add_argument("--src-parent", default="dataset_params")
    parser.add_argument(
        "--dest-parent",
        action="append",
        default=None,
        help="Repeat for multiple destination parent dicts.",
    )
    parser.add_argument("--value", help="Fixed value instead of ckpt source value.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fix_ckpt_keys(
        ckpt_fn=args.ckpt,
        src_key=args.src_key,
        dest_key=args.dest_key,
        src_parent=args.src_parent,
        dest_parent=tuple(args.dest_parent or ("plan_train", "plan_valid", "plan_test")),
        value=args.value,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

