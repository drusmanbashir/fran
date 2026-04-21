# %%
import argparse
from pathlib import Path

from fran.deprecated.fix_ckpt_keys import fix_ckpt_keys


def iter_ckpts(parent):
    yield from sorted(Path(parent).rglob("*.ckpt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", default="/s/fran_storage/checkpoints")
    parser.add_argument("--src-key", default="src_dims")
    parser.add_argument("--dest-key")
    parser.add_argument("--src-parent", default="dataset_params")
    parser.add_argument("--dest-parent", action="append", default=None)
    parser.add_argument("--value")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    dest_parent = tuple(args.dest_parent or ("plan_train", "plan_valid", "plan_test"))
    ckpts = list(iter_ckpts(args.parent))
    if args.limit is not None:
        ckpts = ckpts[: args.limit]

    fixed = 0
    failed = 0
    for ckpt in ckpts:
        print(f"\n== {ckpt}")
        try:
            changes = fix_ckpt_keys(
                ckpt_fn=ckpt,
                src_key=args.src_key,
                dest_key=args.dest_key,
                src_parent=args.src_parent,
                dest_parent=dest_parent,
                value=args.value,
                dry_run=args.dry_run,
            )
            fixed += bool(changes)
        except Exception as e:
            failed += 1
            print(f"FAILED: {type(e).__name__}: {e}")

    print(f"\nckpts={len(ckpts)} fixed={fixed} failed={failed}")


if __name__ == "__main__":
    main()

