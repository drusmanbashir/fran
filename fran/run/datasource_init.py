from __future__ import annotations

import argparse
from pathlib import Path

from fran.data.datasource import Datasource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialise a Datasource and process new cases."
    )
    parser.add_argument("folder", help="Datasource folder containing images/ and lms/")
    parser.add_argument("mnemonic", help="Datasource name/mnemonic")
    parser.add_argument(
        "n_processes",
        type=int,
        help="Number of worker processes for Datasource.process()",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    folder = Path(args.folder).expanduser().resolve()
    ds = Datasource(folder=folder, name=args.mnemonic)
    ds.process(
        return_voxels=False,
        num_processes=args.n_processes,
        multiprocess=args.n_processes > 1,
    )


if __name__ == "__main__":
    main(parse_args())
