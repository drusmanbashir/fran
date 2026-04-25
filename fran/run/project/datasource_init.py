from __future__ import annotations

import argparse
from pathlib import Path

from fran.data.datasource import Datasource


def main(args: argparse.Namespace) -> None:
    folder = Path(args.folder).expanduser().resolve()
    ds = Datasource(folder=folder, name=args.mnemonic)
    ds.process(
        return_voxels=False,
        num_processes=args.num_processes,
        multiprocess=args.num_processes > 1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialise a Datasource and process new cases."
    )
    parser.add_argument("folder", help="Datasource folder containing images/ and lms/")
    parser.add_argument("mnemonic", help="Datasource name/mnemonic")
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        default=None,
        dest="num_processes",
        help="Number of worker processes for Datasource.process()",
    )
    args = parser.parse_known_args()[0]
    if args.num_processes is None:
        args.num_processes = 1 if args.n_processes is None else args.n_processes
    if args.num_processes < 1:
        parser.error("--num-processes must be >= 1")
    main(args)
