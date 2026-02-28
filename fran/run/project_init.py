# %%
import argparse
import ipdb
tr = ipdb.set_trace

# very top of project_init.py (above any 3rd-party imports)
import os

from utilz.stringz import headline

from fran.data.dataregistry import DS
from fran.managers.project import Project


def main(args):
    headline("Arguments:")
    print(f"args.title: {args.title}")
    print(f"args.mnemonic: {args.mnemonic}")
    print(f"args.datasources: {args.datasources}")
    print(f"args.test: {args.test}")
    print(f"args.num_processes: {args.num_processes}")
    multiprocess = False if args.num_processes == 1 else True

    P = Project(project_title=args.title)
    if args.delete:
        P.delete()
        return

    if not P.db.exists():
        P.create(mnemonic=args.mnemonic)
    if args.datasources:
        datas = [DS[name] for name in args.datasources]
        P.add_data(datasources=datas, test=args.test, multiprocess=multiprocess)
    P.maybe_store_projectwide_properties(overwrite=False, multiprocess=multiprocess)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage FRAN projects")
    parser.add_argument("-t", "--title", help="Project title")
    parser.add_argument(
        "-m", "--mnemonic", help="Mnemonic, must be in MNEMONICS"
    )
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        help="number of parallel processes. If 1 (default), a single process is used",
        default=1,
    )
    parser.add_argument(
        "--datasources",
        nargs="*",
        default=[],
        help="Datasources to add, i.e., {}".format(DS.__repr__()),
    )
    parser.add_argument("--test", action="store_true", help="Mark datasources as test")
    parser.add_argument("--delete", action="store_true", help="Delete project")
# %%
    args = parser.parse_known_args()[0]
    # args.multiprocess=False
    # args.title = 'tmp2'
    # args.mnemonic = 'litsmall'
    # args.datasources = ['litsmall']

# %%
    main(args)
# %%
