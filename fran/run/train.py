#!/usr/bin/env python3

# training.py — minimal runner to Tm.fit()
import ipdb
import torch

from fran.callback.test import PeriodicTest
from fran.utils.misc import parse_devices

tr = ipdb.set_trace

import argparse
from typing import List, Union

from fran.configs.parser import ConfigMaker
from fran.managers import Project
from fran.trainers.trainer import Trainer


def print_device_info():
    if not torch.cuda.is_available():
        print("No CUDA devices found")
    else:
        n = torch.cuda.device_count()
        print(f"Found {n} CUDA device(s).")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            print(f"cuda:{i} — {props.name}, {props.total_memory/1024**3:.1f} GB")


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def main(args):

    import torch, os
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())

    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE — running on CPU")
        return

    else:
        print("CUDA AVAILABLE — GPUs:", torch.cuda.device_count())
        # --- Project & configs ----------------------------------------------------
        P = Project(args.project)
        devices = parse_devices(args.devices)
        C = ConfigMaker(P, configuration_filename=None)
        plan_num = int(args.plan_num)
        C.setup(plan_num)
        conf = C.configs

        # Update dataset params from CLI
        cbs=[]
        conf["dataset_params"]["cache_rate"] = args.cache_rate
        if args.ds_type is not None:
            conf["dataset_params"]["ds_type"] = args.ds_type
        if args.fold is not None:
            conf["dataset_params"]["fold"] = args.fold

        if args.test > 0:
            cbs+=[PeriodicTest(every_n_epochs=args.test, limit_batches=50)]


        # --- Trainer --------------------------------------------------------------
        print_device_info()

        Tm = Trainer(P.project_title, conf, args.run_name)

        Tm.setup(
            compiled=args.compiled,
            batch_size=args.batch_size,
            cbs=cbs,
            devices=devices,
            epochs=args.epochs if not args.profiler else 1,
            profiler=args.profiler,
            neptune=args.neptune,
            description=args.description,
            batch_finder=args.batch_finder,
            periodic_test=args.periodic_test,
        )

        Tm.N.compiled = args.compiled
        Tm.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train FRAN model up to Tm.fit(), no preprocessing."
    )
    parser.add_argument(
        "--project",
        default="nodes",
        help="Project title (e.g., nodes, totalseg, lidc2)",
    )
    parser.add_argument(
        "--plan-num",
        type=int,
        default=7,
        help="Active plan index for ConfigMaker.setup()",
    )
    parser.add_argument(
        "--devices",
        type=parse_devices,
        default=1,
        help='GPU devices: "0", "0,1", or count like "2"',
    )
    parser.add_argument(
        "--bs",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        default=None,
        help="If specified, will override conf['dataset_params']['fold']",
    )
    parser.add_argument("--epochs", type=int, default=600, help="Max epochs")
    parser.add_argument(
        "--compiled",
        type=str2bool,
        default=True,
        help="Compile model (Lightning/torch.compile)",
    )
    parser.add_argument(
        "--profiler", type=str2bool, default=False, help="Enable Lightning profiler"
    )
    parser.add_argument(
        "--neptune", type=str2bool, default=True, help="Enable Neptune logging"
    )
    parser.add_argument("--run-name", default=None, help='Run name (e.g., "LITS-1290")')
    parser.add_argument(
        "--description", default=None, help="Optional experiment description"
    )

    parser.add_argument(
        "--cache-rate",
        type=float,
        default=0.0,
        help="conf['dataset_params']['cache_rate']",
    )
    parser.add_argument(
        "--ds-type",
        default=None,
        choices=[None, "lmdb", "memmap", "zarr"],
        help="Dataset backend if supported",
    )
    parser.add_argument("-t", "--test", type=int, default=0, help="Test every n epochs. Default (0) means no test is done")
    parser.add_argument(
        "--batch-finder", type=str2bool, default=False, help="Enable batch size finder"
    )
    parser.add_argument(
        "--periodic-test", type=str2bool, default=True, help="Enable periodic testing"
    )

    args = parser.parse_known_args()[0]
# %%
    # args.fold = 1
    # args.project = "nodes"
    #
    # args.devices = '0'
    #
# %%
    main(args)
# %%
