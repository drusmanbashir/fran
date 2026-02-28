#!/usr/bin/env python3

# training.py — minimal runner to Tm.fit()
import ipdb
import torch
from utilz.stringz import headline

from fran.callback.test import PeriodicTest
from fran.utils.misc import parse_devices

tr = ipdb.set_trace

import argparse
from typing import List, Union

from fran.configs.parser import ConfigMaker
from fran.managers import Project
from fran.trainers.incremental import IncrementalTrainer
from fran.trainers.trainer_bk import TrainerBK


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


class UniqueArgValue(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest, None)
        if current is not None and current != values:
            parser.error(
                f"Conflicting values for {self.dest}: {current!r} vs {values!r}"
            )
        setattr(namespace, self.dest, values)


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
        P = Project(args.project_title)
        devices = parse_devices(args.devices)
        C = ConfigMaker(P)
        plan = int(args.plan)
        C.setup(plan)
        conf = C.configs

        # Update dataset params from CLI
        cbs=[]
        conf["dataset_params"]["cache_rate"] = args.cache_rate
        if args.ds_type is not None:
            conf["dataset_params"]["ds_type"] = args.ds_type
        if args.fold is not None:
            conf["dataset_params"]["fold"] = args.fold


        # --- Trainer --------------------------------------------------------------
        print_device_info()
        if args.incremental:
            inc = IncrementalTrainer(
                project_title=P.project_title,
                configs=conf,
                run_name=args.run_name,
            )
            inc.setup(
                compiled=args.compiled,
                batch_size=args.batch_size,
                cbs=cbs,
                devices=devices,
                epochs=args.epochs if not args.profiler else 1,
                lr=args.lr,
                profiler=args.profiler,
                wandb=args.wandb,
                description=args.description,
                batchsize_finder=args.batchsize_finder,
                periodic_test=args.periodic_test,
                start_n=args.initial_samples,
                n_samples_to_add=args.add_samples,
            )
            inc.N.compiled = args.compiled
            inc.fit()
            headline("Incremental training completed")
            return

        Tm = TrainerBK(project_title=P.project_title, configs=conf, run_name=args.run_name)
        Tm.setup(
            compiled=args.compiled,
            batch_size=args.batch_size,
            cbs=cbs,
            devices=devices,
            epochs=args.epochs if not args.profiler else 1,
            lr=args.lr,
            profiler=args.profiler,
            wandb=args.wandb,
            description=args.description,
            batchsize_finder=args.batchsize_finder,
            periodic_test=args.periodic_test,
        )
        Tm.N.compiled = args.compiled
        Tm.fit()


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train FRAN model up to Tm.fit(), no preprocessing."
    )
    parser.add_argument(
        "-t", "--project-title", help="project title", dest="project_title"
    )
    parser.add_argument(
        "-p",
        "--plan",
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
        "-lr", "--learning-rate", dest="lr", type=float, default=None
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
    parser.add_argument("-e" , "--epochs", type=int, default=600, help="Max epochs")
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
        "--wandb", dest="wandb", type=str2bool, default=True, help="Enable W&B logging"
    )
    parser.add_argument(
        "--neptune",
        dest="wandb",
        type=str2bool,
        help="Deprecated alias for --wandb",
    )
    parser.add_argument(
        "-r",
        "--run-name",
        dest="run_name",
        default=None,
        action=UniqueArgValue,
        help='Run name (e.g., "LITS-1290")',
    )
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
    parser.add_argument( "--periodic-test", type=int, default=0, help="Test every n epochs. Default (0) means no test is done")
    parser.add_argument( "--bsf",
        "--batchsize-finder", type=str2bool, default=False, help="Enable batch size finder", dest="batchsize_finder"
    )
    parser.add_argument("--incremental", type=str2bool, default=False, help="Enable incremental curriculum training loop")
    parser.add_argument("--initial-samples", type=int, default=32, help="Initial number of train cases")
    parser.add_argument("--add-samples", type=int, default=16, help="Number of cases to add per stage")
    args = parser.parse_known_args()[0]
# %%
    # args.fold = 1
    # args.project = "nodes"
    # #
    # args.devices = '1'
    #
    # headline("DEVS")
    # print(args.devices)
    # print("After parse:")
    # print(parse_devices(args.devices))
# %%
    main(args)
# %%
