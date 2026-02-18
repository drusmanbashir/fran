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
            inc = IncrementalTrainer(project_title=P.project_title, configs=conf)
            out = inc.run(
                initial_samples_n=args.initial_samples,
                add_samples_y=args.add_samples,
                epochs_per_stage=args.stage_epochs,
                max_stages=args.max_stages,
                selection_threshold=args.selection_threshold,
                neptune=args.neptune,
                devices=devices,
                batch_size=args.batch_size,
                lr=args.lr,
                min_lr_to_continue=args.min_lr_to_continue,
                w_dice=args.w_dice,
                w_uncertainty=args.w_uncertainty,
                w_diversity=args.w_diversity,
                run_id=args.inc_run_id,
                periodic_test=args.periodic_test,
                early_stopping_patience=args.early_stopping_patience,
            )
            headline(f"Incremental training completed: {out}")
            return

        Tm = Trainer(project_title=P.project_title, configs=conf, run_name=args.run_name)
        Tm.setup(
            compiled=args.compiled,
            batch_size=args.batch_size,
            cbs=cbs,
            devices=devices,
            epochs=args.epochs if not args.profiler else 1,
            lr=args.lr,
            profiler=args.profiler,
            neptune=args.neptune,
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
    parser.add_argument( "--periodic-test", type=int, default=0, help="Test every n epochs. Default (0) means no test is done")
    parser.add_argument( "--bsf",
        "--batchsize-finder", type=str2bool, default=False, help="Enable batch size finder", dest="batchsize_finder"
    )
    parser.add_argument("--incremental", type=str2bool, default=False, help="Enable incremental curriculum training loop")
    parser.add_argument("--initial-samples", type=int, default=32, help="Initial number of train cases")
    parser.add_argument("--add-samples", type=int, default=16, help="Number of cases to add per stage")
    parser.add_argument("--stage-epochs", type=int, default=150, help="Max epochs per incremental stage")
    parser.add_argument("--max-stages", type=int, default=10, help="Maximum incremental stages")
    parser.add_argument("--selection-threshold", type=float, default=0.7, help="Add only cases with Dice <= threshold")
    parser.add_argument("--w-dice", type=float, default=0.5, help="Combined score weight for Dice difficulty (1-Dice)")
    parser.add_argument("--w-uncertainty", type=float, default=0.3, help="Combined score weight for predictive uncertainty")
    parser.add_argument("--w-diversity", type=float, default=0.2, help="Combined score weight for diversity")
    parser.add_argument("--min-lr-to-continue", type=float, default=None, help="Stop incremental loop once LR reaches this floor")
    parser.add_argument("--inc-run-id", default=None, help="Optional run id for curriculum logs")
    parser.add_argument("--early-stopping-patience", type=int, default=20, help="Patience for stage-wise early stopping")

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
