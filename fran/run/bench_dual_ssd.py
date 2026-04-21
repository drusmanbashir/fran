from __future__ import annotations

import argparse
import copy
import time

import torch
from fastcore.basics import warnings
from fran.configs.parser import ConfigMaker
from fran.managers.data.training import DataManagerDual
from fran.managers.project import Project
from fran.transforms.imageio import LoadTorchd
from utilz.imageviewers import ImageMaskViewer

warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

torch.set_float32_matmul_precision("medium")
from fran.utils.common import *


def build_datamodule(dual_ssd: bool):
    project_title = "kits23"
    proj = Project(project_title=project_title)

    CL = ConfigMaker(proj)
    CL.setup(2)
    conf = copy.deepcopy(CL.configs)

    batch_size = 4
    ds_type = "lmdb"
    ds_type = None
    proj_tit = proj.project_title
    conf["dataset_params"]["cache_rate"] = 0.0

    D = DataManagerDual(
        project_title=proj_tit,
        configs=conf,
        batch_size=batch_size,
        ds_type=ds_type,
        dual_ssd=dual_ssd,
    )

    D.prepare_data()
    D.setup("fit")
    return D


def batch_filenames(batch):
    fns = batch["image"].meta["filename_or_obj"]
    return [f.split("/")[-1] for f in fns]


def time_valid_loader(dual_ssd: bool, print_batches: bool = False):
    print(f"\n=== dual_ssd={dual_ssd} ===", flush=True)
    D = build_datamodule(dual_ssd=dual_ssd)
    tmv = D.valid_manager
    tmt = D.train_manager
    tmv.transforms_dict

    dl = tmv.dl
    n_batches = 0
    n_items = 0
    first_fns = None
    first_is_padded = None

    start = time.perf_counter()
    for batch in dl:
        if first_fns is None:
            first_fns = batch_filenames(batch)
            first_is_padded = batch["is_padded"]
        if print_batches:
            print(batch_filenames(batch))
            print(batch["is_padded"])
        n_batches += 1
        n_items += len(batch["image"])
    elapsed = time.perf_counter() - start

    print(f"first_fns={first_fns}", flush=True)
    print(f"first_is_padded={first_is_padded}", flush=True)
    print(
        f"dual_ssd={dual_ssd} batches={n_batches} items={n_items} "
        f"seconds={elapsed:.3f} batches_per_s={n_batches / elapsed:.3f} "
        f"items_per_s={n_items / elapsed:.3f}",
        flush=True,
    )
    return {
        "dual_ssd": dual_ssd,
        "batches": n_batches,
        "items": n_items,
        "seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-batches", action="store_true")
    args = parser.parse_args()

    results = [
        time_valid_loader(False, print_batches=args.print_batches),
        time_valid_loader(True, print_batches=args.print_batches),
    ]
    base, dual = results
    speedup = base["seconds"] / dual["seconds"]
    print(
        f"\nsummary false_seconds={base['seconds']:.3f} "
        f"true_seconds={dual['seconds']:.3f} speedup={speedup:.3f}x",
        flush=True,
    )


if __name__ == "__main__":
    main()
