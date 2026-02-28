#!/usr/bin/env python3
import argparse
from typing import Any

from fran.configs.parser import ConfigMaker, confirm_plan_analyzed
from fran.data.dataregistry import DS
from fran.managers import Project
from fran.run.analyze_resample import PreprocessingManager
from fran.trainers.trainer_bk import TrainerBK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quick FRAN test pipeline using drli_short + liver plan 1."
    )
    parser.add_argument("--project-title", default="liver_test", help="New/existing project title.")
    parser.add_argument("--mnemonic", default="liver", help="Mnemonic for experiment config selection.")
    parser.add_argument("--dataset", default="drli_short", help="Datasource key from DatasetRegistry.")
    parser.add_argument("--plan", type=int, default=1, help="Active plan id.")
    parser.add_argument("--fold", type=int, default=0, help="Training fold.")
    parser.add_argument("--gpu-id", type=int, default=1, help="GPU index to use.")
    parser.add_argument("--epochs", type=int, default=5, help="Quick test epoch count.")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--cache-rate", type=float, default=0.0, help="Dataset cache rate.")
    parser.add_argument("--num-processes", type=int, default=1, help="Preprocessing workers.")
    parser.add_argument("--overwrite-preprocess", action="store_true", help="Force preprocessing overwrite.")
    return parser


def apply_dataset_overrides(conf: dict[str, Any], dataset: str, fold: int, cache_rate: float) -> dict[str, Any]:
    for plan_key in ("plan_train", "plan_valid", "plan_test"):
        if plan_key in conf:
            conf[plan_key]["datasources"] = dataset
    conf["dataset_params"]["fold"] = int(fold)
    conf["dataset_params"]["cache_rate"] = float(cache_rate)
    return conf


def ensure_project(project_title: str, mnemonic: str, dataset: str) -> Project:
    project = Project(project_title)
    if not project.db.exists():
        project.create(mnemonic=mnemonic, datasources=[DS[dataset]])
    elif dataset not in project.datasources:
        project.add_data([DS[dataset]])

    if not project.has_folds:
        project._create_folds()
    if "labels_all" not in project.global_properties or len(project.global_properties["labels_all"]) == 0:
        project.set_labels_all()
        project.save_global_properties()
    if "mean_dataset_clipped" not in project.global_properties:
        project.maybe_store_projectwide_properties(overwrite=False, multiprocess=False)
    return project


def ensure_preprocessed(
    project: Project, conf: dict[str, Any], plan_id: int, overwrite: bool, num_processes: int
) -> None:
    plan = conf["plan_train"]
    completed = confirm_plan_analyzed(project, plan)
    if all(completed.values()) and not overwrite:
        return

    args = argparse.Namespace(
        project_title=project.project_title,
        plan=int(plan_id),
        num_processes=int(num_processes),
        overwrite=bool(overwrite),
        debug=False,
        no_fix=False,
    )
    manager = PreprocessingManager(args, conf=conf)
    manager.resample_dataset(overwrite=overwrite, num_processes=num_processes)
    if plan["mode"] == "patch":
        manager.generate_hires_patches_dataset(overwrite=overwrite)
    elif plan["mode"] == "lbd":
        imported_folder = plan.get("imported_folder", None)
        if imported_folder is None:
            manager.generate_lbd_dataset(overwrite=overwrite, num_processes=num_processes)
        else:
            manager.generate_TSlabelboundeddataset(overwrite=overwrite, num_processes=num_processes)


def run_training(project: Project, conf: dict[str, Any], gpu_id: int, epochs: int, batch_size: int) -> None:
    trainer = TrainerBK(project.project_title, conf, run_name=None)
    trainer.setup(
        compiled=False,
        batch_size=int(batch_size),
        devices=[int(gpu_id)],
        epochs=int(epochs),
        batchsize_finder=False,
        profiler=False,
        wandb=False,
        periodic_test=0,
        tags=["pipeline:test"],
        description="Quick workflow validation run.",
    )
    trainer.N.compiled = False
    trainer.fit()


def run_pipeline(args: argparse.Namespace) -> None:
    project = ensure_project(args.project_title, args.mnemonic, args.dataset)
    config_maker = ConfigMaker(project)
    config_maker.setup(args.plan)
    conf = apply_dataset_overrides(config_maker.configs, args.dataset, args.fold, args.cache_rate)
    ensure_preprocessed(
        project=project,
        conf=conf,
        plan_id=args.plan,
        overwrite=args.overwrite_preprocess,
        num_processes=args.num_processes,
    )
    run_training(project, conf, args.gpu_id, args.epochs, args.batch_size)


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    run_pipeline(parsed)
