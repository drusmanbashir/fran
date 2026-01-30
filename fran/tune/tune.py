# %%
import ast
import os
import re
from datetime import datetime
# from fran.architectures.unet3d.model import  *
from pathlib import Path

import ipdb
import torch
from ray.tune.schedulers import ASHAScheduler
from utilz.string import headline

from fran.managers import Project
from fran.tune.config import RayTuneConfig, out_channels_from_dict_or_cell
from fran.tune.trainer import RayTrainer

tr = ipdb.set_trace
from ray import tune
from ray.tune import FailureConfig
from utilz.fileio import load_json
from utilz.helpers import set_autoreload

from fran.architectures.create_network import create_model_from_conf
from fran.architectures.unet3d.model import UNet3D
from fran.configs.parser import (load_metadata, make_patch_size)
from fran.managers.base import load_checkpoint
# only vars below will be tuned
from fran.utils.common import COMMON_PATHS

ray_results_folder = Path(COMMON_PATHS["ray_results_folder"])
OOM_RE = re.compile(r"CUDA out of memory", re.IGNORECASE)


def load_model_from_raytune_trial(folder_name, out_channels):
    # requires params.json inside raytune trial
    params_dict = load_json(Path(folder_name) / "params.json")
    model = create_model_from_conf(params_dict, out_channels)

    folder_name / ("model_checkpoints")
    list((folder_name / ("model_checkpoints")).glob("model*"))[0]
    load_checkpoint  # (folder_name / ("model_checkpoints"), model)
    # state_dict= torch.load(chkpoint_filename)
    # model.load_state_dict(state_dict['model'])
    return model


class ModelFromTuneTrial:
    def __init__(self, proj_defaults, trial_name, out_channels=None):
        self.metadata = load_metadata(
            settingsfilename=proj_defaults.configuration_filename
        )

        folder_name = get_raytune_folder_from_trialname(proj_defaults, trial_name)
        self.params_dict = load_json(Path(folder_name) / "params.json")
        if out_channels is None:
            if "out_channels" not in self.params_dict["model_params"]:
                self.dest_labels = ast.literal_eval(
                    self.metadata["remapping_train"].item()
                )
                out_channels = out_channels_from_dict_or_cell(self.dest_labels)
            else:
                out_channels = self.params_dict["model_params"]["out_channels"]
        self.model = load_model_from_raytune_trial(folder_name, out_channels)


def model_from_config(config):
    model = UNet3D(
        in_channels=1,
        out_channels=2,
        final_sigmoid=False,
        f_maps=config["base_ch_opts"],
        num_levels=config["depth_opts"],
    )
    return model


def get_raytune_folder_from_trialname(proj_defaults, trial_name: str):
    pat = re.compile("([a-z]*_[0-9]*)_", flags=re.IGNORECASE)
    experiment_name = re.match(pat, trial_name).groups()[0]
    folder = proj_defaults.checkpoints_parent_folder / experiment_name
    assert folder.exists(), "Experiment name not in checkpoints_folder"
    tn = trial_name + "*"
    folder_name = list(folder.glob(tn))[0]
    return folder_name


def trial_str_creator(trial):
    format_string = "%f"
    a_datetime_datetime = datetime.now()
    current_time_string = a_datetime_datetime.strftime(format_string)
    trial_id = "{0}_{1}".format(trial, current_time_string[:3])
    return trial_id


def trial_dirname_creator(trial):
    return trial.custom_trial_name


def store_experiment_name_in_config(proj_defaults, config, experiment_name):
    if not config["metadata"]["most_recent_ray_experiment"] == experiment_name:
        pass
        # Not implemented yet


def tune_from_spec(tune_type: str, tune_value, q=None):
    """
    Convert a YAML tune spec to a ray.tune object.
    Minimal mapping, no heuristics.
    """
    # Special case used in your YAML: two independent uniform ranges
    if tune_type == "double_range":
        lo1, hi1 = tune_value[0]
        lo2, hi2 = tune_value[1]
        return [tune.uniform(lo1, hi1), tune.uniform(lo2, hi2)]

    # choice expects a list of options
    if tune_type == "choice":
        return tune.choice(tune_value)

    # simple 2-tuple/2-elem list ranges
    if tune_type in {"uniform", "loguniform", "randint"}:
        lo, hi = tune_value
        return getattr(tune, tune_type)(lo, hi)

    # quantized distributions (q provided in YAML)
    if tune_type in {"quniform", "qloguniform", "qrandint"}:
        assert q is not None, f"{tune_type} requires 'q' in YAML."
        lo, hi = tune_value
        return getattr(tune, tune_type)(lo, hi, q=q)

    # pass-through for any other Ray Tune function that takes a list/args
    fn = getattr(tune, tune_type)
    # If value is a list/tuple, try arg-unpack; else pass directly
    if isinstance(tune_value, (list, tuple)):
        try:
            return fn(*tune_value)
        except TypeError:
            return fn(tune_value)
    return fn(tune_value)


def setup_tune_params(configs):
    configs["dataset_params"]["src_dims"] = make_patch_size(
        configs["dataset_params"]["src_dim0"], configs["dataset_params"]["src_dim1"]
    )
    configs["plan_train"]["patch_size"] = make_patch_size(
        configs["plan_train"]["patch_dim0"], configs["plan_train"]["patch_dim1"]
    )
    configs["plan_valid"]["patch_size"] = configs["plan_train"]["patch_size"]
    return configs


def train_with_tune(config, project_title, num_epochs=10):
    compiled = False
    # NOTE: if Neptune = False, should store checkpoint locally
    neptune = False
    override_dm = False
    tags = []
    description = f"Ray tune"
    Tm = RayTrainer(project_title, config, None)
    devices = 1

    config = setup_tune_params(config)

    headline(f"Training with config: {config}")

    lr = config["model_params"]["lr"]
    if config["dataset_params"]["src_dims"][0] > 160:
        bs = 1
    else:
        bs = 2

    headline(config["dataset_params"]["src_dims"])

    headline(config["plan_train"]["patch_size"])

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())

    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=devices,
        epochs=num_epochs,
        batchsize_finder=True,
        profiler=False,
        neptune=neptune,
        tags=tags,
        description=description,
        lr=lr,
        override_dm_checkpoint=override_dm,
    )
    Tm.fit()
    # while True:
    #     try:
    #         # re-setup each attempt so dataloaders are rebuilt with new bs
    #         Tm.setup(
    #             compiled=compiled,
    #             batch_size=bs,
    #             devices=devices,
    #             epochs=num_epochs,
    #             batchsize_finder=True,
    #             profiler=False,
    #             neptune=neptune,
    #             tags=tags,
    #             description=description,
    #             lr=lr,
    #             override_dm_checkpoint=override_dm,
    #         )
    #         Tm.fit()  # if this finishes, we’re done
    #         break
    #     except RuntimeError as e:
    #         if OOM_RE.search(str(e)):
    #             new_bs = bs // 2
    #             if new_bs < 2:
    #                 raise  # don't go below 2
    #             print(f"[OOM] bs={bs} -> {new_bs}. Retrying...")
    #             torch.cuda.empty_cache()
    #             bs = new_bs
    #             continue
    #         raise
    #


if __name__ == "__main__":
# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>

    set_autoreload()

    P = Project(project_title="nodes")
    project = P

    C = RayTuneConfig(P)
    C.setup()
    conf = C.configs
# %%
    import argparse

    parser = argparse.ArgumentParser(description="Resampler")

    parser.add_argument(
        "-t", "--project-title", help="project title", dest="project_title"
    )
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        help="number of parallel processes",
        default=1,
    )
    parser.add_argument("-p", "--plan", type=int, help="Just a number like 1, 2")

    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_known_args()[0]
# %%
    args.project_title = P.project_title
    args.plan = conf["plan_train"]
    args.num_processes = 8
    args.overwrite = False
    #
# %%
    reporter = tune.CLIReporter(
        metric_columns=["loss"],
        # parameter_columns=["lr", "batch_size"],
    )
# %%
    # num_gpus = 2
    num_gpus = 2
    gpus_per_trial = 1
    resources_per_trial = {"cpu": 8.0, "gpu": gpus_per_trial}
    num_samples = 50
    # C._set_active_plans(1,1)
    # C.add_output_labels()
    # C.add_out_channels()

    # C.add_dataset_props()
    num_epochs = 300
    grace_period = 20
# %%

    tune_fn_with_params = tune.with_parameters(
        train_with_tune, project_title=P.project_title, num_epochs=num_epochs
    )

# %%
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=3, reduction_factor=2)
    tuner = tune.Tuner(
        tune.with_resources(tune_fn_with_params, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
            max_concurrent_trials=num_gpus,
        ),
        run_config=tune.RunConfig(
            name="tune_UNET",
            progress_reporter=reporter,
            failure_config=FailureConfig(max_failures=2),  # retry actor if it crashes
            storage_path=ray_results_folder,
        ),
        param_space=conf,
    )
    results = tuner.fit()

    #     from fran.run.analyze_resample import main
    # main(args)

