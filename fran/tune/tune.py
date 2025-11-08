# %%
import ast
from pprint import pp
import re
from datetime import datetime
from fran.managers import Project
# from fran.architectures.unet3d.model import  *
from pathlib import Path
import ipdb
tr = ipdb.set_trace
import numpy as np


import pandas as pd
import torch.nn as nn
from ray import tune
from ray.air import session
from utilz.fileio import load_json
from utilz.helpers import make_channels, set_autoreload

from fran.architectures.create_network import create_model_from_conf
from fran.architectures.unet3d.model import UNet3D
from fran.callback.nep import NeptuneImageGridCallback
from fran.managers.base import load_checkpoint, make_patch_size
from fran.configs.parser import ConfigMaker
from fran.configs.parser import load_metadata
import yaml

def load_tune_template(project="base"):
    import importlib.resources
    import fran.templates as tl
    with importlib.resources.files(tl).joinpath("tune.yaml").open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg[project]


#only vars below will be tuned
TUNE_VARS =[
    "base_ch_opts",
    "lr",
    "deep_supervision",
    "src_dim0",
    "src_dim1",
    "contrast",
    "shift",
    "scale",
    "brightness",
    "expand_by",
    "patch_dim0",
    "patch_dim1",
    "patch_overlap",
]


def resolve_tune_fnc(tune_type: str):
    if "_" in tune_type:
        return getattr(tune, tune_type.split("_")[0])
    else:
        return getattr(tune, tune_type)

def out_channels_from_dict_or_cell(src_dest_labels):  
    if isinstance(src_dest_labels, pd.core.series.Series):
        src_dest_labels = ast.literal_eval(src_dest_labels.item())
    out_channels = max([src_dest[1] for src_dest in src_dest_labels])+1
    return out_channels

def load_model_from_raytune_trial(folder_name,out_channels):
    #requires params.json inside raytune trial
    params_dict = load_json(Path(folder_name)/"params.json")
    model =create_model_from_conf(params_dict,out_channels)
    
    checkpoints_folder=folder_name/("model_checkpoints")
    chkpoint_filename = list((folder_name/("model_checkpoints")).glob("model*"))[0]
    load_checkpoint#(folder_name / ("model_checkpoints"), model)
    # state_dict= torch.load(chkpoint_filename)
    # model.load_state_dict(state_dict['model'])
    return  model

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


def train_with_tune(multi_gpu, neptune, max_num_epochs, proj_defaults, config):
    store_experiment_name_in_config(
        proj_defaults, config, session.get_experiment_name()
    )
    La = Trainer(
        proj_defaults=proj_defaults,
        config_dict=config,
        bs=config["dataset_params"]["bs"],
    )
    lr = config["model_params"]["lr"]
    cbs = [TuneTrackerCallback(freq=6), TuneCheckpointCallback(freq=6)]
    if neptune == True:
        cbs += [
            NeptuneCallback(
                proj_defaults, config, run_name=tune.get_trial_name(), freq=6
            ),
            NeptuneCheckpointCallback(
                resume=False,
                checkpoints_parent_folder=proj_defaults.checkpoints_parent_folder,
            ),
            NeptuneImageGridCallback(
                classes=out_channels_from_dict_or_cell(
                    config["metadata"]["remapping_train"]
                ),
                patch_size=make_patch_size(
                    config["dataset_params"]["patch_dim0"],
                    config["dataset_params"]["patch_dim1"],
                ),
            ),
        ]  # resume = False because TuneCheckpointCallback handles all resumptions

    learn = La.create_learner(cbs=cbs, device=None)

    if multi_gpu == True:
        learn.model = nn.DataParallel(learn.model)

    if config["model_params"]["one_cycle"] == True:
        moms = (
            config["model_params"]["mom_high"],
            config["model_params"]["mom_low"],
            config["model_params"]["mom_high"],
        )
        moms2 = (float(m) for m in moms)

        learn.fit_one_cycle(n_epoch=max_num_epochs, lr_max=lr, moms=moms2)
    else:
        learn.fit(max_num_epochs, lr)


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

class RayTuneConfig(ConfigMaker):
    def __init__(self, project, configuration_filename=None):
        super().__init__(project, configuration_filename=configuration_filename)
        if not "mom_low" in self.configs["model_params"].keys() :
            config = {
                "mom_low": tune.sample_from(
                    lambda spec: np.random.uniform(0.6, 0.9100)
                ),
                "mom_high": tune.sample_from(
                    lambda spec: np.minimum(
                        0.99,
                        spec.config.model_params.mom_low
                        + np.random.uniform(low=0.05, high=0.35),
                    )
                ),
            }
            self.configs["model_params"].update(config)
        self.tune_template=load_tune_template(project="base")

    def setup(self):
        super().setup(plan_train=1, plan_valid=1)
        self.insert_tune_vars()

    def insert_tune_vars(self):
        self.patch_dim0_computed,self.src_dim0_computed = False, False
        self.insert_tune_vars_dict(self.configs["dataset_params"])
        self.insert_tune_vars_dict(self.configs["model_params"])
        self.insert_tune_vars_dict(self.configs["plan_train"])

    #
    # def get_tune_variable(self,tune_k)->tuple:
    #             try:
    #                 rr= self.tune_template[tune_k]
    #             except:
    #
    #                 if tune_k == "patch_size" :
    #                     if self.patch_dim0_computed==False:
    #                         tune_k = "patch_dim0"
    #                         self.patch_dim0_computed=True
    #                     else:
    #                         tune_k = "patch_dim1"
    #                         tr()
    #                     rr= self.tune_template[tune_k]
    #                 elif tune_k == "src_size" :
    #                     if self.src_dim0_computed==False:
    #                         tune_k = "src_dim0"
    #                         self.src_dim0_computed=True
    #                     else:
    #                         tune_k = "src_dim1"
    #                 else:
    #                     tr()
    #                 rr = self.tune_template[tune_k]
    #             return rr, tune_k

    def insert_tune_vars_dict(self, cfg_dict):
        tune_keys =list(set(cfg_dict.keys()).intersection(set(TUNE_VARS)))
        for i in range(0,len(tune_keys)):
            tune_k = tune_keys[i]
            rr =  self.tune_template[tune_k]
            var_type = rr['type']
            if(
                var_type == "randint"
                or var_type == "loguniform"
                or var_type == "uniform"
            ):
                tune_fnc = resolve_tune_fnc(var_type)
                vals = rr["value"]
                val_sample = tune_fnc(vals[0], vals[1])
            elif rr["type"] == "double_range":
                val_lower, val_upper = rr["value"]
                val_lower = tune.uniform(lower=val_lower[0], upper=val_lower[1])
                val_upper = tune.uniform(lower=val_upper[0], upper=val_upper[1])
                val_sample = [val_lower, val_upper]
            elif rr["type"] == "choice":
                vals = rr["value"]
                val_sample = tune.choice(vals)
            else:
                tune_fnc = resolve_tune_fnc(var_type)
                if tune_fnc.__name__[0] == "q":
                    quant = float(rr["q"])

                val_lower, val_upper = rr["value"]
                val_sample = tune_fnc(lower=val_lower, upper=val_upper,q=quant)

            try:
                print(tune_k, val_sample.sample())
            except:

                print(tune_k, val_sample[0].sample())
                print("upper: ", val_sample[1].sample())

            cfg_dict[tune_k] = val_sample
        return cfg_dict

if __name__ == "__main__":
# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------

    set_autoreload()

    P = Project(project_title="lidc")
    project = P

    C = RayTuneConfig(P)
    C.setup()
# %%
    # C._set_active_plans(1,1)
    # C.add_output_labels()
    # C.add_out_channels()
    # C.add_dataset_props()

# %%
    C.configs["dataset_params"]
    C.configs["model_params"]
# %%
    conf = C.configs
    pp(conf['plan_train'])

# %%

    cfg_dict = conf['plan_train']
    tune_keys =list(set(cfg_dict.keys()).intersection(set(TUNE_VARS)))
    for i in range(0,len(tune_keys)):
        tune_k = tune_keys[i]
        rr , tune_k= C.get_tune_variable(tune_k)
        if tune_k=="patch_dim1":
            tr()
        var_type = rr['type']
        if(
            var_type == "randint"
            or var_type == "loguniform"
            or var_type == "uniform"
        ):
            tune_fnc = resolve_tune_fnc(var_type)
            vals = rr["value"]
            val_sample = tune_fnc(vals[0], vals[1])
        elif rr["type"] == "double_range":
            val_lower, val_upper = rr["value"]
            val_lower = tune.uniform(lower=val_lower[0], upper=val_lower[1])
            val_upper = tune.uniform(lower=val_upper[0], upper=val_upper[1])
            val_sample = [val_lower, val_upper]
        elif rr["type"] == "choice":
            vals = rr["value"]
            val_sample = tune.choice(vals)
        else:
            tune_fnc = resolve_tune_fnc(var_type)
            if tune_fnc.__name__[0] == "q":
                quant = float(rr["q"])

            val_lower, val_upper = rr["value"]
            val_sample = tune_fnc(lower=val_lower, upper=val_upper,q=quant)

        try:
            print(tune_k, val_sample.sample())
        except:

            print(tune_k, val_sample[0].sample())
            print("upper: ", val_sample[1].sample())

        cfg_dict[tune_k] = val_sample


    pp(cfg_dict)
# %%
    C.plans
    single_gpu = True
    if single_gpu == True:
        try:
            ray.init(local_mode=True, num_cpus=1, num_gpus=2)
        except:
            pass
    cfg = load_tune_template(project="base")
    C.configs
    # dsp = C.configs['dataset_params']
    # mp = C.configs['model_params']
    # pl = C.configs['plan_train']
    # tf = C.configs["transform_factors"]
    # tune_keys =list(set(dsp.keys()).intersection(set(TUNE_VARS)))
    # tune_keys =list(set(mp.keys()).intersection(set(TUNE_VARS)))
    # tune_keys =list(set(pl.keys()).intersection(set(TUNE_VARS)))
# %%
# %%
    debug_mode = False
    df_ray = pd.read_excel("/home/ub/code/fran/configurations/experiment_configs_liver.xlsx", sheet_name="model_params")
    # df_ray = df_ray.dropna(subset=["tune_value"])
    df_ray = df_ray[~df_ray["tune"].isin([0,False])]
# %%
    r = RayTuneManager("/home/ub/code/fran/configurations/experiment_configs_liver.xlsx")
    conf = r.load_config(sheet_name="model_params")

#
