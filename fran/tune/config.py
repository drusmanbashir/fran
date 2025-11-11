# %%
import sys

import ipdb
import numpy as np
import yaml

from fran.configs.parser import ConfigMaker

MNEMONICS = ["litsmall", "lits", "litq", "liver", "lidc", "lungs", "nodes", "totalseg"]
tr = ipdb.set_trace

if not sys.executable == "":  # workaround for slicer as it does not load ray tune
    from ray import tune

from utilz.helpers import *


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



def load_tune_template(project="base"):
    import importlib.resources
    import fran.templates as tl
    with importlib.resources.files(tl).joinpath("tune.yaml").open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg[project]



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

            # try:
            #     print(tune_k, val_sample.sample())
            # except:
            #
            #     print(tune_k, val_sample[0].sample())
            #     print("upper: ", val_sample[1].sample())

            cfg_dict[tune_k] = val_sample
        return cfg_dict



    #
