# %%

# from fran.callback.neptune_manager import *
from fran.utils.config_parsers import *
import torch.nn as nn
from ray import tune
from ray.air import session
from fran.managers.trainer import *
from datetime import datetime
import pandas as pd
import ast,re
from ray import tune
from fran.utils.config_parsers import load_metadata
from fran.managers.base import load_checkpoint
from fran.utils.fileio import load_json
from fran.utils.helpers import make_channels
from fran.architectures.unet3d.model import  *
from pathlib import Path
# %%
class ModelFromTuneTrial():

    def __init__(self, proj_defaults, trial_name,out_channels=None):

        self.metadata = load_metadata(settingsfilename=proj_defaults.configuration_filename)

        folder_name = get_raytune_folder_from_trialname(proj_defaults, trial_name)
        self.params_dict = load_json(Path(folder_name)/"params.json")
        if out_channels is None :
            if not 'out_channels' in self.params_dict['model_params']:
                self.dest_labels = ast.literal_eval(self.metadata['src_dest_labels'].item())
                out_channels = out_channels_from_dict_or_cell(self.dest_labels)
            else:
                out_channels = self.params_dict['model_params']['out_channels']
        self.model = load_model_from_raytune_trial(folder_name,out_channels)


def model_from_config(config):
    model=UNet3D(in_channels=1,out_channels=2,final_sigmoid=False,f_maps = config['base_ch_opts'],num_levels=config['depth_opts'])
    return model


def load_model_from_raytune_trial(folder_name,out_channels):
    #requires params.json inside raytune trial
    params_dict = load_json(Path(folder_name)/"params.json")
    model =create_model_from_conf(params_dict,out_channels)
    
    folder_name/("model_checkpoints")
    list((folder_name/("model_checkpoints")).glob("model*"))[0]
    load_checkpoint(folder_name / ("model_checkpoints"), model)
    # state_dict= torch.load(chkpoint_filename)
    # model.load_state_dict(state_dict['model'])
    return  model

def get_raytune_folder_from_trialname(proj_defaults, trial_name:str):
    pat = re.compile("([a-z]*_[0-9]*)_",flags=re.IGNORECASE)
    experiment_name= re.match(pat,trial_name).groups()[0]
    folder = proj_defaults.checkpoints_parent_folder/experiment_name
    assert folder.exists(), "Experiment name not in checkpoints_folder"
    tn = trial_name+"*"
    folder_name = list(folder.glob(tn))[0]
    return folder_name


class RayTuneManager(object):
    def __init__(self,raytune_settingsfile=None) -> None:
         self.raytune_settingsfile = 'experiments/experiment_config.xlsx' if not raytune_settingsfile else raytune_settingsfile
         # self.book =  load_workbook(self.raytune_settingsfile)
         # self.writer = pd.ExcelWriter(self.raytune_settingsfile,engine='openpyxl')
         self.df_meta= pd.read_excel(self.raytune_settingsfile,sheet_name='metadata')
         # self.metadata = pd.read_excel(self.raytune_settingsfile,sheet_name='metadata')
    def load_config(self,sheet_name):
        df_ray = pd.read_excel(self.raytune_settingsfile,sheet_name=sheet_name)
        config = {}
        for row in df_ray.iterrows():
            rr = row[1]
            var_type = rr['type']
            tune_fnc = getattr(tune,var_type)

            key = rr['var_name']
            vals =ast.literal_eval(rr['values']) 
            
            if var_type =="randint":
                val_sample = tune_fnc(vals[0],vals[1])
            else:
                val_sample = tune_fnc(vals)
            config.update({key:val_sample})
        return config

    def model_from_config(self,config):
        chs = make_channels(config['base_ch_opts'],config['depth_opts'])
        model = UNet3D(chs=chs,num_classes=2,base_conv7x7=config['base_conv7x7'],n_bottleneck=config['n_bottleneck'])
        return model

    def retrieve_metadata(self):
        return self.df_meta

    def update_metadata(self, metadata):
        df_new = pd.DataFrame(metadata, index=[0])
        with pd.ExcelWriter(self.raytune_settingsfile, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df_new.to_excel(writer, sheet_name = 'metadata',index=False)



## 
def trial_str_creator(trial ):
    format_string = "%f"
    a_datetime_datetime = datetime.now()
    current_time_string = a_datetime_datetime.strftime(format_string)
    trial_id =  "{0}_{1}".format(trial,current_time_string[:3])
    return trial_id

def trial_dirname_creator(trial):
    return trial.custom_trial_name

def store_experiment_name_in_config(proj_defaults, config, experiment_name):
    if not config['metadata']['most_recent_ray_experiment']== experiment_name:
        pass
        # Not implemented yet


def train_with_tune(multi_gpu,neptune, max_num_epochs,proj_defaults, config):
    store_experiment_name_in_config(proj_defaults, config, session.get_experiment_name())
    La = Trainer(proj_defaults=proj_defaults, config_dict=config, bs = config['dataset_params']['bs'])
    lr = config['model_params']['lr']
    cbs = [TuneTrackerCallback(freq=6), TuneCheckpointCallback(freq=6)]
    if neptune==True:
        cbs+=[
        NeptuneCallback(proj_defaults,config,run_name=tune.get_trial_name(),freq=6),
        NeptuneCheckpointCallback(resume=False,checkpoints_parent_folder= proj_defaults.checkpoints_parent_folder),
        NeptuneImageGridCallback(classes = out_channels_from_dict_or_cell(config['metadata']['src_dest_labels']),
                         patch_size= make_patch_size(config['dataset_params']['patch_dim0'], config['dataset_params']['patch_dim1'])),

        ] # resume = False because TuneCheckpointCallback handles all resumptions

    learn = La.create_learner(cbs = cbs,device=None)

    if multi_gpu==True:
        learn.model= nn.DataParallel(learn.model)

    if config['model_params']['one_cycle']== True:
        moms = (config['model_params']['mom_high'],config['model_params']['mom_low'],config['model_params']['mom_high'])
        moms2 = (float(m) for m in moms)

        learn.fit_one_cycle(n_epoch=max_num_epochs,lr_max=lr,moms = moms2)
    else:
        learn.fit(max_num_epochs, lr)
# %%
# %%
if __name__ == "__main__":
# %%

    single_gpu=True
    if single_gpu==True:
        ray.init(local_mode=True,num_cpus=1,num_gpus=2)
    debug_mode = False
    r = RayTuneManager()
    conf = r.load_config(sheet_name="model_params")
    meta = r.retrieve_metadata()
    print(meta)
    
    print(conf)
# %%

