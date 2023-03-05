# %%
from pathlib import Path
import os,sys
sys.path+= ["/home/ub/Dropbox/code"]
from types import SimpleNamespace
from fran.utils.fileio import *
import json, yaml

# proj_defaults= dict()
# proj_defaults["num_processes"]= 32
# proj_defaults["project_title"]= "kits19"
# proj_defaults["tumour_label_new"] = 9   # 
# proj_dfaults["tumour_label_dataset"]=2
#

from fran.utils.fileio import maybe_makedirs

# %%
# def load_proj_defaults(project_title: str, num_processes: int=8):
#     proj_defaults = {'project_title':project_title}
#     proj_defaults["num_processes"]=num_processes
#     proj_defaults["ssd"] = Path("/home/ub")
#     proj_defaults["storage"] = Path("/s")
#     proj_defaults["raw_data_folder"] =proj_defaults["storage"]/("datasets/raw_data/"+proj_defaults["project_title"])
#
#     proj_defaults["preprocessing_output_folder"]= proj_defaults["ssd"]/("datasets/preprocessed/"+proj_defaults["project_title"])
#     proj_defaults["stage0_folder"]= proj_defaults["preprocessing_output_folder"]/"stage0"
#     proj_defaults["stage1_folder"]= proj_defaults["preprocessing_output_folder"]/"stage1"
#     # files etc
#     proj_defaults["raw_dataset_properties_filename"]=proj_defaults["raw_data_folder"]/"raw_dataset_properties"
#     proj_defaults["global_properties_filename"]=proj_defaults["raw_data_folder"]/"global_properties"
#     proj_defaults["bboxes_info_filename"]=proj_defaults["raw_data_folder"]/"bboxes_voxels_info"
#     proj_defaults["resampled_dataset_properties_filename"] = proj_defaults["stage0_folder"]/"resampled_dataset_properties"
#     proj_defaults["experiments_folder"] = proj_defaults["ssd"]/("Dropbox/code/fran/experiments/"+proj_defaults["project_title"])
#     proj_defaults["neptune_folder"]=proj_defaults["experiments_folder"].parent/".neptune" 
#     proj_defaults["configuration_filename"]= proj_defaults["experiments_folder"]/("metadata/experiment_configs.xlsx")
#     proj_defaults["validation_folds_filename"]= proj_defaults["experiments_folder"]/("metadata/validation_folds.json")
#     proj_defaults["mask_labels"] = load_dict(proj_defaults["raw_data_folder"]/"mask_labels")[:-1]
#     proj_defaults["label_priority"] = load_dict(proj_defaults["raw_data_folder"]/"mask_labels")[-1]["label_priority"]
#     
#
#     proj_defaults["predictions_folder"] = proj_defaults["storage"]/("datasets/predictions")/proj_defaults["project_title"]
#     proj_defaults["checkpoints_folder"] = Path("/s/checkpoints")/proj_defaults["project_title"]
#     create_project_folder_tree(proj_defaults)
#     # make folders if needed
#     return SimpleNamespace(**proj_defaults)

def load_proj_defaults(common_paths_filename:str, project_title: str):
    with open(common_paths_filename,'r') as file:
        common_paths_ = yaml.safe_load(file)
        common_paths={}
        for ke,val in common_paths_.items():
            common_paths[ke]= Path(val)

    proj_defaults = {'project_title':project_title}
    proj_defaults["raw_data_folder"] =common_paths['slow_storage_folder']/("raw_data/"+proj_defaults["project_title"])

    proj_defaults["preprocessing_output_folder"]= common_paths['fast_storage_folder']/proj_defaults["project_title"]

    proj_defaults["stage0_folder"]= proj_defaults["preprocessing_output_folder"]/"stage0_resampled"
    proj_defaults["stage1_folder"]= proj_defaults["preprocessing_output_folder"]/"stage1_lowres"
    proj_defaults["stage2_folder"]= proj_defaults["preprocessing_output_folder"]/"stage2_patches"
    # files etc
    proj_defaults["raw_dataset_properties_filename"]=proj_defaults["raw_data_folder"]/"raw_dataset_properties"
    proj_defaults["global_properties_filename"]=proj_defaults["raw_data_folder"]/"global_properties"
    proj_defaults["bboxes_voxels_info_filename"]=proj_defaults["raw_data_folder"]/"bboxes_voxels_info"
    proj_defaults["resampled_dataset_properties_filename"] = proj_defaults["stage0_folder"]/"resampled_dataset_properties"
    proj_defaults["experiments_folder"] = common_paths['experiments_folder']/(proj_defaults["project_title"])
    proj_defaults["neptune_folder"]=common_paths['neptune_folder']
    proj_defaults["configuration_filename"]= proj_defaults["experiments_folder"]/("metadata/experiment_configs.xlsx")
    proj_defaults["validation_folds_filename"]= proj_defaults["experiments_folder"]/("metadata/validation_folds.json")
    proj_defaults["mask_labels"] = load_dict(proj_defaults["raw_data_folder"]/"mask_labels")[:-1]
    proj_defaults["label_priority"] = load_dict(proj_defaults["raw_data_folder"]/"mask_labels")[-1]["label_priority"]

    proj_defaults["predictions_folder"] = common_paths['slow_storage_folder']/("predictions/"+proj_defaults["project_title"])
    proj_defaults["checkpoints_parent_folder"] = common_paths['checkpoints_parent_folder']/proj_defaults["project_title"]
    create_project_folder_tree(proj_defaults)
    # make folders if needed
    return SimpleNamespace(**proj_defaults)

def create_project_folder_tree(proj_defaults):
    # confirm folders exist
    for key,value in proj_defaults.items(): 
        if isinstance(value,Path) and "folder" in key:
            maybe_makedirs(value)
    # sub-folders needed later
    additional_folders =[
    (proj_defaults['stage0_folder'])/('volumes'),
    # (proj_defaults['stage2_folder'])/('volumes'),
    # proj_defaults['stage2_folder']/("cropped/images_nii/masks"),
    # proj_defaults['stage2_folder']/("cropped/images_nii/images"),
    ]
    for folder in additional_folders:
        maybe_makedirs(folder)

# %%
if __name__ == "__main__":
    import yaml
    project_title="kits21"
    common_paths_filename= "nbs/config.yaml"
    P = Project(project_title="lits"); proj_defaults= P.proj_summary

# %%
    di = load_dict(proj_defaults.raw_dataset_properties_filename)
    

