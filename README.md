This repository combines the powers of **F**astai, **Ra**ytune, and **N**eptune to manage and automate medical image segmentations tasks. Currently, I have developed this specifically using the KiTS21 challenge. I welcome others to test this library on the same challenge initially (or on others if you are feeling adventurous and don't mind debugging). Using the random search implemented in the raytune library, I have achieved very good kidney tumour segmentation results.

For an introduction to this codebase, please visit my post on the fastai forum https://forums.fast.ai/t/code-collaboration-opportunity-for-ct-radiology-ai-projects-kits-lits/98992

## Installation
This library uses NVIDIA GPU. It does not support training on CPU.

Simply clone this library and start using it.
Make sure you have installed and set up:
- fastai (>=2.7.10)
 - raytune from: https://anaconda.org/conda-forge/ray-tune
 - created an account on neptune-ai at https://neptune.ai/, and 
 - downloaded a dataset (e.g., LITS).

\
To quickly install all pre-requisites in a new environment, I run the commands in file `conda_oneliners.txt` ***(make sure to create a new environment beforehand!)***


## 1. Setting common variables
Once you have installed neptune client, and created a new empty project, open the project and you will find instructions to initialize a project like so:
```
run = neptune.init_run(
    project="{workspacename}/{projectname}",
    api_token="abc ......")  # your credentials
```
Note the workspace name and api token. Then open `nbs/config-test.yaml`. The file provided here has my directory structure which you may emulate if you like. Among other paths, you will need to assign a `{fast_storage}` (used by the library for DL) and a `{slow_storage}`folder (where you download your dataset nifty files). After setting folder paths, you need to store both the api token and project workspacename (NOT project name) inside `config.yaml` provided.

Finally, add path to your `config.yaml` file in `~/.bashrc` as:

```

export FRAN_COMMON_PATHS={PATH-TO-config.yaml}

```
*Note: In this instruction, names inside curly-braces are variable names. You can set them as any word you like.* Names without curly braces are fixed and must be the same in your schema.

## 2. Dataset organization
In each input path, data (nifti or nrrd format) must be organised in sub-folders 'images' and 'masks'



```
   └── kits21
        ├── images
        │   ├── kits21_00000.nii.gz
        │   ├── ...
        │   ├── ...
        │   └── kits21_00299.nii.gz
        └── masks
           ├── kits21_00000.nii.gz
           ├── ...
           ├── ...
           └── kits21_00299.nii.gz
```
In the figure above, I have used `kits_21` as the `{project title}`. You can have any name as long as it follows the [naming rules](#naming-rules) given below
As shown above, mask and image files of a given case will have identical names, e.g., `kits21_00299.nii.gz`, but will be under separate folders (`images` and `masks`). 
### Naming rules 
- Each case file name has the following parts: `{project_title}_{case_id}.ext`
- project_title should be in small letters. 
- case_id should be unique digits +/- alphabets only, e.g., '00000','00021' or 'b0000', 'b0021'.
- Do **NOT** use dash inside folder names for now, hyphens are acceptable. For example,\
`~/fran-test/`*(incorrect)* \
`~/fran_test/`*(correct)*


*Note: There is no validation data folder. The library will create splits by itself keeps track of them.*

*Note: Having separate `{slow_storage}` and `{fast_storage}` is not a requirement. Both folders can be on the same drive. I recommend using SSD for `{fast_storage}` to speed up learning.*
## 3. Project
Run script `fran/runs/project_init.py` to initialize a project. It requires two arguments: -t (project title) and -i (input folders: 1 or more, containing datasets).\

```
python project_init.py -t {project_title} -i {dataset folder 1} {dataset_folder 2} {dataset_folder 3} ...
```

For example, in the fran/runs folder, you may initialise a project called 'lits' and enter:
```
python project_init.py -t lits -i /s/datasets_bkp/drli /s/datasets_bkp/lits_segs_improved/ 
```
### LM Groups
All datasources within a single lm_group are indexed continuously. Subsequent lm_groups are indexed starting after the highest index of the preceding lm_group.

I have provided 2 folders as datasets for  this project in the example above. Typically, most projects will be based on a single datafolder but this provides flexibility to add more data to a project as it becomes available. After the project is initialised, look inside the project folder. You will find a mask_labels.json file. This file sets rules for postprocessing each label after running predictions. 

## 4.Analyze resample
### 1.Generate fixed spacings

### 2. Generate LBD (Labelbounded) dataset
This creates variable shape volumes cropped to designated label.
There are two ways to define the label:
a) Use imported labels
b) Use own labelmap label

#### 1. Steps:
 - LabelBoundedDataGenerator





```
python analyze_resample.py -t {project_title}
```
## 5.Train
As above, enter at the minimum:
```
python train.py -t {project_title}
```
However, you will likely run into problems if your data structure does not meet values stored in training config spreadsheet 

## 6.Inference
In the fran/run folder, ensemble.sh is a shell script. Please edit it with following flags:\
```
-e {run_names of your preferred ensemble, each name separated by a space}.
-i {input folder of test images}
```
(Experimental Note): If you have multi-gpu, replace `ensemble_singlegpu.py` with `ensemble.py` inside ensemble.sh. I have developed it for 2-GPUs.


## 8.Scoring
Once you have predictions and masks ready, examine the contents of file `inference/scoring.py`. The function `compute_dice_fran` wraps dice function from the monai library. It accepts groundtruth mask and prediction in SimpleITK Image format.

## 7.Training on HPC Cluster
See the file hpc.yaml inside templates folder and alter it to your configuration. You will need to create an environment variable HPC_SETTINGS which points to the location of the hpc.yaml file on your system, e.g.:


```
export HPC_SETTINGS="/s/fran_storage/hpc.yaml"
```
This is because inference functions will not natively have the rights to download models stored on the HPC cluster. This file will store the user access settings to allow inference to retrieve  data stored on the cluster.

# Glossary of terms

|Name      |Abbreviation |Info|
|----------|-------------|----|
|Hounsfield Unit| HU |https://radiopaedia.org/articles/hounsfield-unit?lang=gb|
|Voxel||https://radiopaedia.org/articles/voxel?lang=gb|


# Acknowledegments:
Sources of inspiration and code snippets:\
Fastai programming paradigm is at the core of this library.\
I have also drawn lots of inspiration from [nnUNet](https://github.com/MIC-DKFZ/nnUNet) in structuring the pipeline.

