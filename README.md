This repository combines the powers of **F**astai, **Ra**ytune, and **N**eptune to manage and automate medical image segmentations tasks. Currently, I have developed this specifically using the KiTS21 challenge. I welcome others to test this library on the same challenge initially (or on others if you are feeling adventurous and don't mind debugging). Using the random search implemented in the raytune library, I have achieved very good kidney tumour segmentation results.

For an introduction to this codebase, please visit my post on the fastai forum https://forums.fast.ai/t/code-collaboration-opportunity-for-ct-radiology-ai-projects-kits-lits/98992

## Getting Started
This library uses NVIDIA GPU. It does not support training on CPU.

Simply clone this library and start using it.
Make sure you have installed and set up:
- fastai (>=2.7.10)
 - raytune from: https://anaconda.org/conda-forge/ray-tune
 - created an account on neptune-ai at https://neptune.ai/, and 
 - downloaded the KiTS21 challenge from https://kits21.kits-challenge.org/

\
To quickly install pre-requisites, you can run
```
pip install nnunet
conda install -c conda-forge ipython ipywidgets pandas matplotlib medpy numpy openpyxl pillow pygments jupyterlab pyyaml scikit-image scipy SimpleITK connected-components-3d timm torchio tqdm einops monai ipdb gputil ray-tune neptune-client ipympl lxml
#note some of these will downgrade your pytorch copy
```


## Setting up Neptune
Once you have installed neptune client, and created a new empty project, open the project and you will find instructions to initiate like so:
```
run = neptune.init_run(
    project="{workspacename}/{projectname}",
    api_token="abc ......")  # your credentials
```
You need to store both the api token and project workspacename (NOT project name) inside experiements/config.json provided. Make sure every project you create under neptune has a projectname which matches the name you give it when you initialise it inside fran. For example, in neptune my lits project looks like this: project = 'drusmanbashir/lits' where drusmanbashir is the workspace-name and lits is the project_title
## Setting paths

See the file `nbs/config.yaml` to set paths. The file provided here has my directory structure which you may emulate if you like. Among other paths, you will need to assign a `{fast_storage}` (used by the library for DL) and a `{slow_storage}`folder (where you download your dataset nifty files). 

*Note: In this instruction, names inside curly-braces are variable names. You can set them as any word you like.* Names without curly braces are fixed and must be the same in your schema.
## Organizing dataset folders

The dataset is downloaded in nifti format. For the KiTS21 challenge, you will have to re-organise the data since the challenge organizers have stored each case in a separate folder as documented https://github.com/neheller/kits21. Follow the structure below to organise your files (which may need renaming) and folders. 


```
{slow_storage}
└── raw_data
    └── kits21
        ├── imagesTr
        │   ├── kits21_00000.nii.gz
        │   ├── ...
        │   ├── ...
        │   └── kits21_00299.nii.gz
        ├── masksTr
        │   ├── kits21_00000.nii.gz
        |   ├── ...
        │   ├── ...
        │   └── kits21_00299.nii.gz
        └── mask_labels.json
```

#### Step 1: Create folder tree
As shown, create the folder tree under the `{slow_storage}` location of your choice: `raw_data->{project_title}->imagesTr, masksTr`.

In the figure above, I have used `kits_21` as the `{project title}`. You can have any name as long as it follows the [naming rules](#naming-rules) given below
#### Step 2: Create mask_labels.json
Create a `mask_labels.json` file inside the `{slow_storage}/raw_data/{project_title}` folder (`kits21` in the figure above). This file provides label priorites when segmenting and post-processing dusting thresholds.\
I have provided a sample `mask_labels.json` file inside the `experiments/kits21` folder of this repository. Simply place it inside your `raw_data/{project_title}`folder.

#### Step 3: Organise image and mask files under folders named `imagesTr` and `masksTr`
As shown above, mask and image files of a given case will have identical names, e.g., `kits21_00299.nii.gz`, but will be under separate folders (`imagesTr` and `masksTr`). 

##### Naming rules 
- Each case file name has the following parts: `{project_title}_{case_id}.ext`
- project_title should be in small letters. 
- case_id should be unique digits +/- alphabets only, e.g., '00000','00021' or 'b0000', 'b0021'.
- Do **NOT** use dash inside folder names for now, e.g.,  `~/fran-test/`*(incorrect)*, `~/fran_test/`*(correct)*


*Note: There is no validation data folder. The library will create splits by itself keeps track of them.*

*Note: Having separate `{slow_storage}` and `{fast_storage}` is not a requirement. Both folders can be on the same drive. I recommend using SSD for `{fast_storage}` to speed up learning.*
## Using the library
Simply head over to the `nbs/` subfolder and use the jupyter notebooks there to use the various features of the library. Please feel free to post comments and feedback as you encounter problems!



# Glossary of terms


|Name      |Abbreviation |Info|
|----------|-------------|----|
|Hounsfield Unit| HU |https://radiopaedia.org/articles/hounsfield-unit?lang=gb|
|Voxel||https://radiopaedia.org/articles/voxel?lang=gb|


# Acknowledegments:
Sources of inspiration and code snippets:\
Fastai programming paradigm is at the core of this library.\
I have also drawn lots of inspiration from [nnUNet](https://github.com/MIC-DKFZ/nnUNet) in structuring the pipeline.\
[Torchio](https://torchio.readthedocs.io/) code is written very well for creating patches, and I have used it.
