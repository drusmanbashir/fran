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




## 1. Configuration  (excel spreadsheets)

All dataset creation and (some) training blueprints are in excel spreadsheets and thats where we start. Glossary:
### A. Plans


## 1. Setting environment variables
Once you have installed neptune client, and created a new empty project, open the project and you will find instructions to initialize a project like so:
```
run = neptune.init_run( project="{workspacename}/{projectname}",
    api_token="abc ......")  # your credentials
```
Note the workspace name and api token. Then open `nbs/config-test.yaml`. The file provided here has my directory structure which you may emulate if you like. Among other paths, you will need to assign a `{fast_storage}` (used by the library for DL) and a `{slow_storage}`folder (where you download your dataset nifty files). After setting folder paths, you need to store both the api token and project workspacename (NOT project name) inside `config.yaml` provided.

Finally, add path to your `config.yaml` file in `~/.bashrc` as:

```

export FRAN_COMMON_PATHS={PATH-TO-config.yaml}

```
*Note: In this instruction, names inside curly-braces are variable names. You can set them as any word you like.* Names without curly braces are fixed and must be the same in your schema.

## 2. Datasource

The Datasource class is the core component for managing individual medical imaging datasets in FRAN. Each datasource represents a folder containing paired image and label files, handling data validation, preprocessing, and storage of processed voxel statistics.

### Folder Structure Requirements

In each input path, data (nifti or nrrd format) must be organised in sub-folders 'images' and 'lms'



```
   └── kits21
        ├── images
        │   ├── kits21_00000.nii.gz
        │   ├── ...
        │   ├── ...
        │   └── kits21_00299.nii.gz
        └── lms

           ├── kits21_00000.nii.gz
           ├── ...
           ├── ...
           └── kits21_00299.nii.gz
```
In the figure above, I have used `kits_21` as the `{project title}`. You can have any name as long as it follows the [naming rules](#naming-rules) given below
As shown above, mask and image files of a given case will have identical names, e.g., `kits21_00299.nii.gz`, but will be under separate folders (`images` and `lms`). 

### Key Features

- **Automatic Validation**: Verifies matching image-label pairs and equal file counts
- **Incremental Processing**: Tracks processed cases in HDF5 files, skipping already processed data
- **Metadata Storage**: Stores foreground voxel statistics (mean, std, min, max, labels) for each case
- **Label Management**: Supports relabeling operations for standardizing label schemes
- **Error Handling**: Detects duplicate case IDs and provides tools for resolution

### Naming rules 
- Each case file name has the following parts: `{project_title}_{case_id}.ext`
- project_title should be in small letters. 
- case_id should be unique digits +/- alphabets only, e.g., '00000','00021' or 'b0000', 'b0021'.
- Do **NOT** use dash inside folder names for now, hyphens are acceptable. For example,\
`~/fran-test/`*(incorrect)* \
`~/fran_test/`*(correct)*


*Note: There is no validation data folder. The library will create splits by itself keeps track of them.*

*Note: Having separate `{slow_storage}` and `{fast_storage}` is not a requirement. Both folders can be on the same drive. I recommend using SSD for `{fast_storage}` to speed up learning.*

## 3. Project Creation

The project creation workflow is the foundation of the FRAN framework and involves several key steps to set up a complete machine learning project for medical image segmentation. This process creates the necessary folder structure, manages data sources, and prepares configurations for training.

### Core Components

The project creation system consists of several key classes:

- **Project**: Main project management class that handles folder structure, database operations, and data organization
- **Datasource**: Manages individual datasets with integrity checking and preprocessing capabilities  
- **ConfigMaker**: Handles configuration parsing from Excel files and parameter setup

### Step-by-Step Project Creation Process

#### 1. Initialize Project Object
```python
from fran.managers.project import Project

P = Project(project_title="your_project_name")
```

#### 2. Create Project Structure
```python
P.create(mnemonic='project_mnemonic')
```

This step:
- Creates the complete folder hierarchy for the project
- Initializes the SQLite database (`cases.db`) to track all cases
- Sets up global properties for the project
- Creates necessary subfolders for raw data, preprocessed data, predictions, etc.

#### 3. Add Data Sources
```python
# Add predefined data sources
P.add_data([DS.dataset1, DS.dataset2], test=False)

# Or add custom data sources
datasources = [
    {'ds': 'custom_dataset', 'folder': '/path/to/dataset', 'alias': None}
]
P.add_data(datasources)
```

The `add_data` method:
- Validates data integrity (matching image/mask pairs)
- Creates symbolic links in the raw data folder
- Populates the project database with case information
- Handles duplicate detection and filtering
- Registers datasources in global properties
- In a plan,  some datasources may be dropped to change how training occurs, but folders are populated by the full compliment of data

#### 4. Configure Label Groups
```python
# Single group (default)
P.set_lm_groups()  # All datasets in one group

# Multiple groups for different label schemes
P.set_lm_groups([['dataset1', 'dataset2'], ['dataset3']])
```

Label groups (`lm_groups`) organize datasets with different labeling schemes. Each group maintains separate label indices, allowing combination of datasets with overlapping or conflicting label definitions.

#### 5. Generate Training/Validation Folds
```python
P.maybe_store_projectwide_properties()
```

This critical step:
- Creates 5-fold cross-validation splits (80:20 train/validation)
- Computes global dataset properties (mean, std, intensity ranges)
- Collates all unique labels across datasets
- Stores foreground voxel statistics for each case

#### 6. Create Configuration Plans
```python
from fran.configs.parser import ConfigMaker

conf = ConfigMaker(P, raytune=False, configuration_filename=None).config
plans = conf['plan1']  # Select desired plan
P.add_plan(plans)
```

Configuration plans define:
- Target spacing and dimensions
- Processing modes (lbd, patch, whole, etc.)
- Data augmentation parameters
- Model architecture settings
- Training hyperparameters

##### a. Imported labelmaps
If you add imported labelmaps to a plan, you have to include a remapping cell, leave it empty to give logic for remapping in it. It can be a dict, or a TSL class attribute if imported folder is TSL predictions. TSL class attributes simplify remapping and generate the dict for you.

##### b. Expand_by
This setting is used mainly by LBD (labelbounded) datasets. It is ignored by:
- Source dataset

##### c. use_fg_indices. If set to false, training will not use the fg_indices files (hopefully saves some ram)


A complex element is output folder naming. This is managed in a project database table called ```master_plans``` 


You must include imported_folder, remapping,
### Database Schema

The project uses SQLite to track all cases with the following schema:

```sql
CREATE TABLE datasources (
    ds TEXT,              -- Dataset name
    alias TEXT,           -- Dataset alias for filename matching (this is used if datasource title does not match the first token in case names e.g. litstmp)
    case_id TEXT,         -- Unique case identifier
    image TEXT,           -- Original image path
    lm TEXT,              -- Original labelmap path
    img_symlink TEXT,     -- Symlink path for image
    lm_symlink TEXT,      -- Symlink path for labelmap
    fold INTEGER,         -- Cross-validation fold assignment
    test BOOLEAN          -- Test set flag
)
```

### Folder Structure Created

```
project_folder/
├── cases.db                    # SQLite database
├── global_properties.json     # Project-wide settings
├── experiment_configs.xlsx    # Configuration plans
└── validation_folds.json      # Cross-validation splits

cold_storage/
├── raw_data/project_name/      # Symbolic links to original data
│   ├── images/
│   └── lms/
└── preprocessed/               # Generated preprocessed datasets
    ├── fixed_spacing/
    ├── lbd/                   # Label-bounded datasets
    ├── patches/               # Patch datasets
    └── whole_images/          # Whole image datasets

rapid_access/project_name/      # Fast storage for training
├── cache/                      # Cached data
├── patches/                   # Patch datasets
└── lbd/                       # Label-bounded datasets
```

### Advanced Features

#### Data Source Management
- **Integrity Checking**: Verifies matching image/mask pairs
- **Duplicate Detection**: Handles cases with identical names across datasets
- **Incremental Addition**: Add new data to existing projects without rebuilding
- **Alias Support**: Handle datasets with non-standard naming conventions

#### Configuration System
- **Excel-based Configuration**: Define complex training plans in spreadsheets
- **Parameter Inheritance**: Plans can inherit from other plans using `source_plan`
- **Ray Tune Integration**: Automatic hyperparameter search space definition
- **Dynamic Parameter Resolution**: Automatic calculation of output channels, dataset statistics

#### Global Properties Management
- **Intensity Statistics**: Foreground mean, std, min, max across all datasets
- **Label Consolidation**: Unified label mapping across multiple datasets
- **Spacing Analysis**: Dataset spacing statistics and recommendations
- **Clipping Ranges**: Optimal intensity clipping ranges for preprocessing

### Example: Complete Project Setup

```python
# 1. Create project
P = Project(project_title="lung_segmentation")
P.create(mnemonic='lungs')

# 2. Add multiple datasets
P.add_data([DS.task6, DS.lidc2])

# 3. Set up label groups
P.set_lm_groups([['task6'], ['lidc2']])  # Different label schemes

# 4. Generate folds and compute properties
P.maybe_store_projectwide_properties()

# 5. Load configuration and add plan
conf = ConfigMaker(P, raytune=False).config
P.add_plan(conf['plan1'])

# 6. Verify setup
print(f"Project has {len(P)} cases")
print(f"Datasets: {P.datasources}")
print(f"Has folds: {P.has_folds}")
```

This comprehensive project creation system ensures reproducible, well-organized machine learning experiments with proper data management, configuration tracking, and validation protocols.

## 4. Project

### 1. Excel spreadsheet
Make sure to add a sheet labelled 'plan1' at least.
Steps are :
```

    P= Project(project_title="nodes") <-- any title

    P.create(mnemonic='nodes')
    conf = ConfigMaker(
        P, raytune=False, configuration_filename=None

    ).config

```

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

## 5. Analyze and Resample

The analyze_resample.py script handles data preprocessing, including dataset analysis, resampling, and various data generation modes. The process is controlled through a plan configuration that specifies parameters for each step.


#### i) Remapping: 
Remapping patterns can be dicts or 2-tuples mapping srce to dest labels. This remapping bakes the new labels into the dataset, reducing training workload(c.f. `src_dest_labels`). Remapping can be a dict or a list of 2-lists (```[src_labels],[dest_labels]```
Remapping can be done at multiple levels:
- remapping_imported: for example: ```{1:0,2:2:3:3}``` or ```[1,2,3], [0,2,3]```. However if the imported dataset has additional labels, e.g., ```4, 5,``` they will be remapped to ```0``` if they are not explicitly mentioned in the remapping scheme.
- remapping_source
- remapping_lbd
- remapping_train: This is also a remapping but unlike `Remapping` above, this remapping happens on the fly, during training.\
It is a list of 2 lists, list 1 is the source mappings, list 2 dest mappings. Aligns with monai transform MapLabelValue

Folder naming schemes are separate and overlapping for: 
#### A.  Source datasets
This follows ```sze_dim1_dim2_{remapping_code}``` Remapping codes are managed in ```fran/utils/suffixy_registry```. 
### Data Types
The system supports multiple data types:
* **source**: Original raw data
* **lbd**: Label-bounded data (cropped around regions of interest)
* **pbd**: Patient-bounded data
* **patch**: Extracted patches from the data
* **whole**: Complete volumes at specified resolution

### Configuration
1. Create a plan in the configuration Excel sheet with the following key parameters:
   - `spacing`: Target voxel spacing (e.g., [0.8, 0.8, 1.5])
   - `mode`: Processing mode ('lbd', 'patch', etc.)
   - `patch_overlap`: For patch mode, overlap between patches (e.g., 0.25)
   - `expand_by`: Expansion margin around regions of interest
   - `patch_dim0`, `patch_dim1`: Patch dimensions if using patch mode
   - `imported_folder`: Optional path to imported labels
   - `imported_labels`: Label configuration for imported data
   - `merge_imported_labels`: Whether to merge multiple imported labels
   - `src_dest_labels` : #INCOMPLETE
   - `source_plan`: Some plans are derived from others, i.e., `whole`

### Processing Steps

1. **Dataset Analysis**
   ```bash
   python analyze_resample.py -t {project_title}
   ```
   This step:
   - Verifies dataset integrity (matching sizes and spacings)
   - Computes global properties (mean, std, etc.)
   - Stores metadata for subsequent processing

2. **Fixed Spacing Generation**
   - Resamples data to target spacing specified in plan
   - Handles both images and masks
   - Preserves metadata and image properties

3. **Label-Bounded Dataset Generation**
   Two approaches available:
   a) Using imported labels:
      - Specify `imported_folder` and `imported_labels` in plan
      - Supports label remapping and merging
   b) Using own labelmap:
      - Uses project's existing label definitions
      - Controlled by `lm_groups` configuration

4. **Patch Generation**
   For high-resolution analysis:
   - Extracts patches according to `patch_dim0`, `patch_dim1`
   - Supports overlap between patches (`patch_overlap`)
   - Can focus on foreground/background regions
   - Generates bounding box data automatically

### Key Features
- Multi-processing support for faster processing
- Automatic handling of image/mask alignment
- Flexible data augmentation options
- Support for different label mapping schemes
- Progress tracking and error handling

### Example Usage
```bash
# Basic analysis and resampling
python analyze_resample.py -t project_name -p plan1

# With specific options
python analyze_resample.py -t project_name \
  -n 8 \                     # number of processes
  -p patch_size \            # patch dimensions
  -s spacing \               # target spacing
  -po 0.25 \                # patch overlap
  --half_precision \         # use FP16
  --debug                    # enable debug output
```

Note: Before running patch generation, ensure you have generated the required fixed spacing dataset. The system will automatically track dependencies and maintain data consistency.






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
See the file hpc.yaml inside fran.templates folder and alter it to your configuration. You will need to create an environment variable HPC_SETTINGS which points to the location of the hpc.yaml file on your system, e.g.:


```
export HPC_SETTINGS="/s/fran_storage/hpc.yaml"
```
This is because inference functions will not natively have the rights to download models stored on the HPC cluster. This file will store the user access settings to allow inference to retrieve  data stored on the cluster.

## 8. Worflow examples

### A) Creating a patient-bounded dataset for training

**1. Create project.**\
**2. Create a plan, e.g.,**

| **Variable**            | **Value**                                     |
|-------------------------|-----------------------------------------------|
| `var_name`              | `manual_value`                                |
| `datasources`           | `nodesthick, nodes`                           |
| `lm_groups`             |                                               |
| `spacing`               | `0.8, 0.8, 1.5`                               |
| `fg_indices_exclude`    |                                               |
| `mode`                  | `lbd`                                         |
| `patch_overlap`         | `0.25`                                        |
| `expand_by`             | `0`                                           |
| `samples_per_file`      | `2`                                           |
| `imported_folder`       | `/s/fran_storage/predictions/totalseg/LITS-1088` |
| `imported_labels`       | `TSL.all`                                     |
| `merge_imported_labels` | `FALSE`                                       |
| `patch_dim0`            | `128`                                         |
| `patch_dim1`            | `96`                                          |

*Note: impoarted_labels is set a s TSL.all. This is because LITS-1088 is an 8-label (localiser) model. By selecting TSL.all (i.e., no remapping done on labelmaps) is the correct code. Alternatively, a list of labels, i.e., [0,1,2,3,4,5,6,7,8] ought to (not tested) work the same.*

**3. Train.**\
# Glossary of terms

|Name      |Abbreviation |Info|
|----------|-------------|----|
|Hounsfield Unit| HU |https://radiopaedia.org/articles/hounsfield-unit?lang=gb|
|Voxel||https://radiopaedia.org/articles/voxel?lang=gb|


# Acknowledegments:
Sources of inspiration and code snippets:\
Fastai programming paradigm is at the core of this library.\
I have also drawn lots of inspiration from [nnUNet](https://github.com/MIC-DKFZ/nnUNet) in structuring the pipeline.
Good


## 9. Troubleshooting

### HPC
1. n_processes in analyze_resample should be kept low to reflect the number of cpus allocated, e.g., 2 to 4.
