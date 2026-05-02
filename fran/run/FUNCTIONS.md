# FRAN Run Script Inventory

This file is the runnable-entrypoint inventory for `fran/fran/run/`.

- Scope: FRAN-owned scripts and Python CLIs under this folder.
- Goal: help humans and agents find the right maintained entrypoint quickly.
- Preference: use Python modules or FRAN-owned shell wrappers here before adding duplicate wrappers elsewhere.

## Folder: `project/`

- `project/project.sh -t <project_title> -m <mnemonic> -ds <datasource> [<datasource> ...]`
  - Canonical FRAN shell entrypoint for project creation.
  - Thin wrapper over `project_init.py`.

- `project/project_init.py -t <project_title> -m <mnemonic> -ds <datasource> [<datasource> ...]`
  - Creates a FRAN project, initialises folders/database, and registers datasources.

- `project/project_delete.sh <project_title> [project_title ...]`
  - Canonical FRAN shell entrypoint for project deletion.
  - Thin wrapper over `project_delete.py`.

- `project/project_delete.py <project_title> [project_title ...]`
  - Deletes FRAN project folders and associated project state.

- `project/projects_list.sh`
  - Lists project directory names from `$FRAN_CONF/config.yaml -> projects_folder`.

- `project/project_status.py [projects ...] [-n <num_processes>]`
  - Loads project configs and reports per-plan preprocessing status.
  - Uses `FolderNames(...)` to confirm source/final plan folders exist.

- `project/datasource_init.py <folder> <mnemonic> [-n <num_processes>]`
  - Initializes a `Datasource` from a folder containing `images/` and `lms/`.
  - Calls `Datasource.process(...)`.

- `project/resolve_plan_folder.py <project_title> <plan_num> [--key <folder_key>]`
  - Resolves plan folder paths via `Project + ConfigMaker + FolderNames`.
  - Prints full JSON mapping by default.
  - `--key` prints a single resolved folder path.
  - Valid keys:
    - `data_folder_source`
    - `data_folder_lbd`
    - `data_folder_whole`
    - `data_folder_pbd`
    - `data_folder_sourcepbd`
    - `data_folder_rbd`

- `project/proj_to_analyze.py`
  - Project-analysis helper script.
  - Treat as exploratory/diagnostic, not a stable public CLI.

- `project/tmp.py`
  - Scratch script.
  - Not a stable entrypoint.

## Folder: `preproc/`

- `preproc/analyze_resample.py -t <project_title> -p <plan_num> [-n <num_processes>] [-o]`
  - Main FRAN preprocessing entrypoint.
  - Resolves plan config, runs fixed-spacing preprocessing, then plan-mode-specific dataset generation:
    - `lbd`
    - `rbd`
    - `pbd` / `patch`
    - `whole`

- `preproc/preproc.sh`
  - Local FRAN convenience launcher for `analyze_resample.py` through `block_suspend.py`.
  - Current file is a user-specific example invocation, not a general-purpose wrapper.

- `preproc/nifti_to_pt.py`
  - NIfTI-to-torch preprocessing helper.
  - Use when working specifically on raw-to-PT conversion flows.

- `preproc/rawdata_organise.py`
  - Raw-data organisation helper.
  - Use for data staging/cleanup tasks rather than plan execution.

## Folder: `dataregistry/`

- `dataregistry/update_datasources.py [dataset_name ...] [-n <num_processes>] [--dry-run] [--return-voxels]`
  - Reads `$FRAN_CONF/datasets.yaml`.
  - Initializes or updates datasource `fg_voxels.h5` files for the named datasets.
  - If no dataset names are supplied, processes every dataset in the config.

## Folder: `inference/`

- `inference/predict.py --title <project_title> --run-w <run_w> --run-p <run_p> --images-folder <folder> --gpus <gpu ...> --localiser-labels <label ...>`
  - Main CLI for prediction over a folder of images.
  - Supports single-GPU and Ray-based multi-GPU execution.

- `inference/ensemble.py`
  - Multi-run ensemble inference entrypoint.
  - Use when combining multiple FRAN runs into one inference pass.

- `inference/ensemble_singlegpu.py`
  - Single-GPU ensemble inference entrypoint.

- `inference/inference.py`
  - Higher-level inference helper entrypoint.
  - Use when calling FRAN inference flows programmatically or through a thin CLI.

- `inference/main.py`
  - Small bootstrap/demo entrypoint for inference flows.
  - Not the preferred main CLI when `predict.py` or `ensemble.py` already fits.

- `inference/localiser_window_demo.py`
  - Demo/inspection tool for the localiser windowing path.

- `inference/ensemble.sh`
- `inference/ensemble_multi.sh`
  - Example shell launchers for ensemble inference.
  - Treat as convenience examples rather than canonical APIs.

## Folder: `training/`

- `training/train.py`
  - Main FRAN training entrypoint.

- `training/train_retry.py`
  - Training entrypoint with retry/relaunch logic.

- `training/train_retry.sh`
- `training/p_train.sh`
- `training/p_train_suspend_only.sh`
- `training/toy_train.sh`
- `training/lungs.sh`
- `training/nodes.sh`
- `training/wb_train.sh`
  - Convenience shell launchers for specific local training workflows.
  - Most are user/task-specific and often contain hard-coded example arguments.

- `training/download_case_recorder_tables.sh`
  - Fetches W&B case recorder tables for a configured run.

## Folder: `profiling/`

- `profiling/bench_dual_ssd.py`
  - Storage / IO profiling helper.

- `profiling/p_train_perf.py`
  - Training performance profiling entrypoint.

- `profiling/profile_live.py`
  - Live GPU/runtime profiling helper.

- `profiling/profile_python_hotspots.py`
  - Python hotspot profiler.

- `profiling/profile_train_stacks.py`
  - Training-stack profiler with optional trace export.

- `profiling/profile_transform_times.py`
  - Profiles dataloader / transform timing.

- `profiling/*.sh`
  - Thin convenience launchers for the profiling CLIs above.

## Folder: `misc/`

- `misc/block_suspend.py`
  - Wrapper that runs another command while controlling suspend behavior.
  - Used by some shell launchers around long-running training/preproc tasks.

- `misc/view_image.py`
  - Python image viewer helper. Typically you will pass it an image and a labelimage from images/ and lms/ images of a folder.

- `misc/imageviewer.sh <image> [label]`
  - Thin wrapper over `view_image.py`.

## Notes

- `__pycache__/` entries are implementation artifacts, not CLIs.
- Prefer FRAN-owned entrypoints in this tree over duplicate wrappers in other repos.
- When adding a new runnable here, update this file so agents can treat it as the run-registry source of truth.
