# nnUNet v2 Migration

## Current State

FRAN now has a first `nnUNet v2` integration point in [fran/architectures/create_network.py](/home/ub/code/fran/fran/architectures/create_network.py).

What changed:

- Added imports for:
  - `dynamic_network_architectures.architectures.unet.ResidualEncoderUNet`
  - `convert_dim_to_conv_op`
  - `get_matching_instancenorm`
  - `nnunetv2.experiment_planning.experiment_planners.network_topology.get_pool_and_conv_props`
- Added a new model creation function:
  - `create_model_from_conf_nnunetv2_resenc_l(...)`
- Added a new dispatch path in `create_model_from_conf(...)`
  - any `model_params["arch"]` equal to `resunet` case-insensitively now uses the new function

Behavior of the new `resunet` path:

- Uses FRAN's existing `plan["spacing"]` and `plan["patch_size"]`
- Builds a real `nnUNet v2` `ResidualEncoderUNet`
- Derives topology using `get_pool_and_conv_props(...)`
- Hard-codes a ResEnc-L-like policy:
  - base features `32`
  - max features `320` for 3D, `512` for 2D
  - encoder blocks `(1, 3, 4, 6, 6, 6, ...)`
  - decoder convs `(1, 1, 1, ...)`
  - InstanceNorm
  - LeakyReLU
  - no dropout

## Why This Is Currently "L-like"

The current implementation should be treated as ResEnc-L-like, not M or XL.

Why:

- It uses the `ResidualEncoderUNet` family and the ResEnc block schedule used by nnUNet v2 residual presets:
  - encoder blocks `(1, 3, 4, 6, 6, 6, ...)`
  - decoder convs `(1, 1, 1, ...)`
- It uses the standard nnUNet v2 feature scaling policy:
  - base features `32`
  - max features `320` for 3D
  - max features `512` for 2D

Why it is not truly M, L, or XL yet:

- FRAN is not using `nnUNetPlannerResEncM/L/XL`
- FRAN is not using nnUNet v2's planner-level GPU memory target logic
- FRAN is not using the planner-generated patch size and batch size that distinguish M vs L vs XL for the same dataset

In official nnUNet v2, M/L/XL differ mainly by planner memory target and the resulting planned configuration, not by a completely different residual block family.

So the current branch is "L-like" because:

- the architectural defaults were chosen to mirror the common ResEnc-L policy style
- but the actual FRAN geometry still comes from FRAN plan inputs rather than an nnUNet v2 L planner

Short rule:

- current FRAN `resunet` = ResEnc-L-like network policy on top of FRAN-owned planning
- not a full faithful `nnUNetPlannerResEncL` reproduction

What is currently derived from FRAN plan and config:

- From `plan`:
  - `spacing`
  - `patch_size`
- From `model_params`:
  - `in_channels`
  - `out_channels`
  - `compiled`

What is currently derived from helper code:

- `conv_op`
- `norm_op`
- `kernel_sizes`
- `strides`
- `n_stages`
- `features_per_stage`

What is currently hard-coded:

- `ResidualEncoderUNet` family choice
- ResEnc-L-like encoder/decoder block schedules
- `conv_bias=True`
- `norm_op_kwargs={"eps": 1e-5, "affine": True}`
- `dropout_op=None`
- `nonlin=nn.LeakyReLU`
- feature caps

This means the repo is currently in a hybrid state:

- planning remains FRAN-owned
- model backend for `resunet` is now `nnUNet v2`-style ResEnc
- preprocessing, dataset handling, training loop, inference, and config system remain FRAN-owned

## Existing Modules And Their Roles

Current nnUNet v2 migration-related files in the repo:

- [fran/architectures/create_network.py](/home/ub/code/fran/fran/architectures/create_network.py)
  - main live integration point
  - `resunet` case-insensitive dispatch now builds `ResidualEncoderUNet`
  - this is the only current path that affects actual FRAN model creation

- [fran/architectures/nnunetv2/resenc_explorer.py](/home/ub/code/fran/fran/architectures/nnunetv2/resenc_explorer.py)
  - exploratory utility module
  - imports nnUNet v2 residual planner-related classes and topology helpers
  - lets the user vary memory targets and inspect planned-like outputs:
    - preset hint
    - planner class hint
    - patch size
    - batch size
    - stage count
    - feature schedule
    - strides
    - kernel sizes
    - VRAM estimate
  - this is for investigation and discussion, not yet wired into FRAN training

- [fran/architectures/nnunetv2/__init__.py](/home/ub/code/fran/fran/architectures/nnunetv2/__init__.py)
  - convenience exports for the explorer utilities

- [fran/architectures/res_unet_planner.py](/home/ub/code/fran/fran/architectures/res_unet_planner.py)
  - older scratch / exploratory file
  - contains some manual imports and direct experimentation with residual planners and `ResidualEncoderUNet`
  - not currently part of the main FRAN execution path
  - should be treated as exploratory reference, not production integration

- [migration_nnunetv2.md](/home/ub/code/fran/migration_nnunetv2.md)
  - persistent migration state and handoff note
  - should be updated whenever the integration meaningfully changes

- [pyproject.toml](/home/ub/code/fran/pyproject.toml)
  - `nnunetv2>=2` was added to FRAN's `train` optional dependencies

Environment state relevant to this migration:

- `nnunetv2` is installed in the `dl` conda environment
- `monai` was updated to `1.5.2`
- `torch` in `dl` is currently `2.10.0+cu128`
- `torchaudio` remains broken in that env, but is not currently relevant if unused

## Completed So Far

Implemented:

- direct FRAN `resunet` -> `nnUNet v2` `ResidualEncoderUNet` integration
- case-insensitive `resunet` model selector
- topology derivation from FRAN `spacing` and `patch_size`
- separate nnUNet v2 exploration module for memory-target experiments
- migration guide and handoff note

Not implemented:

- planner-driven FRAN patch-size selection
- planner-driven batch-size selection in actual FRAN training
- true M/L/XL preset switching in live model creation
- planner-derived target spacing
- planner-derived lowres/cascade configs
- actual `plans.json` generation from FRAN config state

## What This Is Not Yet

This is not yet full nnUNet v2 planning or training.

Not yet adopted:

- `ExperimentPlanner` / `ResEncUNetPlanner`
- dataset fingerprint driven target spacing selection
- VRAM-constrained patch shrinking loop
- generated `plans.json`
- `get_network_from_plans(...)`
- lowres / cascade planning
- nnUNet v2 trainer / dataloaders / validation path

## Why This Step Was Chosen

The goal of this step is low-risk migration.

Instead of replacing FRAN's planning/config system, the code now:

- keeps FRAN spreadsheet and config flow intact
- keeps FRAN training and inference stack intact
- swaps only the network implementation for one branch
- uses nnUNet v2 topology logic only where it is low-coupling and geometry-focused

This is the smallest useful compatibility layer.

## Code Path Summary

Entry point:

- `create_model_from_conf(model_params, plan, deep_supervision=True)`

New dispatch:

- if `model_params["arch"].lower() == "resunet"`:
  - call `create_model_from_conf_nnunetv2_resenc_l(...)`

New function responsibilities:

1. Read plan geometry
2. Infer dimensionality
3. Build `conv_op` and matching `norm_op`
4. Call `get_pool_and_conv_props(spacing, patch_size, 4, 999999)`
5. Derive:
   - `kernel_sizes`
   - `strides`
   - `n_stages`
6. Build `features_per_stage`
7. Instantiate `ResidualEncoderUNet`

## Comparison With nnUNet v2 Proper

What matches nnUNet v2:

- actual `ResidualEncoderUNet` class
- topology derivation using `get_pool_and_conv_props(...)`
- dimensional conv/norm selection
- ResEnc block pattern
- feature cap policy

What differs from nnUNet v2:

- FRAN passes already-chosen `spacing` and `patch_size`
- nnUNet v2 planner normally derives those from dataset fingerprint + VRAM target
- current FRAN path does not use dataset fingerprint, target spacing heuristics, or VRAM estimation

So the current design is:

- FRAN plan provides geometry
- nnUNet v2 code provides model family and topology interpretation

## Known Risks

- The new path assumes `plan["spacing"]` and `plan["patch_size"]` are already sensible for ResEnc.
- No VRAM fitting is done yet, so some FRAN plans may instantiate models that are too large for actual training.
- Deep supervision output behavior has not been compared in detail against existing FRAN loss expectations for all pipelines.
- No checkpoint compatibility is guaranteed between the old `nnUNet` branch and the new `resunet` branch.
- The selected ResEnc policy is effectively "L-like" but not a complete reproduction of `nnUNetPlannerResEncL`.

## Next Milestones

### Milestone 1

Stabilize the current `resunet` path.

Tasks:

- run one actual FRAN training config with `arch=resunet`
- verify output shapes and deep supervision outputs
- verify loss path and validation path
- compare parameter count and memory against the old FRAN `nnUNet` path

### Milestone 2

Make hard-coded ResEnc policy explicit and configurable.

Tasks:

- move encoder/decoder block schedules and feature caps into one small helper or config dict
- allow future selection among M / L / XL style presets
- keep `resunet` as stable default alias
- likely location:
  - helper within [fran/architectures/create_network.py](/home/ub/code/fran/fran/architectures/create_network.py)
  - or shared utility under [fran/architectures/nnunetv2/](/home/ub/code/fran/fran/architectures/nnunetv2/)

### Milestone 3

Adopt more nnUNet v2 topology/planning functions while still using FRAN config sources.

Good candidates:

- `static_estimate_VRAM_usage(...)`
- `determine_fullres_target_spacing(...)`
- selected pieces of `ExperimentPlanner`
- eventual comparison against:
  - `nnUNetPlannerResEncM`
  - `nnUNetPlannerResEncL`
  - `nnUNetPlannerResEncXL`

Goal:

- derive patch size or validate patch size against VRAM without replacing FRAN spreadsheet planning yet

### Milestone 4

Introduce a normalized internal nnUNet-like plan object.

Goal:

- stop passing large raw FRAN plan dicts everywhere
- create one typed normalized object for:
  - spacing
  - patch size
  - channels
  - labels
  - architecture family
  - optional VRAM target

### Milestone 5

Optionally emit real nnUNet v2-style plans.

Goal:

- serialize FRAN-derived planning into a valid nnUNet-style `plans.json`
- later reuse `get_network_from_plans(...)` and possibly other nnUNet v2 runtime utilities

## Planned Immediate Discussion Topics

If work resumes later, the highest-value discussion topics are:

- whether FRAN should keep owning `spacing` and `patch_size`, or only `spacing`
- whether live `resunet` should stay hard-coded to L-like defaults or expose M/L/XL
- whether batch size should continue to be FRAN-owned or become planner-derived
- whether a normalized internal "nnUNet-like plan" object should be introduced before adding more planner logic
- whether `res_unet_planner.py` should be retired once the new `architectures/nnunetv2/` utilities are sufficient

## Suggested Immediate Next Step

The next safest engineering step is:

- keep `arch=resunet`
- train one small known-good config
- inspect one forward pass and one batch end-to-end
- confirm:
  - logits shape
  - deep supervision shape list
  - loss compatibility
  - GPU memory usage

Only after that should VRAM estimation or target spacing logic be introduced.

## Transition Rules

Until migration is farther along, keep these rules:

- FRAN remains source of truth for plan geometry
- `resunet` means nnUNet v2 `ResidualEncoderUNet`
- do not silently replace old `nnUNet` path
- do not introduce planner-driven spacing changes yet
- compare behavior against existing FRAN runs before expanding scope

## Files Most Relevant To Continue This Work

- [fran/architectures/create_network.py](/home/ub/code/fran/fran/architectures/create_network.py)
- [fran/architectures/nnunetv2/resenc_explorer.py](/home/ub/code/fran/fran/architectures/nnunetv2/resenc_explorer.py)
- [fran/architectures/nnunetv2/__init__.py](/home/ub/code/fran/fran/architectures/nnunetv2/__init__.py)
- [fran/architectures/res_unet_planner.py](/home/ub/code/fran/fran/architectures/res_unet_planner.py)
- [fran/configs/parser.py](/home/ub/code/fran/fran/configs/parser.py)
- [fran/managers/unet.py](/home/ub/code/fran/fran/managers/unet.py)
- [fran/managers/data/training.py](/home/ub/code/fran/fran/managers/data/training.py)
- [fran/inference/base.py](/home/ub/code/fran/fran/inference/base.py)
- [pyproject.toml](/home/ub/code/fran/pyproject.toml)

Upstream references currently used:

- `dynamic_network_architectures.architectures.unet.ResidualEncoderUNet`
- `nnunetv2.experiment_planning.experiment_planners.network_topology.get_pool_and_conv_props`

## Minimal Resume Context

If picking this work up later, the shortest accurate summary is:

"FRAN `resunet` now instantiates nnUNet v2 `ResidualEncoderUNet` using FRAN `spacing` and `patch_size`; topology is derived with `get_pool_and_conv_props`, but planning, spacing selection, VRAM fitting, preprocessing, and training stack are still FRAN-owned."
