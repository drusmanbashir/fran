# NAS Findings and Recommendations (2026-03-03)

## Scope
This note summarizes:
- Recent W&B run status and key metrics.
- Relevant implementation details from tuning code.
- Hyperparameter search recommendations aligned with current configs.
- A curated reading list for modern HPO and segmentation tuning.

## Data Sources
- W&B projects under `drubashir/*` queried on 2026-03-03.
- Local files:
  - `fran/managers/tune.py`
  - `fran/tune/tune.py`
  - `fran/tune/trainer.py`
  - `configurations/experiment_configs.xlsx`
  - `fran/templates/tune.yaml`

## Recent W&B Snapshot
Representative runs:
- Colon: `COLON-0009` (finished, 2026-02-28)
  - URL: https://wandb.ai/drubashir/colon/runs/COLON-0009
  - `val0_loss_dice_epoch = 0.3605444431304931`
  - `val0_loss_epoch = 0.3943847119808197`
  - `epoch = 499`
  - Config highlights: `lr=0.001`, `src_dim0=192`, `src_dim1=128`, `patch_dim0=128`, `patch_dim1=96`, `expand_by=50`

- Nodes: `NODES-0099` (crashed, 2026-02-27)
  - URL: https://wandb.ai/drubashir/nodes/runs/NODES-0099
  - `val0_loss_dice_epoch = 0.5610391497612`
  - `val0_loss_epoch = 1.014270901679993`
  - `epoch = 3`
  - Config highlights: `lr=0.001`, `patch_dim0=128`, `expand_by=0`

- Pancreas: `PANCREAS-0004` (failed, 2026-03-03)
  - URL: https://wandb.ai/drubashir/pancreas/runs/PANCREAS-0004
  - `val0_loss_dice_epoch = 0.48253798484802246`
  - `val0_loss_epoch = 0.5035114884376526`
  - `epoch = 31`
  - Config highlights: `lr=0.01`, `patch_dim0=192`, `patch_dim1=96`, `expand_by=100`
  - `output.log` ends with `Detected KeyboardInterrupt, attempting graceful shutdown ...`

- LIDC: `LIDC-0002` (crashed, 2026-03-01)
  - URL: https://wandb.ai/drubashir/lidc/runs/LIDC-0002
  - `val0_loss_dice_epoch = 0.1217351332306862`
  - `val0_loss_epoch = 0.14936141669750214`
  - `epoch = 46`
  - Config highlights: `lr=0.01`, `patch_dim0=128`, `patch_dim1=96`, `expand_by=50`

Project-level status trend:
- `nodes`: most recent 12 runs are crashed.
- `lidc`: recent runs crashed.
- `pancreas`: latest run failed (at least one appears interrupt-driven).
- `colon`: mixed, with successful long runs.
- `bones`: mixed (one finished plus several crashes).

## Local Config/Tune State
From `configurations/experiment_configs.xlsx`:
- Active tuned model params:
  - `base_ch_opts` (choice `[16,32]`)
  - `lr` (qloguniform `[1e-4,5e-2]`)
  - `deep_supervision` (choice `[True,False]`)
- Active tuned transform factors:
  - `contrast`, `shift`, `scale`, `brightness` (all `double_range`)
- No active tune rows in `dataset_params`, `affine3d`, `loss_params`, `plans` sheets.

Plan/search-relevant manual defaults include:
- `src_dim0=192`, `src_dim1=128`.
- Common plan options include `patch_dim0 in {96,128,192,224}`, `patch_dim1=96`, `expand_by` values including 0, 50, 100.

## Code Findings (Tuning Pipeline)
1. Metric mapping mismatch risk
- In `fran/tune/trainer.py`, Tune reports:
  - `metrics={"loss": "val1_loss_dice"}`
- In W&B summaries, recent runs consistently expose `val0_*` metrics (`val0_loss_dice_epoch`, `val0_loss_epoch`) rather than `val1_*`.
- Risk: search objective may be absent/unstable for some runs, reducing search quality.

2. Scheduler grace-period inconsistency
- `fran/tune/tune.py` sets `grace_period = 20` but constructs ASHA with `grace_period=3`.
- This can prune too aggressively before meaningful convergence for 3D segmentation.

3. Search space breadth vs reliability
- Current space mixes architecture/training with strong augmentation ranges.
- With frequent crash/fail states in some projects, broad coupled search likely wastes budget.

4. Batch size logic is coarse
- Current trial batch size in `fran/tune/tune.py` is derived only from `src_dims[0]` threshold.
- Large patch/context combos may still destabilize memory/time in ways not captured by this rule.

## Recommendations for HParam Search
1. Stabilize objective plumbing first
- Ensure Tune objective always maps to logged metric present in all runs.
- Prefer a single primary objective for search:
  - Minimize validation dice-loss (`val0_loss_dice_epoch` or equivalent canonical key).

2. Run staged HPO (modern practical pattern)
- Stage A (exploration):
  - Random + ASHA.
  - 40-60 trials.
  - 40-80 epochs max.
  - Moderate pruning (grace period aligned to warm-up behavior).
- Stage B (exploitation):
  - TPE/Optuna + ASHA.
  - 20-40 trials centered on top Stage-A region.

3. Narrow LR early based on observed outcomes
- Start with `lr` centered near `1e-3` and practical range around `3e-4` to `3e-3`.
- Reintroduce `1e-2` only if project-specific evidence supports it.

4. Use conditional search constraints
- If `patch_dim0 >= 192` or `expand_by` high, constrain LR and augmentation amplitude.
- Tie resource-heavy geometry with conservative optimizer settings.

5. Decouple augmentation tuning
- First tune core optimizer/model/patch geometry with fixed augmentations.
- Then run a second dedicated augmentation sweep.

6. Per-project strategy
- `colon`: use as stable reference prior.
- `nodes/lidc`: focus first on crash-free regimen before broad HPO.
- `pancreas`: investigate operational stop conditions separately from model quality.

## Suggested Immediate Sweep Template
Phase 1 (stability-first):
- Tune: `lr`, `base_ch_opts`, `deep_supervision`, `patch_dim0`, `expand_by`.
- Fix transforms to conservative defaults.
- Objective: `val0_loss_dice_epoch`.

Phase 2 (augmentation):
- Freeze top 3-5 configs from Phase 1.
- Tune transform ranges/probabilities only.
- Keep same objective and pruning policy.

## Curated Reading List
1. Ray Tune key concepts
- https://docs.ray.io/en/latest/tune/key-concepts.html

2. Ray Tune search algorithms
- https://docs.ray.io/en/latest/tune-searchalg.html

3. W&B sweeps configuration guide
- https://docs.wandb.ai/guides/sweeps/define-sweep-configuration

4. W&B sweep config keys
- https://docs.wandb.ai/models/sweeps/sweep-config-keys

5. Optuna samplers and pruners
- https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html

6. ASHA paper
- https://arxiv.org/abs/1810.05934

7. Hyperband paper
- https://www.jmlr.org/papers/v18/16-558.html

8. BOHB paper
- https://proceedings.mlr.press/v80/falkner18a.html

9. Random Search baseline
- https://www.jmlr.org/papers/v13/bergstra12a.html

10. nnU-Net design priors
- https://www.nature.com/articles/s41592-020-01008-z

## Notes
- This report uses the latest available run metadata at query time (2026-03-03).
- Some runs lack full artifacts/logs (common in crash cases), so exact crash root-cause may require extra local runtime logs.
