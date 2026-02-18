# Incremental Training Rewrite Plan (Lightning-First)

## Goal
Implement incremental training using Lightning-native loops/callbacks as the primary control surface, while preserving:
- fixed validation split for plateau detection,
- adaptive sample admission from train pool,
- combined acquisition metric (Dice difficulty + uncertainty + diversity),
- full run/stage tracking.

## Step 1: Baseline Contract + Interfaces
- Define a single config contract for incremental runs (`dataset_params` + `incremental_params`).
- Define stage-state schema (active cases, candidate pool, thresholds, weights, stop reason).
- Keep tracking files (`stages.csv`, `state.json`) as authoritative artifacts.

**Checkpoint with user:** confirm config keys and default values.

## Step 2: Lightning-Native Case Metrics
- Move per-case collection fully into Lightning callback/module outputs.
- Use Lightning validation/predict loops (not ad-hoc manual loops) to gather candidate metrics.
- Keep metric computation deterministic and batch-size agnostic.

**Checkpoint with user:** verify metric quality on a small split.

## Step 3: Lightning-Native Stage Controller
- Implement an `IncrementalStageController` callback that:
  - evaluates plateau state,
  - triggers candidate scoring on train pool,
  - updates datamodule active case IDs,
  - logs stage transition artifacts.
- Use `reload_dataloaders_every_n_epochs` and datamodule state mutation for stage transitions.

**Checkpoint with user:** validate stage transition behavior and case counts.

## Step 4: Plateau/LR Logic via Lightning Callbacks
- Keep `EarlyStopping`, `ModelCheckpoint`, `LearningRateMonitor`.
- Add a thin Lightning callback for LR-floor stop.
- Ensure stage stop reasons are emitted from callback events.

**Checkpoint with user:** verify stop reasons and checkpoint continuity.

## Step 5: Combined Acquisition Metric
- Compute normalized components:
  - `difficulty = 1 - dice`
  - `uncertainty = normalized entropy`
  - `diversity = farthest-first embedding distance`
- Score:
  - `acq = w_dice*difficulty + w_uncertainty*uncertainty + w_diversity*diversity`
- Keep hard eligibility gate: `dice <= selection_threshold`.

**Checkpoint with user:** tune initial weights and threshold.

## Step 6: CLI + Troubleshooting Workflow
- Expose all key knobs in `fran/run/train.py`.
- Add a dry-run mode for stage-control sanity checks.
- Run minimal verification and capture known failure signatures + fixes.

**Checkpoint with user:** run together, inspect logs, adjust one knob at a time.

## Proposed Defaults (initial)
- `w_dice=0.5`, `w_uncertainty=0.3`, `w_diversity=0.2`
- `selection_threshold=0.7`
- `early_stopping_patience=20`
- `min_lr_to_continue=1e-5`

