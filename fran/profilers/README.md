# FRAN Profilers

Profiler outputs are written to `/s/fran_storage/logs/profiler/<project>/`.

## Training Timing

Use this first to measure wall time per batch and GPU idle during a real short `fit()`:

```bash
python -m fran.run.p_train_perf -t kits23 -p 2 --devices '[0]' --bs 6 --train-indices 192 --val-indices 1 --limit-train-batches 24 --num-workers 24 --prefetch-factor 4 --batch-affine false
```

Switch `--batch-affine true` to remove CPU `RandAffined` from the dataloader and apply the batch affine in Lightning `on_after_batch_transfer`.

## Transform Timing

Use this to attribute dataloader time to individual transforms inside worker processes:

```bash
python -m fran.run.profile_transform_times -t kits23 -p 2 --devices '[0]' --bs 6 --train-indices 192 --limit-batches 24 --num-workers 24 --prefetch-factor 4 --batch-affine false
```

Read `transform_summary_*.csv`. The highest `total_ms` rows are the worker-side bottlenecks. Current runs showed `LoadImaged` and `LoadTorchDict` dominate, not affine/crop/normalise.

## Live GPU Sampling

Use this beside any training command to sample GPU utilisation:

```bash
python -m fran.run.profile_live --gpu 0
```

## Torch Op Summary

Use this for CPU/CUDA op tables without producing a huge Perfetto trace:

```bash
python -m fran.run.profile_train_stacks -t kits23 -p 2 --devices '[0]' --batch-size 1 --n-samples 2 --skip-val true --num-workers 0 --limit-train-batches 5
```

Only request a Chrome/Perfetto timeline deliberately:

```bash
python -m fran.run.profile_train_stacks -t kits23 -p 2 --devices '[0]' --batch-size 1 --n-samples 2 --skip-val true --num-workers 0 --limit-train-batches 5 --export-chrome-trace true --cpu-profiling false --profile-with-stack false --export-stacks false
```

Avoid combining Chrome traces with CPU profiling and stack export unless you explicitly want very large JSON files.

