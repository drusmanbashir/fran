# Benchmark Context

Date: 2026-04-21

## Dataset / Config

- Project: `kits23`
- Config: `ConfigMaker(Project("kits23")).setup(2)`
- Train mode: `rbd`
- Data folder: `/r/datasets/preprocessed/kits23/rbd/spc_080_080_150_54787144`
- HDF5 shard folder: `hdf5_shards/src_192_192_128`
- Shards: 98 total, 489 cases, 5 cases/shard except final
- Batch size: 4
- Train dataset length: 408
- Train dataloader length: 102
- Batch shape: `(4, 1, 160, 160, 96)`
- `ds_type=None`
- `cache_rate=0.0`
- `torch.set_float32_matmul_precision("medium")`

## Key Code State

`fran/managers/data/training2.py` is the HDF5-sharded variant.

For train RBD, transform order is:

```text
Ld,Rtr,L2,E,F1,F2,Affine,ResizePC,N,IntensityTfms
```

Meanings:

```text
Ld  = LoadHDF5ShardIndexd, loads fg/bg flat indices and shard path from manifest
Rtr = RandCropByFlatIndicesd, random crop center
L2  = LoadHDF5Cropd, reads crop from HDF5 shard
```

`training.py` is legacy full `.pt` image/lm load:

```text
L,Remap,Ld,E,Rtr,N,F1,F2,Affine,ResizePC,IntensityTfms
```

`training2.py` dual-SSD behavior was changed:

- If HDF5 shard manifest exists, `dual_ssd=True` stages odd `shard_*.h5` files to `COMMON_PATHS["rapid_access_folder2"]`.
- It writes a staged manifest with absolute shard paths.
- It does not use legacy image/lm/indices copy in the HDF5 path.
- `LoadHDF5ShardIndexd` now accepts absolute shard paths in manifest.
- `training.py` legacy image/lm pair copy path left separate.

Staged dual-SSD manifest:

```text
/home/ub/datasets/preprocessed/kits23/rbd/spc_080_080_150_54787144/hdf5_shards/src_192_192_128/manifest.json
```

Manifest check:

```text
98 shards total
49 /r paths
49 /home paths
```

Temp duplicate staging from aborted tests may exist:

```text
/home/ub/tmp/kits23_kbd_dualssd_spc_080_080_150_54787144
/home/ub/tmp/kits23_kbd_dualssd_shards
```

These are disposable.

## Benchmarks Run

### HDF5 Sharded vs Full `.pt`

Benchmark: 20 train batches, batch size 4.

```text
training2.py HDF5 sharded:
  prepare_data: 5.46s
  setup:        0.006s
  first batch:  1.99s
  20 batches:   15.10s
  throughput:   5.30 samples/s

training.py full .pt load:
  prepare_data: 7.43s
  setup:        0.006s
  first batch:  9.11s
  20 batches:   23.85s
  throughput:   3.35 samples/s
```

Conclusion: HDF5 sharding faster than full `.pt` load for this test.

JSON:

```text
/tmp/fran_kits23_training_vs_training2_bench.json
```

### HDF5 Worker / Cache Variants

Benchmark: `training2.py`, 2 full train epochs, batch size 4, GPU default in that run.

JSON:

```text
/tmp/fran_kits23_training2_2epoch_worker_h5cache_bench.json
```

Results:

```text
baseline
  epoch1 49.25s  8.28 samples/s
  epoch2 50.98s  8.00 samples/s
  total  100.24s 8.14 samples/s

persistent_workers=True
  epoch1 68.05s  6.00 samples/s
  epoch2 62.57s  6.52 samples/s
  total  130.62s 6.25 samples/s
  worse

handle_cache_lru8 only
  epoch1 57.89s  7.05 samples/s
  epoch2 54.52s  7.48 samples/s
  total  112.41s 7.26 samples/s
  worse

persistent + handle_cache_lru8
  epoch1 50.19s  8.13 samples/s
  epoch2 50.10s  8.14 samples/s
  total  100.29s 8.14 samples/s
  about same as baseline
```

Conclusion:

```text
persistent alone: no improvement
handle cache alone: no improvement
persistent + handle cache: neutral
```

### Dual-SSD on GPU 1

Command:

```bash
CUDA_VISIBLE_DEVICES=1 python /tmp/bench_training2_dual_ssd.py \
  --epochs 2 \
  --batch-size 4 \
  --output-json /tmp/bench_training2_dual_ssd_gpu1_20260420_233827.json
```

Files:

```text
/tmp/bench_training2_dual_ssd.py
/tmp/bench_training2_dual_ssd_gpu1_20260420_233827.json
/tmp/bench_training2_dual_ssd_gpu1_20260420_233827.log
```

Results:

```text
dual_ssd=False
  prepare+setup: 6.694s
  epoch1: 54.611s  7.471 samples/s
  epoch2: 53.959s  7.561 samples/s
  RSS E1: 1.826 -> 3.802 GiB
  RSS E2: 3.802 -> 3.927 GiB

dual_ssd=True
  prepare+setup: 5.932s
  epoch1: 62.044s  6.576 samples/s
  epoch2: 76.730s  5.317 samples/s
  RSS E1: 3.948 -> 3.947 GiB
  RSS E2: 3.947 -> 3.947 GiB
  staging reused existing copied shards
```

Conclusion: current dual-SSD staging is slower in this 2-epoch dataloader benchmark.

## Interpretation

MONAI `CacheDataset` / `PersistentDataset` likely do not help current HDF5 train path because caching stops before first random transform.

Current HDF5 train path:

```text
Ld deterministic
Rtr random
L2 deterministic crop read but after random
```

MONAI cache only caches through `Ld`, not random crop or crop read. `CacheNTransDataset` could cache through `Rtr/L2`, but would freeze random crops, wrong for training.

`pin_memory=True` remains OK for GPU training unless memory pressure/page-lock issues appear.

Past concern: training crashed after 10+ epochs. The theory was persistent workers / pinned memory / handles might accumulate memory, but 2-epoch test did not prove it. Current RSS for dual-SSD run did not grow, but 2 epochs is too short to rule out long-run leak.

## Next Useful Checks

1. To confirm dual-drive simultaneous reads:

```bash
iostat -xm 1 nvme0n1 nvme1n1
```

Run during epoch. Need both devices showing read throughput.

2. If strict dual drive use is desired, current implementation only gives opportunistic mixing through shuffled cases. It does not guarantee balanced workers/batches.

Potential future design:

```text
custom sampler/batch sampler:
  each batch includes 2 cases backed by /r shards + 2 cases backed by /home shards
```

or worker-aware routing:

```text
workers 0-3 read /r-backed cases
workers 4-7 read /home-backed cases
```

3. For `compile=True`, likely helps only if model compute dominates. It does not speed HDF5 I/O or MONAI CPU transforms. Test actual train step after warmup, not loader-only.
