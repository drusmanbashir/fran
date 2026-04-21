# HPC Preproc Context

Date: 2026-04-21

## Confirmed Facts

- Failed preproc job: `8100653`, job name `fran_preproc`.
- Failure node/partition: `ehc6` on `compute`.
- Failure cause:
  - `module: command not found`
  - `conda: command not found`
  - `/etc/profile.d/conda.sh: No such file or directory`
  - `ModuleNotFoundError: No module named 'fran'`
- This was environment/node mismatch, not memory/time:
  - `Elapsed=00:00:00`
  - `MaxRSS=5516K`
  - `ExitCode=1:0`

## Working Pattern

- `train.sh` works because it uses Andrena:
  - `#SBATCH -p andrena`
  - `#SBATCH -A pilot_andrena`
  - `#SBATCH --gres=gpu:1`
  - `#SBATCH --mem-per-cpu=7500M`
  - `module load miniforge`
  - `conda activate dl`

## Trace Result

- Trace job: `8174491`
- Job name: `fran_preproc_trace`
- Partition: `andrena`
- Node: `sbg10`
- State: `COMPLETED`
- Exit: `0:0`
- Runtime: `00:00:05`
- Trace output:
  - `python=/data/home/mpx588/.conda/envs/dl/bin/python`
  - `fran=/data/EECS-LITQ/fran_storage/code/fran/fran/__init__.py`

## Andrena Constraint

- CPU-only Andrena trace failed with `QOSMinGRES`.
- Same trace with `--gres=gpu:1` completed.
- Practical conclusion: current `andrena`/`pilot_andrena` QoS requires GPU GRES.

## Current Next Intent

- Next run should test training/preproc path with:
  - `gpu=0`
  - `compile=True`
- Expectation: `compile=True` may speed long GPU training after warmup if model shapes are stable, but may slow short runs because first iterations pay compile overhead.
- Risk: PyTorch compile can increase memory use or fail on dynamic/control-flow-heavy MONAI transforms/models. Run a short tracer/smoke first before long job.

## Useful Slurm Paths

- Slurm binaries on HPC:
  - `/opt/slurm/bin/sbatch`
  - `/opt/slurm/bin/squeue`
  - `/opt/slurm/bin/sacct`
- Logs:
  - `/data/EECS-LITQ/fran_storage/logs/%x-%j.out`
  - `/data/EECS-LITQ/fran_storage/logs/%x-%j.err`

## Docs Checked

- https://slurm-docs.hpc.qmul.ac.uk/using/uge_to_slurm/
- https://slurm-docs.hpc.qmul.ac.uk/nodes/andrena/
- https://slurm-docs.hpc.qmul.ac.uk/apps/languages/miniforge/
