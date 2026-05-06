# Repo-specific instructions
- any time a script (shell or python) in fran/run is modified, compare for downstream effect in ~/code/agent/agent/hpc/ and fix those too.
- Avoid environment-variable wrapper boilerplate in `.sh` scripts unless the task specifically needs it.
- changes under /home/ub/code/fran outside /home/ub/code/fran/fran/run/ require explicit user approval first.

## Common File Locations And Purposes
- `/s/fran_storage/conf/datasets.yaml`: canonical dataset aliases and root folders.
- `/s/fran_storage/conf/best_runs.yaml`: curated best runs for non-LIDC projects.
- `/s/fran_storage/inference_image_folders.yaml`: quick lookup for common inference image folders.
- `/home/ub/code/fran/fran/inference/base.py`: Base inferer (`source` mode, sliding-window).
- `/home/ub/code/fran/fran/inference/cascade.py`: Whole and Cascade inferers (`whole`, `lbd` flows).
- `/home/ub/code/fran/fran/inference/ensemble.py`: multi-run inference orchestration.
- `/home/ub/code/fran/fran/tests/`: ALL test scripts
