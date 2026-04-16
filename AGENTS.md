# Repo-specific instructions
- For new or edited `.sh` scripts, follow the style of [`fran/run/analyze.sh`](/home/ub/code/fran/fran/run/analyze.sh): keep them simple, prefer a few commented example commands with sensible defaults, and one direct active command.
- Avoid environment-variable wrapper boilerplate in `.sh` scripts unless the task specifically needs it.

## Common File Locations And Purposes
- `/s/fran_storage/conf/datasets.yaml`: canonical dataset aliases and root folders.
- `/s/fran_storage/conf/best_runs.yaml`: curated best runs for non-LIDC projects.
- `/s/fran_storage/inference_image_folders.yaml`: quick lookup for common inference image folders.
- `/home/ub/code/fran/fran/inference/base.py`: Base inferer (`source` mode, sliding-window).
- `/home/ub/code/fran/fran/inference/cascade.py`: Whole and Cascade inferers (`whole`, `lbd` flows).
- `/home/ub/code/fran/fran/inference/ensemble.py`: multi-run inference orchestration.
