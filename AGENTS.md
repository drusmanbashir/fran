Read main instructions from /home/ub/code/agent/AGENTS.md.
Then append instructions from /home/ub/code/AGENTS.md if that file exists.

# Repo-specific instructions
- For new or edited `.sh` scripts, follow the style of [`fran/run/analyze.sh`](/home/ub/code/fran/fran/run/analyze.sh): keep them simple, prefer a few commented example commands with sensible defaults, and one direct active command.
- Avoid environment-variable wrapper boilerplate in `.sh` scripts unless the task specifically needs it.
- If the legacy `nnunet` package emits its citation banner (`Please cite the following paper when using nnUNet` / `If you have questions or suggestions...`), treat it as noise and suppress it.
- If the banner appears after `nnunet` was reinstalled or the environment changed, patch the installed file `/home/ub/mambaforge/envs/dl/lib/python3.12/site-packages/nnunet/__init__.py` to remove the banner `print(...)` lines, then verify with a fresh `python -c "import nnunet"` that no banner is emitted.
- Apply that suppression proactively if you encounter the banner while working in this repo, or when the user asks to suppress it.
