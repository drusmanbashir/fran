# W&B Runtime Behavior

This project uses `fran/managers/wandb.py` (`WandbManager`) for training-time W&B init and fallback handling.

## Init and watchdog

- W&B init is configured with `wandb.Settings(init_timeout=120)`.
- A hard watchdog wraps each init attempt (`45s` default).
- If an init attempt exceeds the watchdog, control returns to training code (`TimeoutError` path).

## Offline fallback

Offline fallback is triggered on init timeout paths that fail online startup.
When offline is activated, runs continue locally and can be synced later with `wandb sync`.

## Logs and sync artifacts

The manager writes runtime diagnostics under:

- `<project.log_folder>/wandb_sync/events.log`
- `<project.log_folder>/wandb_sync/latest_sync.txt`
- `<project.log_folder>/wandb_sync/<run_id>_sync.txt`

`latest_sync.txt` contains the exact `wandb sync ...` command for the most recent offline run.

## Quick operations

Tail event log:

```bash
tail -f <project_log_folder>/wandb_sync/events.log
```

Sync latest offline run:

```bash
cat <project_log_folder>/wandb_sync/latest_sync.txt
```
