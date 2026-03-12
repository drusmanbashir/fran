# %%
from __future__ import annotations
import ipdb
tr = ipdb.set_trace

import os
import re
import secrets
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch._dynamo
import wandb
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch.loggers import WandbLogger
from paramiko import SSHClient

from fran.utils.common import common_vars_filename
from utilz.fileio import load_yaml, maybe_makedirs

try:
    import numpy as np
except Exception:
    np = None

import stat
torch._dynamo.config.suppress_errors = True

def download_path_no_wandb(remote_dir_parent, local_dir_parent)->list:
        hpc_settings = load_yaml(os.environ["HPC_SETTINGS"])
        client = SSHClient()
        client.load_system_host_keys()
        client.connect(
            hpc_settings["host"],
            username=hpc_settings["username"],
            password=hpc_settings["password"],
        )

        with client.open_sftp() as ftp_client:
            def _recursive_filenames(remote_dir_parent):
                fnames = []
                aa = ftp_client.listdir_attr(remote_dir_parent)
                for aa1 in aa:
                    if stat.S_ISDIR(aa1.st_mode):
                        fnames.extend(
                            _recursive_filenames(
                                os.path.join(remote_dir_parent, aa1.filename)
                            )
                        )
                    elif ".ckpt" in aa1.filename:
                        fnames.append(os.path.join(remote_dir_parent, aa1.filename))
                    else:
                        print(f"Skipping {aa1.filename} as not a ckpt")
                return fnames

            ckpt_files = _recursive_filenames(remote_dir_parent)
            if local_dir_parent is None:
                return ckpt_files

            local_dir_parent = Path(local_dir_parent)
            local_files = []
            for remote_file in ckpt_files:
                rel = remote_file.split("/")[-1]
                local_file = local_dir_parent / rel
                local_file.parent.mkdir(parents=True, exist_ok=True)
                ftp_client.get(remote_file, str(local_file))
                local_files.append(local_file)
        return local_files




def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".wandb_write_probe"
        probe.touch(exist_ok=True)
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _resolve_wandb_save_dir(project) -> str:
    env_dir = os.environ.get("WANDB_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates += [Path(project.log_folder), Path.cwd() / "wandb", Path("/tmp") / "wandb"]
    for path in candidates:
        if _is_writable_dir(path):
            os.environ.setdefault("WANDB_DIR", str(path))
            return str(path)
    return str(Path.cwd())


def _to_plain(x):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if np is not None and isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, tuple):
        return [_to_plain(v) for v in x]
    if isinstance(x, list):
        return [_to_plain(v) for v in x]
    if isinstance(x, set):
        return [_to_plain(v) for v in sorted(x, key=lambda z: str(z))]
    if isinstance(x, dict):
        return {str(k): _to_plain(v) for k, v in x.items()}
    return str(x)


def get_wandb_config():
    commons = load_yaml(common_vars_filename)
    api_token = commons.get("wandb_api_token")
    entity = commons.get("wandb_entity")
    return entity, api_token


def get_wandb_project(project, mode: str = "online"):
    entity, api_token = get_wandb_config()
    if api_token:
        os.environ["WANDB_API_KEY"] = api_token
    if entity:
        os.environ.setdefault("WANDB_ENTITY", entity)
    api = wandb.Api()
    path = f"{entity}/{project.project_title}" if entity else project.project_title
    return {"api": api, "path": path, "mode": mode}


def _normalize_run_prefix(project_title: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9]+", "-", str(project_title).strip()).strip("-")
    return prefix.upper() if prefix else "RUN"


def _extract_seq(value: str, prefix: str) -> Optional[int]:
    m = re.fullmatch(rf"{re.escape(prefix)}-(\d+)", str(value or ""))
    if not m:
        return None
    return int(m.group(1))


def _next_ordered_run_id(entity: Optional[str], project_title: str, width: int = 4) -> str:
    prefix = _normalize_run_prefix(project_title)
    path = f"{entity}/{project_title}" if entity else project_title
    max_seq = 0
    mode = str(os.environ.get("WANDB_MODE", "")).lower()
    if mode in {"offline", "disabled", "dryrun"}:
        return f"{prefix}-{max_seq + 1:0{int(width)}d}"
    try:
        api = wandb.Api()
        for run in api.runs(path):
            for candidate in (run.id, run.name):
                seq = _extract_seq(candidate, prefix)
                if seq is not None:
                    max_seq = max(max_seq, seq)
    except Exception:
        # If API lookup fails, start from 1.
        max_seq = 0
    return f"{prefix}-{max_seq + 1:0{int(width)}d}"


def _next_random_run_id(entity: Optional[str], project_title: str, width: int = 2) -> str:
    prefix = _normalize_run_prefix(project_title)
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    path = f"{entity}/{project_title}" if entity else project_title
    mode = str(os.environ.get("WANDB_MODE", "")).lower()

    used: set[str] = set()
    if mode not in {"offline", "disabled", "dryrun"}:
        try:
            api = wandb.Api()
            for run in api.runs(path):
                run_id = getattr(run, "id", None)
                run_name = getattr(run, "name", None)
                if run_id:
                    used.add(str(run_id))
                if run_name:
                    used.add(str(run_name))
        except Exception:
            pass

    total = len(alphabet) ** int(width)
    max_attempts = min(total, 512)
    for _ in range(max_attempts):
        suffix = "".join(secrets.choice(alphabet) for _ in range(int(width)))
        candidate = f"{prefix}-{suffix}"
        if candidate not in used:
            return candidate

    raise RuntimeError(f"Could not allocate unique W&B run id for prefix '{prefix}' with width={width}")


def get_wandb_checkpoint(project, run_id):
    wl = WandbManager(
        project=project,
        run_id=run_id,
        wb_mode="disabled",
        log_model_checkpoints=False,
    )
    ckpt = wl.model_checkpoint
    wl.stop()
    return ckpt

def download_wandb_checkpoint(project, run_id):
    wl = WandbManager(
        project=project,
        run_id=run_id,
        wb_mode="disabled",
        log_model_checkpoints=False,
    )
    wl.download_checkpoints()
    ckpt = wl.model_checkpoint
    wl.stop()
    return ckpt


class WandbManager(WandbLogger):
    def __init__(
        self,
        *,
        project,
        wb_mode: str = "online",
        run_id: Optional[str] = None,
        log_model_checkpoints: Optional[bool] = False,
        prefix: str = "training",
        **wandb_init_kwargs: Any,
    ):
        self.project = project
        self.prefix = prefix.rstrip("/")
        self._offline_fallback_used = False
        self._watchdog_timeout_s = 45
        self._sync_hint_written = False
        self._sync_hint_file = None
        self.entity, api_token = get_wandb_config()
        save_dir = _resolve_wandb_save_dir(project)

        if api_token:
            os.environ["WANDB_API_KEY"] = api_token
        if self.entity:
            os.environ.setdefault("WANDB_ENTITY", self.entity)

        if run_id:
            resolved_run_id = run_id
            wandb_init_kwargs.setdefault("id", resolved_run_id)
            wandb_init_kwargs.setdefault("resume", "must")
            name = resolved_run_id
        else:
            resolved_run_id = _next_random_run_id(
                entity=self.entity,
                project_title=project.project_title,
                width=2,
            )
            wandb_init_kwargs.setdefault("id", resolved_run_id)
            name = resolved_run_id
        self._resolved_run_id = resolved_run_id

        env_mode = os.environ.get("WANDB_MODE")
        wandb_init_kwargs.setdefault("mode", env_mode if env_mode else wb_mode)
        self._mode = str(wandb_init_kwargs.get("mode", wb_mode)).lower()
        wandb_init_kwargs.setdefault("settings", wandb.Settings(init_timeout=120))

        super().__init__(
            project=project.project_title,
            entity=self.entity,
            name=name,
            save_dir=save_dir,
            log_model=("all" if log_model_checkpoints else False),
            **wandb_init_kwargs,
        )

        self.df = self.fetch_project_df() if self._mode not in {"disabled", "offline"} else self._empty_df()

    def _path(self, leaf: str) -> str:
        return f"{self.prefix}/{leaf}" if self.prefix else leaf

    def _log_event(self, message: str) -> None:
        print(message)
        try:
            out_dir = Path(self.project.log_folder) / "wandb_sync"
            out_dir.mkdir(parents=True, exist_ok=True)
            line = f"{datetime.now(timezone.utc).isoformat()} | {message}\n"
            with (out_dir / "events.log").open("a") as f:
                f.write(line)
        except Exception:
            pass

    def log_hyperparams(self, params):
        self.experiment.config.update(_to_plain(params), allow_val_change=True)

    @property
    def wb_run(self):
        return self.experiment

<<<<<<< HEAD
    def _project_path(self) -> str:
        return f"{self.entity}/{self.project.project_title}" if self.entity else self.project.project_title

    def _run_path(self, run_id: Optional[str] = None) -> str:
        run_id = run_id or self.run_id
        return f"{self._project_path()}/{run_id}"

    def _remote_run(self):
        try:
            api = wandb.Api()
            return api.run(self._run_path())
        except Exception:
            return None

    def _checkpoint_summary(self) -> dict[str, Any]:
        best_key = self._path("best_model_path")
        last_key = self._path("last_model_path")
        summary = {
            "best": self.experiment.summary.get(best_key),
            "last": self.experiment.summary.get(last_key),
        }
        if summary["best"] or summary["last"]:
            return summary
        remote_run = self._remote_run()
        if remote_run is None:
            return summary
        remote_summary = remote_run.summary
        summary["best"] = remote_summary.get(best_key)
        summary["last"] = remote_summary.get(last_key)
        return summary
=======
    def _is_init_timeout_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        return "run initialization has timed out" in msg or "wandb init hard timeout" in msg

    def _experiment_with_watchdog(self, timeout_s: Optional[int] = None):
        timeout_s = int(timeout_s or self._watchdog_timeout_s)
        if threading.current_thread() is not threading.main_thread():
            return super().experiment
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)

        def _on_timeout(signum, frame):
            raise TimeoutError(f"wandb init hard timeout after {timeout_s}s")

        signal.signal(signal.SIGALRM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
        try:
            return super().experiment
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])

    def _resolve_sync_dir(self, run) -> Optional[str]:
        for settings_attr in ("settings", "_settings"):
            settings = getattr(run, settings_attr, None)
            sync_dir = getattr(settings, "sync_dir", None)
            if sync_dir:
                return str(Path(sync_dir))
        run_dir = getattr(run, "dir", None)
        if run_dir:
            run_dir = Path(run_dir)
            if run_dir.name == "files":
                return str(run_dir.parent)
            return str(run_dir)
        return None

    def _write_sync_hint(self, run, reason: str) -> None:
        sync_dir = self._resolve_sync_dir(run)
        if not sync_dir:
            return
        run_id = getattr(run, "id", None) or self._wandb_init.get("id") or "unknown-run"
        out_dir = Path(self.project.log_folder) / "wandb_sync"
        out_dir.mkdir(parents=True, exist_ok=True)
        sync_cmd = f"wandb sync {sync_dir}"
        ts = datetime.now(timezone.utc).isoformat()
        payload = "\n".join(
            [
                f"timestamp_utc: {ts}",
                f"reason: {reason}",
                f"run_id: {run_id}",
                f"mode: {self._mode}",
                f"sync_dir: {sync_dir}",
                f"command: {sync_cmd}",
            ]
        )
        latest = out_dir / "latest_sync.txt"
        run_file = out_dir / f"{run_id}_sync.txt"
        latest.write_text(payload + "\n")
        run_file.write_text(payload + "\n")
        self._sync_hint_written = True
        self._sync_hint_file = str(latest)
        self._log_event(f"W&B sync info saved: {latest}")
        self._log_event(f"W&B sync command: {sync_cmd}")

    def _activate_offline_fallback(self, error: Exception) -> None:
        self._offline_fallback_used = True
        self._mode = "offline"
        self._wandb_init["mode"] = "offline"
        self._wandb_init.pop("resume", None)
        self._log_event(f"W&B init issue ({error}). Falling back to offline mode.")

    @property
    @rank_zero_experiment
    def experiment(self):
        try:
            run = self._experiment_with_watchdog()
        except Exception as error:
            if not self._offline_fallback_used and self._is_init_timeout_error(error):
                self._activate_offline_fallback(error)
                run = self._experiment_with_watchdog()
                self._write_sync_hint(run, reason="wandb_init_timeout_fallback")
                return run
            raise
        if self._mode == "offline" and not self._sync_hint_written:
            self._write_sync_hint(run, reason="wandb_offline_mode")
        return run
>>>>>>> fc7a7aa (wandb fix)

    @property
    def model_checkpoint(self):
        summary = self._checkpoint_summary()
        return summary["best"] or summary["last"]

    @model_checkpoint.setter
    def model_checkpoint(self, value):
        key = self._path("best_model_path")
        self.experiment.summary.update({key: str(value)})

    def fetch_project_df(self, columns=None):
        try:
            api = wandb.Api()
            path = f"{self.entity}/{self.project.project_title}" if self.entity else self.project.project_title
            runs = list(api.runs(path))
        except Exception:
            return self._empty_df(columns=columns)
        rows = []
        for run in runs:
            row = {
                "sys/id": run.id,
                "sys/name": run.name,
                "sys/creation_time": run.created_at,
                "sys/url": run.url,
            }
            row[self._path("best_model_path")] = run.summary.get(self._path("best_model_path"))
            row[self._path("last_model_path")] = run.summary.get(self._path("last_model_path"))
            if columns is None:
                rows.append(row)
            else:
                rows.append({k: row.get(k) for k in columns})
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except Exception:
            return rows

    def _empty_df(self, columns=None):
        cols = columns or ["sys/id", "sys/name", "sys/creation_time", "sys/url"]
        try:
            import pandas as pd

            return pd.DataFrame(columns=cols)
        except Exception:
            return []

    def load_run(self, run_name, wb_mode: str = "online"):
        run_id, msg = self.get_run_id(run_name)
        print(f"{msg}. Loading")
        return WandbManager(
            project=self.project,
            run_id=run_id,
            wb_mode=wb_mode,
            log_model_checkpoints=False,
        ).experiment

    def get_run_id(self, run_id):
        if run_id == "most_recent":
            run_id = self.id_most_recent()
            msg = "Most recent run"
        elif run_id in ("", None):
            raise Exception(f"Illegal run name: {run_id}. No ids exist with this name")
        else:
            self.id_exists(run_id)
            msg = f"Run id matching {run_id}"
        return run_id, msg

    def id_exists(self, run_id):
        if not hasattr(self.df, "loc"):
            return None
        row = self.df.loc[self.df["sys/id"] == run_id]
        try:
            existing = row["sys/id"].item()
            print(f"Existing Run found. Run id {existing}")
            return existing
        except Exception as e:
            print(f"No run with that name exists .. {e}")

    def _has_checkpoints(self, row):
        return bool(row.get(self._path("best_model_path")))

    def id_most_recent(self):
        if not hasattr(self.df, "sort_values") or len(self.df) == 0:
            raise RuntimeError("No runs found in project or remote API unavailable")
        df = self.df.sort_values(by="sys/creation_time", ascending=False)
        for _, row in df.iterrows():
            if self._has_checkpoints(row):
                print(f"Loading most recent run. Run id {row['sys/id']}")
                return row["sys/id"]
        if len(df) > 0:
            return df.iloc[0]["sys/id"]
        raise RuntimeError("No runs found in project")

    def download_checkpoints(self):
        summary = self._checkpoint_summary()
        ckpt_path = summary["best"] or summary["last"]
        if not ckpt_path:
            print("No checkpoints in this run")
            return
        remote_dir = str(Path(ckpt_path).parent)
        latest_ckpt = self.shadow_remote_ckpts(remote_dir)
        if latest_ckpt:
            updates = {self._path("last_model_path"): str(latest_ckpt)}
            if summary["best"]:
                updates[self._path("best_model_path")] = str(latest_ckpt)
            self.experiment.summary.update(updates)

    def shadow_remote_ckpts(self, remote_dir):
        hpc_settings = load_yaml(os.environ["HPC_SETTINGS"])
        local_dir = self.project.checkpoints_parent_folder / self.run_id / "checkpoints"
        print(f"\nSSH to remote folder {remote_dir}")

        client = SSHClient()
        client.load_system_host_keys()
        client.connect(
            hpc_settings["host"],
            username=hpc_settings["username"],
            password=hpc_settings["password"],
        )

        ftp_client = client.open_sftp()
        try:
            fnames = []
            for f in sorted(
                ftp_client.listdir_attr(remote_dir),
                key=lambda k: k.st_mtime,
                reverse=True,
            ):
                fnames.append(f.filename)
        except FileNotFoundError:
            print("\n------------------------------------------------------------------")
            print(f"Error:Could not find {remote_dir}.\nIs this a remote folder and exists?\n")
            return

        remote_fnames = [os.path.join(remote_dir, f) for f in fnames]
        local_fnames = [os.path.join(local_dir, f) for f in fnames]
        maybe_makedirs(local_dir)
        downloaded_files = []
        for rem, loc in zip(remote_fnames, local_fnames):
            if Path(loc).exists():
                print(f"Local file {loc} exists already.")
                downloaded_files.append(loc)
            else:
                print(f"Copying file {rem} to local folder {local_dir}")
                ftp_client.get(rem, loc)
                downloaded_files.append(loc)

        if not downloaded_files:
            return None
        latest_ckpt = max(downloaded_files, key=lambda f: Path(f).stat().st_mtime)
        return latest_ckpt

    def stop(self):
        run = self.experiment
        if self._mode == "offline":
            self._write_sync_hint(run, reason="wandb_stop")
        if wandb.run is run:
            wandb.finish()

    @property
    def run_id(self):
        return self.experiment.id or self._resolved_run_id

    @property
    def save_dir(self) -> Optional[str]:
        return str(self.project.checkpoints_parent_folder)

# %%
#SECTION:-------------------- setup--------------------------------------------------------------------------------------
if __name__ == '__main__':
  import json
  from fran.managers.project import Project
  from pathlib import Path
  import pandas as pd
  import wandb

  entity = "drubashir"
  project_ti = "kits"                # W&B project name
  project = Project(project_ti)

# %%

# %%
  run_id = "KITS-0018"            # run id/name from URL
  out_dir = Path("wandb_case_recorder_tables")
  out_dir.mkdir(exist_ok=True, parents=True)

  api = wandb.Api()
  run = api.run(f"{entity}/{project_ti}/{run_id}")

# %%
  seen = 0
  for row in run.scan_history():
      for k, v in row.items():
          if not k.startswith("case_recorder/"):
              continue
          if not isinstance(v, dict) or v.get("_type") != "table-file":
              continue

          tr()
          # W&B file path like media/table/....table.json
          fpath = v["path"]
          local = run.file(fpath).download(root=out_dir, replace=True).name

          table_json = json.loads(Path(local).read_text())
          df = pd.DataFrame(table_json["data"], columns=table_json["columns"])

          safe_key = k.replace("/", "__")
          df.to_csv(out_dir / f"{safe_key}__step_{row.get('_step','na')}.csv", index=False)
          seen += 1
