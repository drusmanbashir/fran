from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import torch._dynamo
import wandb
from lightning.pytorch.loggers import WandbLogger
from paramiko import SSHClient

from fran.utils.common import common_vars_filename
from utilz.fileio import load_yaml, maybe_makedirs

try:
    import numpy as np
except Exception:
    np = None


torch._dynamo.config.suppress_errors = True


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
    api_token = commons.get("wandb_api_token") or os.environ.get("WANDB_API_KEY")
    entity = commons.get("wandb_entity") or os.environ.get("WANDB_ENTITY")
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
            width = int(os.environ.get("FRAN_WANDB_SEQ_WIDTH", "4"))
            resolved_run_id = _next_ordered_run_id(
                entity=self.entity,
                project_title=project.project_title,
                width=width,
            )
            wandb_init_kwargs.setdefault("id", resolved_run_id)
            name = resolved_run_id

        env_mode = os.environ.get("WANDB_MODE")
        wandb_init_kwargs.setdefault("mode", env_mode if env_mode else wb_mode)
        self._mode = str(wandb_init_kwargs.get("mode", wb_mode)).lower()

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

    def log_hyperparams(self, params):
        self.experiment.config.update(_to_plain(params), allow_val_change=True)

    @property
    def wb_run(self):
        return self.experiment

    @property
    def model_checkpoint(self):
        key = self._path("best_model_path")
        return self.experiment.summary.get(key)

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
        ckpt_path = self.model_checkpoint
        if not ckpt_path:
            print("No checkpoints in this run")
            return
        remote_dir = str(Path(ckpt_path).parent)
        latest_ckpt = self.shadow_remote_ckpts(remote_dir)
        if latest_ckpt:
            self.experiment.summary.update(
                {self._path("best_model_path"): str(latest_ckpt)}
            )

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
        if wandb.run is self.experiment:
            wandb.finish()

    @property
    def run_id(self):
        return self.experiment.id

    @property
    def save_dir(self) -> Optional[str]:
        return str(self.project.checkpoints_parent_folder)
