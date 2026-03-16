# %%
from __future__ import annotations
import ipdb
tr = ipdb.set_trace
from tqdm.auto import tqdm

import os
import re
import secrets
from pathlib import Path
from typing import Any, Optional

import torch._dynamo
import wandb
from lightning.pytorch.loggers import WandbLogger
from paramiko import SSHClient

from fran.utils.common import common_vars_filename
from utilz.fileio import load_yaml, maybe_makedirs
from utilz.random_word_maker import (
    logical_word_capacity,
    ordered_word_suffixes,
    random_pronounceable_suffix,
)

try:
    import numpy as np
except Exception:
    np = None


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
            print(f"Found {len(ckpt_files)} ckpts:\n {ckpt_files}")
            if local_dir_parent is None:
                return ckpt_files

            local_dir_parent = Path(local_dir_parent)
            existing_local_ckpts = {
                path.name: path for path in local_dir_parent.rglob("*.ckpt")
            }
            local_files = []
            for remote_file in tqdm(ckpt_files):
                rel = remote_file.split("/")[-1]
                existing_local = existing_local_ckpts.get(rel)
                if existing_local is not None:
                    print(
                        f"Warning: skipping {remote_file} because local checkpoint exists at {existing_local}"
                    )
                    continue
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

import stat
torch._dynamo.config.suppress_errors = True

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


def _flatten_for_wandb(payload: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in payload.items():
        leaf = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_for_wandb(value, leaf))
        else:
            flat[leaf] = _to_plain(value)
    return flat


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
    prefix = "".join(ch for ch in str(project_title).strip() if ch.isalnum())
    return prefix.upper() if prefix else "RUN"


def _extract_seq(value: str, prefix: str) -> Optional[int]:
    m = re.fullmatch(rf"{re.escape(prefix)}-(\d+)", str(value or ""))
    if not m:
        return None
    return int(m.group(1))


def _extract_word_suffix(value: str, prefix: str) -> Optional[str]:
    value = str(value or "")
    token = f"{prefix}-"
    if value.startswith(token):
        return value[len(token):]
    return None


def _new_run_id(entity: Optional[str], project_title: str, width: int = 4) -> str:
    prefix = _normalize_run_prefix(project_title)
    path = f"{entity}/{project_title}" if entity else project_title
    width = int(width or 5)
    mode = str(os.environ.get("WANDB_MODE", "")).lower()
    if mode in {"offline", "disabled", "dryrun"}:
        suffix = random_pronounceable_suffix(width)
        return f"{prefix}-{suffix}"
    used = set()
    used_suffixes = set()
    try:
        api = wandb.Api()
        for run in api.runs(path):
            for candidate in (run.id, run.name):
                if candidate:
                    candidate = str(candidate)
                    used.add(candidate)
                    suffix = _extract_word_suffix(candidate, prefix)
                    if suffix:
                        used_suffixes.add(suffix)
    except Exception:
        pass

    for suffix in ordered_word_suffixes():
        run_id = f"{prefix}-{suffix}"
        if run_id not in used and suffix not in used_suffixes:
            return run_id

    while True:
        suffix = random_pronounceable_suffix(width)
        run_id = f"{prefix}-{suffix}"
        if run_id not in used and suffix not in used_suffixes:
            return run_id


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
            width = int(os.environ.get("FRAN_WANDB_SEQ_WIDTH", "2"))
            resolved_run_id = _new_run_id(
                entity=self.entity,
                project_title=project.project_title,
                width=width,
            )
            wandb_init_kwargs.setdefault("id", resolved_run_id)
            name = resolved_run_id
        self._resolved_run_id = resolved_run_id

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
        params = _to_plain(params)
        if isinstance(params, dict):
            params = _flatten_for_wandb(params)
        self.experiment.config.update(params, allow_val_change=True)

    @property
    def wb_run(self):
        return self.experiment

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
        if wandb.run is self.experiment:
            wandb.finish()

    @property
    def run_id(self):
        return self.experiment.id or self._resolved_run_id

    @property
    def save_dir(self) -> Optional[str]:
        return str(self.project.checkpoints_parent_folder)


if __name__ == "__main__":
# %%
#SECTION:-------------------- --------------------------------------------------------------------------------------    

        from fran.managers.project import Project
        P = Project(project_title="kits2")
        _resolve_wandb_save_dir(P)
# %%
        W = WandbManager(project=P)
        os.environ["WANDB_MODE"] = "online"
        api= wandb.Api()
        runs = api.runs()
        aa = list(runs)
        print(_new_run_id("drubashir", "kits2"))
        df = W.fetch_project_df()
        width =4
        prefix= "kits2"
        aa = df['sys/name'].sort_values().iloc[-1]
        number = int(aa.split("-")[-1])
        max_seq = aa.split("-")[-1]
        max_seq = int(max_seq)
        id = f"{prefix}-{max_seq + 1:0{int(width)}d}"

# %%
