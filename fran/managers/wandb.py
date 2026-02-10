# wandb_manager.py

import os
from typing import Any, Optional

from lightning.pytorch.loggers import WandbLogger
import wandb
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import wandb


@dataclass
class WandbRunRef:
    entity: str
    project: str
    run_id: str

    @property
    def path(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"


class WandbManagerUtils:
    @staticmethod
    def load_run(*, entity: str, project: str, run_id: str) -> wandb.apis.public.Run:
        api = wandb.Api()
        return api.run(f"{entity}/{project}/{run_id}")

    @staticmethod
    def id_exists(*, entity: str, project: str, run_id: str) -> bool:
        api = wandb.Api()
        try:
            api.run(f"{entity}/{project}/{run_id}")
            return True
        except wandb.errors.CommError:
            return False

    @staticmethod
    def id_kmost_recent(
        *,
        entity: str,
        project: str,
        k: int = 1,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Return the k most recent run IDs by created_at descending.
        """
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}", filters=filters or {}, order="-created_at")
        out: List[str] = []
        for r in runs:
            out.append(r.id)
            if len(out) >= int(k):
                break
        return out

    @staticmethod
    def get_checkpoint_artifacts(
        *,
        entity: str,
        project: str,
        run_id: str,
        type_name: str = "model",
    ) -> List[wandb.apis.public.Artifact]:
        """
        Lists artifacts logged by a run. You only get anything here if you log ckpts as artifacts.
        """
        run = WandbManagerUtils.load_run(entity=entity, project=project, run_id=run_id)
        return list(run.logged_artifacts())  # can be large; filter below if needed

    @staticmethod
    def download_checkpoint_artifact(
        *,
        artifact_path: str,
        dst_dir: str | Path,
    ) -> Path:
        """
        artifact_path example:
          "ENTITY/PROJECT/ARTIFACT_NAME:version"
        Returns local directory containing artifact contents.
        """
        api = wandb.Api()
        art = api.artifact(artifact_path)
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        local_dir = Path(art.download(root=str(dst_dir)))
        return local_dir

class WandbManager(WandbLogger):
    def __init__(
        self,
        *,
        project,
        run_id: Optional[str] = None,
        prefix: str = "training",
        tags=None,
        description: str = "",
        log_model: bool = False,
        **wandb_init_kwargs: Any,
    ):
        # Minimal parity with your NeptuneManager signature:
        # - project: your Project instance (used for project_title and log folder)
        # - run_id: resume id (optional)
        # - tags/description: map to wandb init fields
        # - prefix: keep (you may or may not use it)
        self.project_obj = project
        tags = tags or []

        # If you store your WANDB key in a file/env, do it before this class is constructed:
        # os.environ["WANDB_API_KEY"] = ...
        # os.environ["WANDB_ENTITY"] = ...

        # W&B project name: use your project title to mirror Neptune "sys/name"
        wb_project = getattr(project, "project_title", "fran")

        # If run_id is provided, resume that run
        if run_id:
            wandb_init_kwargs.setdefault("id", run_id)
            wandb_init_kwargs.setdefault("resume", "allow")

        # Map description to "notes" (W&B convention)
        if description:
            wandb_init_kwargs.setdefault("notes", description)

        # Tags
        if tags:
            wandb_init_kwargs.setdefault("tags", tags)

        super().__init__(
            project=wb_project,
            name=wb_project if not run_id else None,
            save_dir=str(getattr(project, "log_folder", ".")),
            log_model=log_model,  # "all"/True/False depending on what you want
            **wandb_init_kwargs,
        )

    def log_hyperparams(self, params: Any) -> None:
        # Lightning calls this; keep it simple.
        self.experiment.config.update(params, allow_val_change=True)

    @property
    def wb_run(self):
        return self.experiment  # wandb.Run

    @property
    def run_id(self) -> str:
        return self.experiment.id

    @property
    def save_dir(self) -> Optional[str]:
        return str(getattr(self.project_obj, "checkpoints_parent_folder", None))

# %%
if __name__ == '__main__':
    from fran.managers.project import Project
    proj_nodes = Project(project_title="nodes")

    from lightning.pytorch.loggers import WandbLogger
    import wandb

    logger = WandbLogger(
        project="wandb-auth-toy",
        name="logger-only-test",
    )

    exp = logger.experiment
    exp.log({"acc": 0.9})
    exp.log({"acc": 0.95})

    print("Run ID:", exp.id)
    wandb.finish()
# %%
