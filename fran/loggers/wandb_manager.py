from __future__ import annotations

from typing import Any, Optional

from lightning.pytorch.loggers import WandbLogger


class WandbManager(WandbLogger):
    """
    Training-time logger only (Lightning-owned).
    No querying, no downloads, no SSH.
    """

    def __init__(
        self,
        *,
        project,
        run_id: Optional[str] = None,
        tags=None,
        description: str = "",
        log_model: bool = False,
        **wandb_init_kwargs: Any,
    ):
        tags = tags or []
        wb_project = getattr(project, "project_title", "fran")
        save_dir = str(getattr(project, "log_folder", "."))

        if run_id:
            wandb_init_kwargs.setdefault("id", run_id)
            wandb_init_kwargs.setdefault("resume", "allow")

        if description:
            wandb_init_kwargs.setdefault("notes", description)

        if tags:
            wandb_init_kwargs.setdefault("tags", tags)

        super().__init__(
            project=wb_project,
            save_dir=save_dir,
            log_model=log_model,
            **wandb_init_kwargs,
        )
        self.project_obj = project

    @property
    def wb_run(self):
        return self.experiment  # wandb.Run

    @property
    def run_id(self) -> str:
        return self.experiment.id
