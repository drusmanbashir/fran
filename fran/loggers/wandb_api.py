
# fran/trackers/wandb_api.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import wandb


@dataclass(frozen=True)
class WandbRunRef:
    entity: str
    project: str
    run_id: str

    @property
    def path(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"


class WandbAPI:
    """
    Post-hoc / offline API utilities.
    No Lightning dependency.
    """

    def __init__(self):
        self.api = wandb.Api()

    def load_run(self, ref: WandbRunRef) -> wandb.apis.public.Run:
        return self.api.run(ref.path)

    def id_exists(self, ref: WandbRunRef) -> bool:
        try:
            self.api.run(ref.path)
            return True
        except wandb.errors.CommError:
            return False

    def most_recent_ids(
        self,
        *,
        entity: str,
        project: str,
        k: int = 1,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        runs = self.api.runs(f"{entity}/{project}", filters=filters or {}, order="-created_at")
        out: List[str] = []
        for r in runs:
            out.append(r.id)
            if len(out) >= int(k):
                break
        return out

    def runs_table(
        self,
        *,
        entity: str,
        project: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        runs = self.api.runs(f"{entity}/{project}", filters=filters or {})
        rows: List[Dict[str, Any]] = []
        for r in runs:
            row: Dict[str, Any] = {
                "id": r.id,
                "name": r.name,
                "state": r.state,
                "created_at": r.created_at,
                "url": r.url,
                "tags": list(getattr(r, "tags", []) or []),
            }
            for k, v in dict(r.config).items():
                if str(k).startswith("_"):
                    continue
                row[f"config/{k}"] = v
            for k, v in dict(r.summary).items():
                row[f"summary/{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def run_history_df(
        self,
        ref: WandbRunRef,
        *,
        keys: Optional[Iterable[str]] = None,
        samples: int = 10000,
    ) -> pd.DataFrame:
        r = self.load_run(ref)
        df = r.history(keys=list(keys) if keys else None, samples=samples, pandas=True)
        df.insert(0, "run_id", r.id)
        df.insert(1, "run_name", r.name)
        return df
