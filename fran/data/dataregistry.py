# %%
from __future__ import annotations

import ipdb

tr = ipdb.set_trace

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import ipdb
import yaml

if "XNAT_CONFIG_PATH" in os.environ:
    pass
# from fran.utils.common import COMMON_PATHS
DATASET_PATHS = os.environ["FRAN_COMMON_PATHS"] + "/datasets.yaml"


@dataclass(frozen=True)
class DatasetSpec:
    ds: str
    folder: Path
    alias: Optional[str] = None

    def __len__(self):
        images_folder = self.folder / "images"
        if images_folder.exists():
            return len(list(images_folder.glob("*")))
        return 0


class DatasetRegistry:
    def __init__(self, cfg_path: Path | None = None):
        with open(DATASET_PATHS, "r") as f:
            raw = yaml.safe_load(f) or {}
        base = raw.get("datasets", {})

        specs: Dict[str, DatasetSpec] = {}
        for name, d in base.items():
            ds = d.get("ds", name)
            fld = Path(os.path.expandvars(os.path.expanduser(d["folder"])))
            alias = d.get("alias")
            specs[name] = DatasetSpec(ds=ds, folder=fld, alias=alias)
        self._specs = specs

    def names(self):
        return self._specs.keys()

    def get(self, name: str) -> DatasetSpec:
        return self._specs[name]

    def __getitem__(self, name: str) -> DatasetSpec:
        return self.get(name)

    def __getattr__(self, name: str) -> DatasetSpec:
        if name in self._specs:
            return self.get(name)
        raise AttributeError(f"DatasetRegistry has no attribute {name}")

    def __str__(self):
        star ="DataRegistry items: "+ ",".join(DS.names()) 
        return star

    def __repr__(self):
        return str(self)

DS = DatasetRegistry()

# %%
if __name__ == "__main__":
    pass
    
