# %%
from pathlib import Path

from fran.configs.parser import ConfigMaker
from fran.managers.project import Project


def load_project_cfg(project_name: str, mnemonic: str | None = None):
    """
    High-level helper for C++:
    - builds Project
    - builds ConfigMaker
    - runs setup
    - returns whatever C++ needs
    """
    proj = Project(project_name)
    cfg = ConfigMaker(proj, configuration_filename=None)

    return proj, cfg
# %%
if __name__ == "__main__":

    proj, cfg = load_project_cfg("nodes")
    cfg.add_preprocess_status()
    print(cfg.plans["preprocessed"])
# %%
