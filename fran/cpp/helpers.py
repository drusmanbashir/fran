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


def cases_in_folder(fldr) -> int:
    fldr = Path(fldr)
    if not fldr.exists():
        return 0
    img_fldr = fldr / ("images")
    cases = list(img_fldr.glob("*"))
    n_cases = len(cases)
    return n_cases


# %%
if __name__ == "__main__":
    proj, cfg = load_project_cfg("lidc")
    cfg.add_preprocess_status()
    print(cfg.plans["preprocessed"])
# %%
