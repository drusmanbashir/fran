# %%
from __future__ import annotations


import warnings

import torch

from fran.configs.parser import ConfigMaker
from fran.managers import Project
SEQ_LEN = 16


from fran.twod.datamanagers import DataManagerMulti2



# %%
# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
if __name__ == "__main__":

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")
    project_title = "litsmc"
    torch.set_float32_matmul_precision("medium")
    proj_litsmc = Project(project_title=project_title)

    C = ConfigMaker(proj_litsmc)
    C.setup(1)
    conf_litsmc = C.configs
# %%

    conf_litsmc["plan_train"]["patch_size"] = [256, 256,SEQ_LEN]
    batch_size = 8
    ds_type = "lmdb"
    from pathlib import Path

    data_fldr = Path("/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150_ex070/slices")
# %%

    D = DataManagerMulti2(
        project_title=proj_litsmc.project_title,
        configs=conf_litsmc,
        batch_size=batch_size,
        ds_type=ds_type,
        data_folder=None,
    )
# %%
    D.prepare_data()
    D.configs["plan_train"]
    D.setup()
# %%
    dl =D.train_dataloader()

    iteri = iter(dl)
    batch = next(iteri)
# %%
