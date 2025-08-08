
# %%
import warnings
from fran.inference.cascade import WholeImageInferer
from fran.managers import  Project
import torch
from fran.utils.common import *
from fran.trainers.trainer import Trainer
from fran.utils.config_parsers import ConfigMaker

# %%
#SECTION:-------------------- TRAINING--------------------------------------------------------------------------------------
if __name__ == '__main__':

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")


    proj_nodes = Project(project_title="nodes")
    proj_tsl = Project(project_title="totalseg")
    proj_litsmc = Project(project_title="litsmc")
    conf_litsmc = ConfigMaker(proj_litsmc, raytune=False).config
    conf_nodes = ConfigMaker(proj_nodes, raytune=False).config
    conf_tsl = ConfigMaker(proj_tsl, raytune=False).config

    # conf['model_params']['lr']=1e-3
    conf_litsmc["dataset_params"]["cache_rate"]
    # run_name = "LITS-1007"
    # device_id = 1
    device_id = 0
    run_none = None
    run_tsl = "LITS-1120"
    run_nodes = "LITS-1110"
    run_litsmc = "LITS-1131"
    bs = 10  # is good if LBD with 2 samples per case
    # run_name ='LITS-1003'
    compiled = False
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    neptune = True
    tags = []
    description = f"Partially trained up to 100 epochs"
# %%
# SECTION:-------------------- TOTALSEG TRAINING-------------------------------------------------------------------------------------- <CR> <CR> <CR>
    run_name = run_litsmc
    run_name = run_tsl
    conf = conf_litsmc
    conf = conf_tsl
    proj = "litsmc"
    proj = "totalseg"
# %%
    Tm = Trainer(proj, conf, run_name)
# %%
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=[device_id],
        epochs=600 if profiler == False else 1,
        batchsize_finder=batch_finder,
        profiler=profiler,
        neptune=neptune,
        tags=tags,
        description=description,
    )
# %%
    # Tm.D.batch_size=8
    Tm.N.compiled = compiled
# %%
    Tm.fit()
    # model(inputs)
# %%
# S
# %%
#SECTION:-------------------- INFERENCE Whole Image--------------------------------------------------------------------------------------

    safe_mode = False
    run_tot = ["LITS-1088"]
    W = WholeImageInferer(
        run_tot[0], safe_mode=safe_mode, k_largest=None, save_channels=False
    )
# %%

    nodesthick_imgs = list(nodesthick_fldr.glob("*"))
    nodes_imgs = list(nodes_fldr.glob("*"))
    preds = W.run(nodes_imgs, chunksize=1, overwrite=False)
    p = preds[0]["pred"][0]

# %%
# %%

# %%
# S
