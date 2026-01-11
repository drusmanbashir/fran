# %%
from monai.data import GridPatchDataset, PatchIterd,DataLoader
from monai.inferers.inferer import SlidingWindowInferer
from fran.data.collate import grid_collated
from fran.managers import  Project
from fran.trainers.trainer import Trainer
from fran.utils.common import *
from fran.configs.parser import ConfigMaker
import torch

from utilz.imageviewers import ImageMaskViewer
from fran.configs.parser import ConfigMaker
from fran.managers.project import Project
from fran.managers.data.training import DataManager

# fran / project utils (already used elsewhere in your codebase)



def bs1_collated(batch):
    stacked = {}
    images, lms = [],[]
    for i,item in enumerate(batch):
        images.append(item['image'])
        lms.append(item['lm'])

    stacked['image'] = torch.stack(images, dim=0)
    stacked['lm'] = torch.stack(lms, dim=0)
    return stacked

# def bs1_collated(batch):
#     return batch[0]

class DataManagerFullScan(DataManager):
    def __init__(
        self,
        project,
        configs: dict,
        cache_rate=0,
        ds_type=None,
        save_hyperparameters=False,
        data_folder=None,
        device= "cuda:0",

    ):
        super().__init__(
            project=project,
            device= device,
            configs=configs,
            batch_size=1,
            keys="L,E,N,Remap",
            cache_rate=cache_rate,
            ds_type=ds_type,
            split="valid",
            save_hyperparameters=save_hyperparameters,
            data_folder=data_folder,
        )

    def set_collate_fn(self):
        self.collate_fn =bs1_collated
        # keys_val="L,N,Remap,Ld,E,ResizePC",

# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
if __name__ == '__main__':
    import torch,warnings

    warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision("medium")
    from fran.utils.common import *

    project_title = "litsmc"
    proj_litsmc = Project(project_title=project_title)

    CL = ConfigMaker(proj_litsmc, configuration_filename=None)
    CL.setup(1)
    config_litsmc = CL.configs

    project_title = "totalseg"
    proj_tot = Project(project_title=project_title)
    proj_nodes = Project(project_title="nodes")

    config_nodes = ConfigMaker(
        proj_nodes,
    ).configs
    configuration_filename = (
        "/s/fran_storage/projects/lits32/experiment_configs_wholeimage.xlsx"
    )
    configuration_filename = None

    CT = ConfigMaker(
        proj_tot,
    )
    CT.setup(6)
    config_tot = CT.configs

    CN = ConfigMaker(
        proj_nodes,
    )
    CN.setup(1)
    config_nodes = CN.configs

# %%
    D = DataManagerFullScan(project=proj_nodes,configs=config_nodes,cache_rate=0)
    D.prepare_data()
    D.setup()

    dl = D.dl
# %%
    itier = iter(dl)
    batch = next(itier)
    image = batch["image"]
# %%
    print(image.shape)
    lm = batch["lm"]
    ImageMaskViewer([image.detach().cpu(), lm.detach().cpu()])
    image.shape
# %%

    dici = D.ds[7]
# %%

# SECTION:-------------------- TRAINING-------------------------------------------------------------------------------------- <CR> <CR> <CR> devices = 2
    from lightning.pytorch.tuner import Tuner
    devices= [1]
    bs = 2

    # run_name ='LITS-1285'
    compiled = False
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    neptune = True
    override_dm = False
    tags = []
    description = f"Partially trained up to 100 epochs"
# %%
    run_name="LITS-1327"
    lr= 1e-3
    run_name = "LITS-1327"
    Tm = Trainer(proj_nodes.project_title, config_nodes, run_name)
    # Tm.configs
    Tm.configs['dataset_params']['fold']
# %%
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=devices,
        epochs=500 if profiler == False else 1,
        batchsize_finder=batch_finder,
        profiler=profiler,
        neptune=neptune,
        tags=tags,
        description=description,
        lr=lr,
        override_dm_checkpoint=override_dm
    )
# %%
    N= Tm.N

    ds=D.ds
# %%


    patch_iter = PatchIterd(
            keys=["image", "lm"], patch_size=Tm.configs["plan_valid"]["patch_size"], mode="constant"
        )

    dsG = GridPatchDataset(data=ds, patch_iter=patch_iter)

# %%
    dl2 = DataLoader(
        dsG,
        batch_size=bs,
        collate_fn=grid_collated,
    )
# %%

    iteri = iter(dl2)
    # while True:
# %%
    batch = next(iteri)
    print(batch['image'].shape)


# %%
    n =1
    im = batch["image"][n][0]
    lm = batch["lm"][n][0]
    ImageMaskViewer([im.detach().cpu(), lm.detach().cpu()])
# %%

    patch_overlap = 0.1
    mode = "constant"
    device= "cpu"
    sw_device="cuda"
    inferer = SlidingWindowInferer(
        roi_size=Tm.configs["plan_train"]["patch_size"],
        sw_batch_size=2,
        overlap=patch_overlap,
        mode=mode,
        progress=True,
        sw_device=sw_device,
        device=device)


# %%
    model = Tm.N.model.to("cuda")
    logits = inferer(inputs=image, network=Tm.N.model)  # [B,117,D,H,W]
# %%
