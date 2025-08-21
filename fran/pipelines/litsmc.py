# %%

import argparse
import torch

from fran.managers.db import add_plan_to_db
from fran.trainers import Trainer
from monai.data.dataset import GDSDataset
from utilz.imageviewers import ImageMaskViewer
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.trainers import Trainer
from pathlib import Path
from fran.managers import Project
from fran.managers import Datasource, _DS
from fran.run.analyze_resample import PreprocessingManager
from fran.managers.data import DataManagerDual
from fran.utils.config_parsers import ConfigMaker


# %%
# %%
#SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- P = Project("nodes")
if __name__ == '__main__':
    torch.set_float32_matmul_precision("medium")
    from fran.utils.common import *
    P = Project("litsmc")
    P._create_plans_table()
    P.create("lits")

    P.add_data([_DS().litq, _DS().lits, _DS().drli, _DS().litqsmall])
    # P.create('litsmc')
    conf = ConfigMaker(P, raytune=False, configuration_filename=None).config
    plan = conf['plan_train']

    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    ss = """ALTER TABLE master_plans ADD COLUMN remapping"""
    cur.execute(ss)
# %%
#SECTION:-------------------- FINE-TUNING RUN--------------------------------------------------------------------------------------
    bs = 14  # is good if LBD with 2 samples per case
    compiled = False
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    neptune = True
    tags = []
    description = None

    # device_id = 1
    device_id = 0
# %%
    # conf["dataset_params"]["ds_type"] ='lmdb'
    # conf["dataset_params"]["cache_rate"] = None

    run_name=None
    Tm = Trainer(P.project_title, conf, run_name)
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
    Tm.fit()
# %%
# %%
#SECTION:-------------------- TS--------------------------------------------------------------------------------------

#    
    Tm.D.prepare_data()
    Tm.D.setup()
    Tm.D.train_manager.keys_tr
# %%
    tm = Tm.D.train_manager
    dici = tm.ds[0]

#SECTION:-------------------- Project creation--------------------------------------------------------------------------------------



    
# P.delete()
    DS = _DS()
    P.add_data([DS.litq,DS.lits,DS.drli,DS.litqsmall])

# P.add_data([DS.totalseg])
# %%
#SECTION:-------------------- DATA FOLDER H5PY file--------------------------------------------------------------------------------------

    test =False
    ds = Datasource(folder=Path("/s/xnat_shadow/nodes"), name="nodes", alias="nodes", test=test)
    ds.process()

# %%
#SECTION:-------------------- GLOBAL PROPERTIES--------------------------------------------------------------------------------------
    if not "labels_all" in P.global_properties.keys():
        P.set_lm_groups(plan["lm_groups"])
        P.maybe_store_projectwide_properties(overwrite=True)
# %%
#SECTION:-------------------- ANALYSE RESAMPLE------------------------------------------------------------------------------------  <CR>
    active_plan = "plan10"
    overwrite=False
# %%

    parser = argparse.ArgumentParser(description="Resampler")

    parser.add_argument("-t", help="project title", dest="project_title")
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        help="number of parallel processes",
        default=8,
    )
    parser.add_argument(
        "-r",
        "--clip-range",
        nargs="+",
        help="Give clip range to compute dataset std and mean",
    )
    parser.add_argument(
        "-m", "--mode", default="fgbg", help="Mode of Patch generator, 'fg' or 'fgbg'"
    )
    parser.add_argument(
        "-p",
        "--patch-size",
        nargs="+",
        default=[192, 192, 128],
        help="e.g., [192,192,128]if you want a high res patch-based dataset",
    )
    parser.add_argument(
        "-nf",
        "--no-fix",
        action="store_false",
        help="By default if img/mask sitk arrays mismatch in direction, orientation or spacing, FRAN tries to align them. Set this flag to disable",
    )
    parser.add_argument("-d", "--debug", action="store_true")

    args = parser.parse_known_args()[0]

    # args.num_processes = 1
    args.debug = True
    args.plan_name = active_plan
    args.project_title = "litsmc"



    plans = conf[args.plan_name]
#SECTION:-------------------- Initialize--------------------------------------------------------------------------------------
    I = PreprocessingManager(args)
    # I.spacing =
# %%
#SECTION:-------------------- Resampling --------------------------------------------------------------------------------------
    overwrite=False
    I.resample_dataset(overwrite=overwrite)
    I.R.get_tensor_folder_stats()

# %%
#SECTION:--------------------  Processing based on MODE ------------------------------------------------------------------
    if I.plan["mode"] == "patch":
        # I.generate_TSlabelboundeddataset("lungs","/s/fran_storage/predictions/totalseg/LITS-827")
        I.generate_hires_patches_dataset()
    elif I.plan["mode"] == "lbd":
        if "imported_folder" not in plans.keys():
            I.generate_lbd_dataset(overwrite=overwrite)
        else:
            I.generate_TSlabelboundeddataset(
                imported_labels=plans["imported_labels"],
                imported_folder=plans["imported_folder"],
                overwrite=overwrite, device="cuda")
# %%

    L = LabelBoundedDataGeneratorImported( project=P,
            plan = plan,
            folder_suffix=plan_name,
            imported_folder=imported_folder,
            merge_imported_labels=merge_imported_labels,
            remapping=remapping,
        )

# %%
    device="cpu"
    overwrite=True
    I.L = LabelBoundedDataGenerator(
        project=I.project,
        plan=I.plan,
        plan_name=I.plan_name,
    )
    I.L.setup(overwrite=overwrite,device=device)
    I.L.process()
# %%

# %%
#SECTION:-------------------- DATA MANAGER--------------------------------------------------------------------------------------

    batch_size = 8
    ds_type="lmdb"
    ds_type=None
    device = 0

    conf["dataset_params"]["mode"] = None
    conf["dataset_params"]["cache_rate"] = 0.5

    D = DataManagerDual(
        project_title=P.project_title,
        config=conf,
        batch_size=batch_size,
        ds_type=ds_type
    )

# %%
    D.prepare_data()
    D.setup()
    tm = D.train_manager
    tm = D.valid_manager
    tm.transforms_dict

# %%
    ds =tm.ds
    dat= ds[0]
    dici = ds.data[0]
    tm.tfms_list


# %%

# %%
    D.train_ds[0]
    dlt =D.train_dataloader()
    dlv =D.val_dataloader()
# %%

    # iteri = iter(dlv)
    for num,batch in enumerate(dlv):
        print(batch["image"].shape)
# %%

    for num,batch in enumerate(dlt):
        print(batch["image"].shape)
# %%
# %%
#SECTION:-------------------- TROUBLESHOOTING--------------------------------------------------------------------------------------
    
    Tm.D.prepare_data()
    Tm.D.setup()
    Tm.D.train_manager.keys_tr
# %%
    tm = Tm.D.train_manager
    dici = tm.ds[0]
    ds =   GDSDataset
# %%
    img = dici[0]['image']
    lm = dici['lm']


    im = dici[0]['image']
    lm =  dici[0]['lm']
    ImageMaskViewer([im[0], lm[0]])
# %%

    tv = Tm.D.valid_manager
    # dici = tv.ds[0]
    dl = Tm.D.val_dataloader()
    dlt = Tm.D.train_dataloader()
    iteri = iter(dlt)
    batch = next(iteri)

    batch['lm'].max()

# %%
    n=0
    lm = batch['lm']
    im = batch['image']
    im = im.permute(0,1,4,2,3)
    lm = lm.permute(0,1,4,2,3)
    ImageMaskViewer([im[n][0], lm[n][0]])
# %%


    batch_size = 2
    ds_type="lmdb"
    ds_type=None


    D = DataManagerDual(
        project_title=P.project_title,
        config=conf,
        batch_size=batch_size,
        ds_type=ds_type
    )

# %%
    D.prepare_data()
    D.setup()
    tm = D.train_manager
    tm = D.valid_manager
    tm.transforms_dict

# %%
    ds =tm.ds
    dat= ds[0]
    dici = ds.data[0]
    tm.tfms_list


# %%

# %%
    # D.train_ds[0]
    # dl =D.train_dataloader()
    dl =D.val_dataloader()
# %%

    iteri = iter(dl)
    batch = next(iteri)


# %%
    n=0
    im = batch['image'][n][0]
    ImageMaskViewer([im, batch['lm'][n][0]])
    # while iteri:
    #     print(batch['image'].shape)
#SECTION:-------------------- INFERENCE--------------------------------------------------------------------------------------

