# %%
import argparse
import pprint as pp
from pathlib import Path

from utilz.imageviewers import ImageMaskViewer

from fran.managers import _DS, Datasource, Project
from fran.managers.data import DataManagerDual
from fran.managers.db import add_plan_to_db, find_matching_plan
from fran.preprocessing.imported import LabelBoundedDataGeneratorImported
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.run.analyze_resample import PreprocessingManager
from fran.trainers import Trainer
from fran.utils.config_parsers import ConfigMaker

# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- P = Project("nodes") <CR>
if __name__ == "__main__":
    from fran.utils.common import *

    P = Project("nodes")
    # P._create_plan_table()
    C= ConfigMaker(P, raytune=False, configuration_filename=None)
    C.setup()
    conf = C.configs
    plan = conf["plan_train"]
    # add_plan_to_db(plan,,P.db)
# %%
# SECTION:-------------------- FINE-TUNING RUN-------------------------------------------------------------------------------------- <CR>
    run_nodes = "LITS-1230"
    lr = 1e-3
    bs = 5  # is good if LBD with 2 samples per case
    compiled = False
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    neptune = True
    tags = []
    description = f"Partially trained up to 100 epochs"

    device_id = 1
    # device_id = 0
    conf["dataset_params"]["cache_rate"] = 0
    conf["dataset_params"]["ds_type"] 

# %%
    run_name = run_nodes
    run_name = None
    Tm = Trainer(P.project_title, conf, run_name)
    conf["dataset_params"]
# %%
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=[device_id],
        lr=lr,
        epochs=900 if profiler == False else 1,
        batchsize_finder=batch_finder,
        profiler=profiler,
        neptune=neptune,
        tags=tags,
        description=description,
    )
# %%
    add_plan_to_db(plan,"/r/datasets/preprocessed/nodes/lbd/spc_080_080_150_plan2",P.db)

    # Tm.D.batch_size=8
    Tm.N.compiled = compiled
    Tm.fit()
# %%
    N = Tm.N
    Tm.D.setup()
    Tm.D.prepare_data()
    dl = Tm.D.val_dataloader()
    batch = next(iter(dl))

    image = batch["image"]
    pred = N(image)
# %%

# SECTION:-------------------- Project creation-------------------------------------------------------------------------------------- <CR>

    # P.delete()
    DS = _DS()
    P.add_data([DS.nodes, DS.nodesthick])
    # P.add_data([DS.totalseg])
# %%
# SECTION:-------------------- DATA FOLDER H5PY file-------------------------------------------------------------------------------------- <CR>

    test = False
    ds = Datasource(
        folder=Path("/s/xnat_shadow/nodes"), name="nodes", alias="nodes", test=test
    )
    ds.process()
# %%

# SECTION:-------------------- ANALYSE RESAMPLE------------------------------------------------------------------------------------  <CR> <CR>

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
    args.plan_name = "plan2"
    args.project_title = "nodes"

    plan = conf[args.plan_name]
# SECTION:-------------------- Initialize-------------------------------------------------------------------------------------- <CR>
# %%
    I = PreprocessingManager(args)
    # I.spacing =
# %%
# SECTION:-------------------- Resampling -------------------------------------------------------------------------------------- <CR>
    overwrite = True
    I.resample_dataset(overwrite=overwrite)
    I.R.get_tensor_folder_stats()

# %%
# SECTION:--------------------  Processing based on MODE ------------------------------------------------------------------ <CR>
    overwrite = True
    if I.plan["mode"] == "patch":
        # I.generate_TSlabelboundeddataset("lungs","/s/fran_storage/predictions/totalseg/LITS-827")
        I.generate_hires_patches_dataset()
    elif I.plan["mode"] == "lbd":
        if "imported_folder" not in plan.keys():
            I.generate_lbd_dataset(overwrite=overwrite)
        else:
            I.generate_TSlabelboundeddataset(
                imported_labels=plan["imported_labels"],
                imported_folder=plan["imported_folder"],
                overwrite=overwrite,
            )
# %%
    L = LabelBoundedDataGenerator(project=I.project, plan=I.plan, plan_name=I.plan_name)
# %%

    L = LabelBoundedDataGeneratorImported(
        project=P,
        plan=plan,
        folder_suffix=plan_name,
        imported_folder=imported_folder,
        merge_imported_labels=merge_imported_labels,
        remapping=remapping,
    )

# %%
    overwrite = True
    L.setup(overwrite=overwrite)
    L.process()

# %%
# SECTION:-------------------- DATA MANAGER-------------------------------------------------------------------------------------- <CR>

    batch_size = 10
    ds_type = None
    ds_type = "lmdb"

    conf["dataset_params"]["mode"] = None
    conf["dataset_params"]["cache_rate"] = 0.5

    D = DataManagerDual(
        project_title=P.project_title,
        config=conf,
        batch_size=batch_size,
        ds_type=ds_type,
    )

# %%
    D.prepare_data()
    D.setup()
    tm = D.train_manager
    tm = D.valid_manager
    tm.transforms_dict

# %%
    ds = tm.ds
    dat = ds[0]
    dici = ds.data[0]
    tm.tfms_list

# %%

# %%
    D.train_ds[0]
    dlt = D.train_dataloader()
    dlv = D.val_dataloader()
# %%

    # iteri = iter(dlv)
    for num, batch in enumerate(dlv):
        print(batch["image"].shape)
# %%

    for num, batch in enumerate(dlt):
        print(batch["image"].shape)
# %%
    #
    Tm.D.prepare_data()
    Tm.D.setup()
    Tm.D.train_manager.keys_tr
# %%
    tm = Tm.D.train_manager
    dici = tm.ds[0]
# %%
    img = dici["image"]
    lm = dici["lm"]

    im = dici[0]["image"]
    lm = dici[0]["lm"]
    ImageMaskViewer([im[0], lm[0]])
# %%

    tv = Tm.D.valid_manager
    # dici = tv.ds[0]
    dl = Tm.D.val_dataloader()
    dlt = Tm.D.train_dataloader()
    iteri = iter(dlt)
    batch = next(iteri)

    batch["lm"].max()

# %%
    n = 0
    lm = batch["lm"]
    im = batch["image"]
    im = im.permute(0, 1, 4, 2, 3)
    lm = lm.permute(0, 1, 4, 2, 3)
    ImageMaskViewer([im[n][0], lm[n][0]])
# %%
# SECTION:-------------------- TROUBLE-------------------------------------------------------------------------------------- <CR>

    batch_size = 2
    ds_type = "lmdb"
    ds_type = None

    D = DataManagerDual(
        project_title=P.project_title,
        config=conf,
        batch_size=batch_size,
        ds_type=ds_type,
    )

# %%
    D.prepare_data()
    D.setup()
    tm = D.train_manager
    tm = D.valid_manager
    tm.transforms_dict

# %%
    ds = tm.ds
    dat = ds[0]
    dici = ds.data[0]
    tm.tfms_list

# %%

# %%
    # D.train_ds[0]
    # dl =D.train_dataloader()
    dl = D.val_dataloader()
# %%

    iteri = iter(dl)
    batch = next(iteri)

# %%
    n = 0
    im = batch["image"][n][0]
    ImageMaskViewer([im, batch["lm"][n][0]])
    # while iteri:
    #     print(batch['image'].shape)
# %%

    find_matching_plan(P.db, plan)
    add_plan_to_db(
        plan, "/r/datasets/preprocessed/nodes/lbd/spc_080_080_150_plan2", P.db
    )
# %%
# %%
# SECTION:-------------------- INFERENCE-------------------------------------------------------------------------------------- <CR>
    dl = Tm.D.val_dataloader()
    iteri = iter(dl)
    batch = next(iteri)
    img = batch["image"]
    preds = Tm.N(batch["image"])
    [print(a.shape) for a in preds]
    pp = preds[0]
    n = 1

    im = img[n][0].detach().cpu()
    lm = pp[n][0].detach().cpu()
    ImageMaskViewer([im, lm])
