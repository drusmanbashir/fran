## %%

from fran.managers import  Project
from fran.managers.db import COLUMNS_CRITICAL,  find_matching_plan
from fran.run.analyze_resample import PreprocessingManager
from fran.trainers.trainer import Trainer
from fran.utils.common import *
from fran.utils.config_parsers import ConfigMaker
import argparse
#SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- P = Project("nodes")
if __name__ == '__main__':
    from fran.utils.common import *
    P = Project("nodes")

    # P.add_data([DS.totalseg])
    C = ConfigMaker(P, raytune=False, configuration_filename=None)
    C.setup(7)
    C.plans
    conf = C.configs
    print(conf["model_params"])

    plan = conf['plan_train']
    pp(plan)

    # plan['mode']
    # add_plan_to_db(plan,"/r/datasets/preprocessed/totalseg/lbd/spc_100_100_100_plan5",P.db)


# %%
# SECTION:-------------------- TRAINING-------------------------------------------------------------------------------------- <CR> <CR> <CR> devices = 2
    devices= [1]
    bs = 4

    run_name ='LITS-1290'
    run_name =None
    compiled = True
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    neptune = True
    tags = []
    description = f"Partially trained up to 100 epochs"

    conf['plan_train']


    conf["dataset_params"]["cache_rate"]=0.0
    conf["dataset_params"]["fold"]=0
    print(conf['model_params']['out_channels'])
    

    conf['dataset_params']['cache_rate']

# %%
    Tm = Trainer(P.project_title, conf, run_name)
# %%
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=devices,
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
    matching_plan = folder_names_from_plan(P,plan)

    matching_plan

# %%
    plan = {k: plan.get(k) for k in COLUMNS_CRITICAL}  # align to fixed schema
    keys = [k for k in COLUMNS_CRITICAL if k in plan]
    conds, params = [], []
# %%
    for k in keys:
        v = _normalize_for_db(plan[k])
        if v is None:
            conds.append(f'"{k}" IS NULL')
        else:
            conds.append(f'"{k}" = ?')
            params.append(v)
# %%
    sql = (
        f'SELECT id, data_folder_source, data_folder_lbd, data_folder_whole, data_folder_patch FROM "{TABLE}" WHERE '
        + " AND ".join(conds)
        + " LIMIT 1"
    )
    db_path =P.db
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(sql, params).fetchone()
# %%

    row_out = {
        "id": row[0],
        "data_folder_source": row[1],
        "data_folder_lbd": row[2],
        "data_folder_whole": row[3],
        "data_folder_patch": row[4],
    }

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
    DS = DS
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

    folder_names_from_plan(P, plan)
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
