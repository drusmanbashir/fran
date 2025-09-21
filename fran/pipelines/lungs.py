# %%
import argparse

from utilz.imageviewers import ImageMaskViewer
from fran.trainers import Trainer
from pathlib import Path
from fran.managers import Project
from fran.data.datasource import Datasource
from fran.run.analyze_resample import PreprocessingManager
from fran.managers.data import DataManagerDual
from fran.utils.config_parsers import ConfigMaker


# %%
# %%
#SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- P = Project("nodes")
if __name__ == '__main__':
    from fran.utils.common import *
    P = Project("lidc2")


    # P._create_plans_table()
    P.add_data([DS.lidc2])
    C = ConfigMaker(P, raytune=False, configuration_filename=None)
    C.setup(3)
    C.plans
    conf = C.configs
    print(conf["model_params"])

    plan = conf['plan_train']
    pp(plan)


# %%
#SECTION:-------------------- Project creation--------------------------------------------------------------------------------------



    
# P.delete()
    DS = DS
    P.add_data([DS.nodes,DS.nodesthick])
# P.add_data([DS.totalseg])
# %%
#SECTION:-------------------- DATA FOLDER H5PY file--------------------------------------------------------------------------------------

    test =False
    ds = Datasource(folder=Path("/s/xnat_shadow/nodes"), name="nodes", alias="nodes", test=test)
    ds.process()
# %%
#SECTION:-------------------- ANALYSE RESAMPLE------------------------------------------------------------------------------------  <CR>


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
    args.plan_num = 2
    args.plan = plan
    args.project_title = "lidc2"

# %%
    if not "labels_all" in P.global_properties.keys():
        P.set_lm_groups(plan["lm_groups"])
        P.maybe_store_projectwide_properties(overwrite=True)
#SECTION:-------------------- Resampling --------------------------------------------------------------------------------------
    I = PreprocessingManager(args)
    I.resample_dataset(overwrite=False)
    I.R.get_tensor_folder_stats()

# %%
#SECTION:--------------------  Processing based on MODE ------------------------------------------------------------------


    overwrite=False
    I.plan_name= "jj"
    if I.plan["mode"] == "patch":
        # I.generate_TSlabelboundeddataset("lungs","/s/fran_storage/predictions/totalseg/LITS-827")
        I.generate_hires_patches_dataset()
    elif I.plan["mode"] == "lbd":
        if plan['imported_folder'] is None:
            I.generate_lbd_dataset(overwrite=overwrite)
        else:
            I.generate_TSlabelboundeddataset(
                imported_labels=plan["imported_labels"],
                imported_folder=plan["imported_folder"],)
# %%

    L = LabelBoundedDataGeneratorImported( project=P,
            plan = plan,
            folder_suffix=plan_name,
            imported_folder=imported_folder,
            merge_imported_labels=merge_imported_labels,
            remapping=remapping,
        )

# %%
    overwrite=True
    L.setup(overwrite=overwrite)
    L.process()

# %%
#SECTION:-------------------- DATA MANAGER--------------------------------------------------------------------------------------

    batch_size = 10
    ds_type=None
    ds_type="lmdb"

    conf["dataset_params"]["mode"] = None
    conf["dataset_params"]["cache_rate"] = 0

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
#SECTION:-------------------- FINE-TUNING RUN--------------------------------------------------------------------------------------
# %%
    run_nodes = "LITS-1110"
    bs = 10  # is good if LBD with 2 samples per case
    compiled = False
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    neptune = True
    tags = []
    description = f"Partially trained up to 100 epochs"

    # device_id = 1
    device_id = 0
# %%
    conf["dataset_params"]["cache_rate"] = 0
    conf["dataset_params"]["ds_type"] ='lmdb'

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
    
    Tm.D.prepare_data()
    Tm.D.setup()
    Tm.D.train_manager.keys_tr
# %%
    tm = Tm.D.train_manager
    dici = tm.ds[0]
# %%
    img = dici['image']
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
#SECTION:-------------------- TROUBLE--------------------------------------------------------------------------------------

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

# %%
#SECTION:-------------------- Imported dataset--------------------------------------------------------------------------------------
