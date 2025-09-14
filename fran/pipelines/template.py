# %%

from fran.managers.db import *
import torch
import argparse
from pathlib import Path
from fran.trainers import Trainer

from fran.managers import Project, Datasource, DS
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.run.analyze_resample import PreprocessingManager
from fran.utils.config_parsers import ConfigMaker
from fran.utils.folder_names import folder_names_from_plan


# %%
#SECTION:-------------------- Project creation--------------------------------------------------------------------------------------



if __name__ == '__main__':
    from fran.utils.common import *
    
    P = Project("litstmp")
    P.create(mnemonic="litsmall")

    P.add_data([DS['litsmall']])
    P.set_labels_all()
    P.maybe_store_projectwide_properties(overwrite=True)
# %%

    C = ConfigMaker(P, raytune=False, configuration_filename=None)
    C.setup(7)
    C.plans
# %%
    conf = C.configs
    plan = conf['plan_train']
    print(conf["model_params"])

    plan = conf["plan_train"]
# P.add_data([DS.totalseg])
# %%
#SECTION:-------------------- FINE-TUNING RUN--------------------------------------------------------------------------------------
    bs = 8# is good if LBD with 2 samples per case
    compiled = True
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    neptune = True
    tags = []
    description = None

    # device_id = 1
    devices = [0]
    devices = 2


    
    # conf["dataset_params"]["ds_type"] ='lmdb'
    # conf["dataset_params"]["cache_rate"] = None
    matching_plan = find_matching_plan(P.db,plan)
    pp(matching_plan)
# %%
    run_name=None
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
    P.get_train_val_files(
                conf["dataset_params"]["fold"], plan['datasources']
            )
# %%
    # Tm.D.prepare_data()
    # Tm.D.setup()
    # Tm.D.batch_size=8
    Tm.N.compiled = compiled
    Tm.fit()
# %%

    train_cases, valid_cases = P.get_train_val_files(
                conf["dataset_params"]["fold"], plan['datasources']
            )
# %%
#SECTION:-------------------- DATA FOLDER H5PY file--------------------------------------------------------------------------------------

    test =False
    ds = Datasource(folder=Path("/s/datasets_bkp/litstmp"), name="litstmp", alias="tmp", test=test)
    ds.process()
# %%
#SECTION:-------------------- ANALYSE RESAMPE------------------------------------------------------------------------------------  <CR>

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
    args.plan_num = 11
# %%
    args.plan = plan
    args.project_title = "litstmp"

    plan = conf["plan10"]
    plan["remapping_train"] = {1:0}
# %%
    if not "labels_all" in P.global_properties.keys():
        P.set_lm_groups(plan["lm_groups"])
        P.maybe_store_projectwide_properties(overwrite=True)

# %%
#SECTION:-------------------- Initialize--------------------------------------------------------------------------------------
    I = PreprocessingManager(args)
    # I.spacing =
# %%
#SECTION:-------------------- Resampling --------------------------------------------------------------------------------------
    overwrite=True
    I.resample_dataset(overwrite=overwrite)
    I.R.get_tensor_folder_stats()

# %%
#SECTION:--------------------  Processing based on MODE ------------------------------------------------------------------
    overwrite=False
    if I.plan["mode"] == "patch":
        # I.generate_TSlabelboundeddataset("lungs","/s/fran_storage/predictions/totalseg/LITS-827")
        I.generate_hires_patches_dataset()
    elif I.plan["mode"] == "lbd":
        imported_folder=plan.get("imported_folder")
        if imported_folder is None:
            I.generate_lbd_dataset(overwrite=overwrite)
        else:
            I.generate_TSlabelboundeddataset(
                imported_labels=plan["imported_labels"],
                imported_folder=plan["imported_folder"],)
# %%
# %%
#SECTION:-------------------- troubleshoot--------------------------------------------------------------------------------------


    plan_name = args.plan_name
    L = LabelBoundedDataGenerator( project=P,
            plan = plan,
            plan_name=plan_name,
            
        )

# %%
    overwrite=True
    L.setup(overwrite=overwrite)
    L.process()

    add_plan_to_db(P,L.plan,L.output_folder, db_path="plans.db")
# %%

    
    img_file, lm_file = L.image_files[0], L.lm_files[0]
    img = torch.load(img_file,weights_only=False)
    # Load and process single case
    data = {
        "image": img_file,
        "lm": lm_file,
    }

    print(L.tfms_keys)
    fn = "/s/fran_storage/datasets/preprocessed/fixed_spacing/nodes/spc_080_080_150/images/nodes_21b_20210202_Thorax0p75I70f3.pt"
    t2 = torch.load(fn,weights_only=False)
    print(t2.meta.keys())
# %%
    # Apply transforms
    data = L.transforms(data)
    data2 = L.transforms_dict["LT"](data)
    
    data3 = L.transforms_dict["E"](data2)
    data4 = L.transforms_dict["D"](data3)
    data5 = L.transforms_dict["C"](data4)
    data6 = L.transforms_dict["Ind"](data5)

    print(data4['lm'].max())
    print(data5['lm'].max())
# %%
    print(lm_file)
    ttt = torch.load(img_file,weights_only=False)
    print(ttt.meta.keys())
# %%
    # Get metadata and indices
    fg_indices = L.get_foreground_indices(data["lm"])
    bg_indices = L.get_background_indices(data["lm"])
    coords = L.get_foreground_coords(data["lm"])
    
    # Process the case
    L.process_single_case(
        data["image"],
        data["lm"],
        fg_indices,
        bg_indices,
        coords["start"],
        coords["end"]
            )
# S
# %%

    add_plan_to_db(I.L.plan, data_folder_lbd = I.L.output_folder, db_path=I.L.project.db)
# %%
