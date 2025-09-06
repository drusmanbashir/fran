# %%

from fran.managers.db import add_plan_to_db, find_matching_plan
from fran.preprocessing.fixed_spacing import ResampleDatasetniftiToTorch
from fran.run.analyze_resample import PreprocessingManager
import argparse
from fran.inference.cascade import WholeImageInferer
from fran.managers import  Project
from fran.utils.common import *
from fran.trainers.trainer import Trainer
from fran.utils.config_parsers import ConfigMaker
# %%
#SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- P = Project("nodes")
if __name__ == '__main__':
    from fran.utils.common import *
    P = Project("totalseg")

    # P._create_plans_table()
    # P.add_data([_DS().totalseg])
    C = ConfigMaker(P, raytune=False, configuration_filename=None)
    C.setup(6)
    C.plans
    conf = C.configs
    print(conf["model_params"])

    plan = conf['plan_train']
    pp(plan)
    # plan['mode']
    # add_plan_to_db(plan,"/r/datasets/preprocessed/totalseg/lbd/spc_100_100_100_plan5",P.db)

# %%

# SECTION:-------------------- TOTALSEG TRAINING-------------------------------------------------------------------------------------- <CR> <CR> <CR>
    device_id = [1]
    devices = 1
    bs = 4

    run_name =None
    run_name ='LITS-1271'
    compiled = True
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    neptune = True
    tags = []
    description = f"Partially trained up to 100 epochs"

    conf['plan_train']

    conf["dataset_params"]["cache_rate"]=0.0
    print(conf['model_params']['out_channels'])
    
    conf['dataset_params']['cache_rate']
# %%
    Tm = Trainer(P.project_title, conf, run_name)
# %%
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=device_id,
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
#SECTION:-------------------- TROUBLESHOOTING--------------------------------------------------------------------------------------
# %%
    D = Tm.D
    D.prepare_data()
    D.setup()
    dlt = D.train_dataloader()

# %%
    for batch in dlt:
        print(batch['image'].shape)
        # break
# %%
    iteri = iter(dlt)
    batch = next(iteri)
    batch['image'].shape
    # model(inputs)
# %%
#SECTION:-------------------- Project creation--------------------------------------------------------------------------------------



    
# P.delete()
    DS = _DS()
    P.add_data([DS.totalseg])
# %%

#SECTION:-------------------- GLOBAL PROPERTIES--------------------------------------------------------------------------------------
    if not "labels_all" in P.global_properties.keys():
        P.set_lm_groups(plan["lm_groups"])
        P.maybe_store_projectwide_properties(overwrite=False)
# %%
#ECTION:-------------------- ANALYSE RESAMPLE------------------------------------------------------------------------------------  <CR>

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
    parser.add_argument("--plan", type=int, default=1)
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
    args.plan = 6
    args.project_title = "totalseg"



    C._set_active_plans(6,6)
    plan = conf['plan_train']

    pp(plan)
#SECTION:-------------------- Initialize--------------------------------------------------------------------------------------

    aa = find_matching_plan(P.db,plan)
    
    I = PreprocessingManager(args)

# %%
    Rs = ResampleDatasetniftiToTorch(
        P,
        plan,
        data_folder=P.raw_data_folder,

    )
# %%
    overwrite=False
    Rs.setup(overwrite=overwrite)
    Rs.process()

    add_plan_to_db(Rs.plan, db_path=Rs.project.db, data_folder_source = Rs.output_folder)
    # I.spacing =
# %%
#SECTION:-------------------- Resampling --------------------------------------------------------------------------------------
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
# %%
#SECTION:-------------------- TRAINING--------------------------------------------------------------------------------------
if __name__ == '__main__':

# %%
# %%
#SECTION:-------------------- TS--------------------------------------------------------------------------------------
    I.R = ResampleDatasetniftiToTorch(
        project=I.project,
        spacing=I.plan["spacing"],
        data_folder=I.project.raw_data_folder,
    )
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

