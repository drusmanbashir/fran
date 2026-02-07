# %%
from lightning.pytorch.callbacks import BatchSizeFinder
from fran.callback.test import PeriodicTest
from fran.data.datasource import Datasource
from fran.data.dataregistry import DS
from fran.managers import  Project
from fran.run.analyze_resample import PreprocessingManager
from fran.trainers.trainer import Trainer
from fran.utils.common import *
from fran.configs.parser import ConfigMaker, confirm_plan_analyzed
import argparse

# %%
#SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- P = Project("nodes")
if __name__ == '__main__':
    from fran.utils.common import *
    set_autoreload()
    P = Project("totalseg")
    # P.add_data([DS.totalseg])
    C = ConfigMaker(P , configuration_filename=None)
    C.setup(2)
    C.plans
    conf = C.configs
    print(conf["model_params"])

    planT = conf['plan_train']
    planV = conf["plan_valid"]
    conf["plan_valid"] = conf["plan_train"].copy()
    pp(planT)

    planT['mode']
    # add_plan_to_db(plan,"/r/datasets/preprocessed/totalseg/lbd/spc_100_100_100_plan5",P.db)
    # if (lm==3).any():
    #     print("Bad values 3 ->0")
    #     lm[lm==3]=1
    #     torch.save(lm, bad_case_fn)
    #
    # find_matching_fn(Path(bad_names[0])[0],fixed, tags=["all"])
# %%
#SECTION:-------------------- COnfirm plans exist--------------------------------------------------------------------------------------

    statusesT    = confirm_plan_analyzed(P, planT)
    statusesV    = confirm_plan_analyzed(P, planV)
    print("Training:", statusesT, "Validation:", statusesV)
# %%
# SECTION:-------------------- TRAINING-------------------------------------------------------------------------------------- <CR> <CR> <CR> devices = 2
    devices= [1]
    bs = 8

    # run_name ='LITS-1285'
    compiled = False
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    neptune = True
    override_dm = False
    tags = []
    description = f"Partially trained up to 100 epochs"

    conf['plan_train']

    # cbs = [PeriodicTest(every_n_epochs=1,limit_batches=50), BatchSizeFinder(batch_arg_name="batch_size")]
    cbs =[]

    conf["dataset_params"]["cache_rate"]=0.0
    print(conf['model_params']['out_channels'])
    

    conf['dataset_params']['cache_rate']

# %%
    conf["dataset_params"]["fold"]=0
    run_name=None
    lr= 1e-2
# # %%
#     run_name="LITS-1327"
#     lr= 1e-3
    # lr=None
# %%
    Tm = Trainer(P.project_title, conf, run_name,)
    # Tm.configs
    Tm.configs['dataset_params']['fold']
# %%
    Tm.setup(
        compiled=compiled,
        periodic_test=0,
        batch_size=bs,
        devices=devices,
        cbs=cbs,
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
    Tm.configs['plan_train']['mode']
    Tm.configs['plan_train']['patch_size']
    Tm.configs['dataset_params']['fold']
    # Tm.D.configs = Tm.configs.copy()
    # Tm.D.batch_size=8
    Tm.N.compiled = compiled
# %%

    # tuner = Tuner(Tm.trainer)
    # tuner.scale_batch_size(Tm.N.model,mode="binsearch")
    Tm.fit()

# %%
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
    dici = D.train_ds[0]
# %%
    keys_tr = "L,E,F1,F2,Affine,ResizeW,N,IntensityTfms"
    keys = keys_tr.split(",")
    tm = D.train_manager
    tfms = tm.transforms_dict
    dat =  tm.data[0]

    dat2 = tfms[keys[0]](dat)
    dat3 = tfms[keys[1]](dat2)
    dat4 = tfms[keys[2]](dat3)
    dat5=tfms[keys[3]](dat4)
    dat6 = tfms[keys[4]](dat5)
    dat7 = tfms[keys[5]](dat6)
    dat8 = tfms[keys[6]](dat7)
    dat9 = tfms[keys[8]](dat8)
    


#SECTION:-------------------- Project creation--------------------------------------------------------------------------------------



    
# P.delete()
    DS = DS
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

    aa = folder_names_from_plan(P,plan)
    
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

    add_plan_to_db(P,Rs.plan, db_path=Rs.project.db, data_folder_source = Rs.output_folder)
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

