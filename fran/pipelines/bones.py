from fran.callback.test import PeriodicTest
from fran.managers import  Project
from fran.trainers.trainer_bk import TrainerBK
from fran.utils.common import *
from fran.configs.parser import ConfigMaker, confirm_plan_analyzed

#SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- P = Project("bones")
if __name__ == '__main__':
    set_autoreload()
    from fran.utils.common import *
    P = Project("bones")
    # P.add_data([DS.totalseg])
    C = ConfigMaker(P )
    C.setup(1)
    C.plans
    conf = C.configs
    print(conf["model_params"])

    planT = conf['plan_train']
    planV = conf["plan_valid"]
    pp(planT)

    print(planT['mode'])
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
# %%
# SECTION:-------------------- TRAINING-------------------------------------------------------------------------------------- <CR> <CR> <CR> devices = 2
    devices= [0]
    bs = 4

    # run_name ='LITS-1285'
    compiled = False
    profiler = False
    # NOTE: if Neptune = False, should store checkpoint locally
    batch_finder = False
    wandb = True
    override_dm = False
    tags = []
    description = f"Partially trained up to 100 epochs"

    conf['plan_train']

    cbs = [PeriodicTest(every_n_epochs=1,limit_batches=50)]

    conf["dataset_params"]["cache_rate"]=0.0
    print(conf['model_params']['out_channels'])
    

    conf['dataset_params']['cache_rate']

# %%
    conf["dataset_params"]["fold"]=0
    run_name="juwswjs6"
    lr= 1e-2
# # %%
#     run_name="LITS-1327"
#     lr= 1e-3
    # lr=None
# %%
    Tm = TrainerBK(P.project_title, conf, run_name,)
    # Tm.configs
    Tm.configs['dataset_params']['fold']
# %%
    Tm.setup(
        compiled=compiled,
        batch_size=bs,
        devices=devices,
        cbs=cbs,
        epochs=500 if profiler == False else 1,
        batchsize_finder=batch_finder,
        profiler=profiler,
        wandb=wandb,
        wandb_grid_epoch_freq=1,
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

