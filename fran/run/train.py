# %%
from fran.utils.common import *
import ast
from fran.managers.training import *
_translations =     {
    'bs': 'dataset_params,bs',
    'fold':'dataset_params,fold',
    'lr': 'model_params,lr',
    'labels':  'dataset_params,src_dest_labels',
    'arch': 'model_params,arch',

    }

# %%
       
def process_run_name(run_name):
        if run_name == None: return None
        elif run_name =='': return 'most_recent'
        else: return run_name


def override_configs(args , configs:dict):
    def _alter_config_key(ans,val):
        if isinstance(ans,str):
            keys = str_to_key(ans)
        elif isinstance(ans,dict):
            keys = str_to_key(ans['config_name'])
            val = ans[val]
        inner_dict = {keys[1]:val}
        # outer_dict = {keys[0]:inner_dict}
        if keys[0] in configs.keys():
            configs[keys[0]].update(inner_dict)
        else: configs[keys[0]] =inner_dict
    str_to_key = lambda  x: x.split(",")
    for key,val in  vars(args).items():
        if key in _translations.keys() and val is not None:
            ans = _translations[key]
            _alter_config_key(ans,val)
    if len(configs)>0:
        return configs
    else : return None

def maybe_compute_bs(project,configs ,args):
    # if hasattr(args, "bs") or hasattr(args,"resume"):
    if any([s is not None for s in [args.bs,args.resume]]):
        args.bs =args.bs
    else:
        args.bs = compute_bs(project=project,config=configs, bs=6)
    return  args.bs


    pass
def load_and_update_configs(project, args,compute_bs=True):
    # if recompute_bs==True:
    # if args.resume is None or args.update == True:

    configs = ConfigMaker(
        project,
        raytune=False,
        configuration_filename=args.conf_fn
    ).config

    args.bs = maybe_compute_bs(project,configs, args)
    # else:
    #     configs = {}
    updated_configs =override_configs(args, configs)
    return updated_configs


def load_run(project,run_name,args):
    if args.update==True:
        updated_configs = load_and_update_configs(project, args )
    else: updated_configs = None
    La = Trainer.fromNeptuneRun(
        project,
        run_name=run_name,
        update_nep_run_from_config=updated_configs,
        device= args.gpu
    )
    return La


def initialize_run(project ,args):

    configs = load_and_update_configs(project,args)
    configs['model_params']['compiled'] = args.compiled
    cbs = [
    ModelCheckpoint(),
     LearningRateMonitor(logging_interval='epoch')
    ]
    
    run_name = process_run_name(args.resume)
    Tm = TrainingManager(project, configs)
    Tm.setup(batch_size = args.bs , run_name= run_name,cbs=cbs,devices=args.devices,neptune=args.neptune,epochs=args.epochs,compiled=args.compiled,description=args.desc)
    return Tm

def main(args):
# %%

    torch.set_float32_matmul_precision('medium')
    assert( args.t), "No project title given. Restart and set the -t flag"
    project_title = args.t
    project = Project(project_title=project_title);
    print("Project: {0}".format(project_title))
    Tm = initialize_run(project, args)
    Tm.fit()
# %%

if __name__ == "__main__":
    from fran.utils.common import *
    import argparse
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("-t", help="project title")#, required=True)
    # parser.add_argument("t", help="project title")
    parser.add_argument("-e","--epochs", help="num epochs", default=1000,type=int)
    parser.add_argument(
        "-r","--resume",
        const="",
        nargs='?',
        help="Leave empty to resume last training session or enter a run name.",
    )  # neptune manager saves last session's name in excel spreadsheet

    parser.add_argument("-b", "--bs", help="batch size",type=int)
    parser.add_argument("--desc")
    parser.add_argument("-f","--fold", type=int, default=0)
    parser.add_argument("-d","--devices", type=str, default='1')
    parser.add_argument("-c","--compiled", action='store_true')
    parser.add_argument("-cf","--conf-fn", default = None)
    parser.add_argument("--lr", help="learning rate",type=float )
    parser.add_argument("--gpu", help="gpu id",type=int, default=0)

    parser.add_argument("-a", "--arch", help="Architecture. Supports: nnUNet, SwinUNETR, DynUNet")
    parser.add_argument("-u", "--update", help="Update existing run from configs excel spreadsheet.",action='store_true')
    parser.add_argument(
        "-p", "--patch", default=None)
    parser.add_argument("--labels", help="list of mappings source to dest label values, e.e.,g [[0,0],[1,1],[2,1]] will map all foreground to 1")
    parser.add_argument("-n","--neptune", help="No Neptune",action='store_false')
# %%
    args = parser.parse_known_args()[0]
    args.devices=ast.literal_eval(args.devices)
    # args.neptune = True 
    # args.bs=8
    # # args.resume="LIT-184"
    # args.compiled= True
    # args.t = 'lits32'
    #
    # args.conf_fn = "/s/fran_storage/projects/lits32/experiment_configs_wholeimage.xlsx"
    # args.bs = 8
    # args.lr = 1e-4
    # args.devices = 2
    # # args.resume=''
    # # args.resume='LITS-456'
    # args.update = True

# %%
    main(args)
# %% 
