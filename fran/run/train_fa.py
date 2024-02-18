# %%
from fran.utils.common import *
from fran.managers.trainer import *
_translations =     {'mode' :{'config_name':'metadata,patch_based', 'hi':True,'lo':False},
    'bs': 'dataset_params,bs',
    'fold':'metadata,fold',
    'lr': 'model_params,lr',
    'labels':  'metadata,src_dest_labels',
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
        if key in _translations.keys() and val:
            ans = _translations[key]
            _alter_config_key(ans,val)
    if len(configs)>0:
        return configs
    else : return None

def maybe_compute_bs(project,args):
    # if hasattr(args, "bs") or hasattr(args,"resume"):
    if any([s is not None for s in [args.bs,args.resume]]):
        args.bs =args.bs
    else:
        args.bs = compute_bs(project=project,distributed=args.distributed,bs=12)
    return  args.bs


def load_and_update_configs(project, args,compute_bs=True):
    # if recompute_bs==True:
    # if args.resume is None or args.update == True:
    args.bs = maybe_compute_bs(project,args)
    configs = ConfigMaker(project, raytune=False).config

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
    configs['model_params/compiled'] = args.compile


    cbs = [
            ReduceLROnPlateau(patience=50),
            NeptuneCallback(project, configs, run_name=None),
            NeptuneCheckpointCallback(project.checkpoints_parent_folder),
            NeptuneImageGridCallback(
                classes=out_channels_from_dict_or_cell(
                    configs["metadata"]["src_dest_labels"]
                ),
                patch_size=make_patch_size(
                    configs["dataset_params"]["patch_dim0"],
                    configs["dataset_params"]["patch_dim1"],
                ),
            ),
            #
        ]
    La = Trainer(project, configs,cbs, device =args.gpu)
    return La

def main(args):

    project_title = args.t
    project = Project(project_title=project_title);
    print("Project: {0}".format(project_title))

    n_epoch = args.epochs
    if args.distributed:   args.gpu  = 0

    run_name = process_run_name(args.resume)
    if not run_name:
        La = initialize_run(project, args)
    else:
        La = load_run(project, run_name,args)

    learn = La.create_learner(compile=args.compile,distributed=args.distributed)

#     # learn.model = model
    if args.lr :
        lr = args.lr 
    else:
        lr = La.model_params['lr']
    learn.fit(n_epoch=n_epoch, lr=lr)
# %%

if __name__ == "__main__":
    from fran.utils.common import *
    import argparse
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("-t", help="project title")#, required=True)
    # parser.add_argument("t", help="project title")
    parser.add_argument("-e","--epochs", help="num epochs", default=500,type=int)
    parser.add_argument(
        "-r","--resume",
        const="",
        nargs='?',
        help="Leave empty to resume last training session or enter a run name.",
    )  # neptune manager saves last session's name in excel spreadsheet

    parser.add_argument("--bs", help="batch size",type=int)
    parser.add_argument("-f","--fold", type=int, default=0)
    parser.add_argument("-d","--distributed", action='store_true')
    parser.add_argument("-c","--compile", action='store_true')
    parser.add_argument("--lr", help="learning rate",type=float)
    parser.add_argument("--gpu", help="gpu id",type=int, default=0)

    parser.add_argument("-a", "--arch", help="Architecture. Supports: nnUNet, SwinUNETR, DynUNet")
    parser.add_argument("-u", "--update", help="Update existing run from configs excel spreadsheet.",action='store_true')
    parser.add_argument(
        "--mode",
        choices=["hi", "lo"],
        help="To train low-res organ localizer select 'l'. To train high-res patches selectively on the target organ, select 'h",
        required=False
    )
    parser.add_argument("--labels", help="list of mappings source to dest label values, e.e.,g [[0,0],[1,1],[2,1]] will map all foreground to 1")
    parser.add_argument("-n", help="No Neptune",action='store_true')
# %%
    args = parser.parse_known_args()[0]
    args.t = 'lits'
    args.lr = 1e-4
    args.distributed = False
    # args.resume=''
    # args.resume='LITS-456'
    args.compiled= False
    args.bs = 4
    # args.update = True

# %%
    main(args)
# %%
