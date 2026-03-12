
# %
import os
import argparse
# from fran.inference.transforms import *
from fran.utils.common import *
from fran.transforms.spatialtransforms import *
from fran.managers.tune import *
from fran.inference.cascade import *
from utilz.imageviewers import *

    # ImageMaskViewer([img_np,mask_np])



import os

# %%

def slice_list(listi,start_end:list):
    # print(listi)
    return listi[start_end[0]:start_end[1]]

class EnsembleActor(object):
    def __init__(self):
        self.value = 0

    def process(self,project,run_name_w,runs_ensemble ,fnames,half,debug,overwrite=False):
        En = CascadeInferer(project=project, run_name_w=run_name_w, runs_p=runs_ensemble, debug=debug,overwrite_p=overwrite )
        preds = En.run(fnames)
        return 1

# %%

def main(args):
    run_name_w= "LITS-464" # best trial
    input_folder = args.input_folder
    overwrite=args.overwrite
    half = args.half
    debug = args.debug
    ensemble = args.ensemble
    P = Project(project_title=args.t); proj_defaults= P
    # if not input_folder:
    #     mo_df = pd.read_csv(Path("/s/datasets_bkp/litq/complete_cases/cases_metadata.csv"))
    #     fnames = list(mo_df.image_filenames)
    fnames = list(Path(input_folder).glob("*"))

    actor = EnsembleActor()
    results = [actor.process(proj_defaults,run_name_w,ensemble, fnames ,half, debug,overwrite) for fname in fnames]

    print(results)  # prints [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# gpu_actor = GPUActor.remote()
# %%

if __name__ == "__main__":
        
    common_vars_filename=os.environ['FRAN_CONF']+"/config.yaml"
    varsi = load_yaml(common_vars_filename)
    # runs_ensemble=["LITS-444","LITS-443","LITS-439","LITS-436","LITS-445"]
    # runs_ensemble=["LITS-265","LITS-255","LITS-270","LITS-271","LITS-272"]

    # runs_ensemble=["LITS-408","LITS-385","LITS-383","LITS-357"]
    parser = argparse.ArgumentParser(description="Ensemble Predictor")
    parser.add_argument('-o','--overwrite',action='store_true')
    parser.add_argument('-d','--debug',action='store_true')

    parser.add_argument("-t", help="project title")
    parser.add_argument('-i','--input-folder')
    parser.add_argument('-f','--half', action='store_true')
    parser.add_argument('-e','--ensemble', nargs='+')
    parser.add_argument('--gpus', type=int,default=0)
    args = parser.parse_known_args()[0]

# %%

    args.overwrite=False
    args.t= 'litsmc'
    args.input_folder ="/s/xnat_shadow/crc/wxh/completed"
    args.ensemble= ["LITS-787", "LITS-810", "LITS-811"]
    # args.overwrite=False
# %%
    main(args)


# Increment each Counter once and get the results. These tasks all happen in
# parallel.
# %%
