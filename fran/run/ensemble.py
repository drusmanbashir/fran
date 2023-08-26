
# %
import os
import argparse
# from fran.inference.transforms import *
from fran.utils.common import *
from fran.transforms.spatialtransforms import *
from fran.managers.trainer import *
from fran.managers.tune import *
from fran.inference.inference_base import *
from fran.utils.imageviewers import *
import itertools as il

    # ImageMaskViewer([img_np,mask_np])


tot_gpus=2
n_lists = 2

import os
import ray
ray.init(num_gpus=tot_gpus)

# %%

def slice_list(listi,start_end:list):
    # print(listi)
    return listi[start_end[0]:start_end[1]]

@ray.remote(num_cpus=8, num_gpus=tot_gpus/n_lists)
class EnsembleActor(object):
    def __init__(self):
        self.value = 0

    def process(self,proj_defaults,run_name_w,runs_ensemble ,fnames,half,debug,overwrite=False):
        self.En = EnsemblePredictor2(proj_defaults,3,run_name_w,runs_ensemble,bs=3,half=half,device='cuda',debug=debug,overwrite=overwrite)
        for img_fn in fnames:

            fname= Path(img_fn)
            img_sitk =sitk.ReadImage(fname)
            self.En.set_pred_fns(fname)
            self.En.run(img_sitk)
            self.En.save_prediction()
            self.En.unload_case()
# %%

def main(args):
    run_name_w= "LITS-490" # best trial
    input_folder = args.input_folder
    overwrite=args.overwrite
    half = args.half
    debug = args.debug
    ensemble = args.ensemble
    P = Project(project_title=args.t); proj_defaults= P
    # ensemble=["LITS-451","LITS-452","LITS-453","LITS-454","LITS-456"]
    # ensemble=["LITS-451"]
    # if not input_folder:
    #     mo_df = pd.read_csv(Path("/s/datasets_bkp/litq/complete_cases/cases_metadata.csv"))
    #     fnames = list(mo_df.image_filenames)
    fnames = list(Path(input_folder).glob("*"))

    fpl= int(len(fnames)/n_lists)
    inds = [[fpl*x,fpl*(x+1)] for x in range(n_lists-1)]
    inds.append([fpl*(n_lists-1),None])

    chunks = list(il.starmap(slice_list,zip([fnames]*n_lists,inds)))
    actors = [EnsembleActor.remote() for _ in range(n_lists)]
    results = ray.get([c.process.remote(proj_defaults,run_name_w,ensemble, fnames ,half, debug,overwrite) for c,fnames in zip(actors,chunks)])
    print(results)  # prints [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# gpu_actor = GPUActor.remote()
# %%

if __name__ == "__main__":
        
    common_vars_filename=os.environ['FRAN_COMMON_PATHS']
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
    # args.overwrite=False
    # args.t= 'lits'
    # # args.input_folder ="/s/datasets_bkp/litq/nifti/patient_60/" "/media/ub2/datasets/drli/sitk/images/"
    # args.input_folder = ""
    #
    main(args)


# Increment each Counter once and get the results. These tasks all happen in
# parallel.
# %%
