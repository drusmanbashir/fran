# %%
import os
from fran.managers.project import Project
import argparse

# from fran.inference.transforms import *
from fran.transforms.spatialtransforms import *
from fran.managers.tune import *
from fran.inference.cascade import *
from utilz.imageviewers import *
import itertools as il

# ImageMaskViewer([img_np,mask_np])


tot_gpus = 2
n_lists = 2

import os
import ray

ray.init(num_gpus=tot_gpus)



@ray.remote(num_cpus=8, num_gpus=tot_gpus / n_lists)
class EnsembleActor(object):
    def __init__(self):
        self.value = 0

    def process(
        self, project, run_name_w, runs_ensemble,localiser_labels,safe_mode,   k_largest, fnames,  chunksize, overwrite
    ):
        En = CascadeInferer(
            project=project,
            run_name_w=run_name_w,
            runs_p=runs_ensemble,
            localiser_labels=localiser_labels,
            safe_mode=safe_mode,
            save_channels=False,
            save=True,
            overwrite=overwrite,
            k_largest=k_largest,

        )
        preds = En.run(fnames,chunksize=chunksize)
        return 1


# %%


def main(args):

    run_name_w = "LITS-1088"
    input_folder = args.input_folder
    project = Project(project_title=args.t)
    overwrite = args.overwrite
    runs_ensemble = args.ensemble
    localiser_labels = args.localiser_labels
    chunksize=args.chunksize
    safe_mode = args.safe_mode
    save_channels=False
    k_largest=1
    # run_ps=['LIT-62','LIT-63','LIT-64' 'LIT-44','LIT-59']
    # ensemble=["LITS-451","LITS-452","LITS-453","LITS-454","LITS-456"]
    # ensemble=["LITS-451"]
    # if not input_folder:
    #     mo_df = pd.read_csv(Path("/s/datasets_bkp/litq/complete_cases/cases_metadata.csv"))
    #     fnames = list(mo_df.image_filenames)
    save=True
    fns = list(Path(input_folder).glob("*"))

    fpl = int(len(fns) / n_lists)
    inds = [[fpl * x, fpl * (x + 1)] for x in range(n_lists - 1)]
    inds.append([fpl * (n_lists - 1), None])

    chunks = list(il.starmap(slice_list, zip([fns] * n_lists, inds)))
    actors = [EnsembleActor.remote() for _ in range(n_lists)]
    results = ray.get(
        [
            c.process.remote(
                project, run_name_w, runs_ensemble, localiser_labels, safe_mode,   k_largest, fns, chunksize, overwrite
            )
            for c, fns in zip(actors, chunks)
        ]
    )
    print(results)  # prints [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


# gpu_actor = GPUActor.remote()
# %%

if __name__ == "__main__":

    from fran.utils.common import *
    common_vars_filename = os.environ["FRAN_COMMON_PATHS"]
    # runs_ensemble=["LITS-444","LITS-443","LITS-439","LITS-436","LITS-445"]
    # runs_ensemble=["LITS-265","LITS-255","LITS-270","LITS-271","LITS-272"]

    # runs_ensemble=["LITS-408","LITS-385","LITS-383","LITS-357"]
    parser = argparse.ArgumentParser(description="Ensemble Predictor")
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")

    parser.add_argument("-t", help="project title")
    parser.add_argument("-i", "--input-folder")
    parser.add_argument("-f", "--half", action="store_true")
    parser.add_argument("-e", "--ensemble", nargs="+")
    parser.add_argument("--gpus", type=int, default=0)
    args = parser.parse_known_args()[0]

    args.overwrite=True
    args.chunksize=4
    args.safe_mode=True
    args.t= 'litsmc'
    args.input_folder ="/s/xnat_shadow/crc/images"
    args.ensemble= ["LITS-1018"]
    args.localiser_labels = [3]

# %%
    main(args)


# Increment each Counter once and get the results. These tasks all happen in
# parallel.
# %%


# %%

# %%
