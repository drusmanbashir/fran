# %%
import itertools
import yaml
from types import SimpleNamespace
import argparse

from fran.run.analyze_resample import PreprocessingManager
from utilz.fileio import *
from utilz.helpers import *

from fran.managers import Project

from fran.preprocessing.datasetanalyzers import *
from fran.tune.config import RayTuneConfig, load_tune_template

common_vars_filename = os.environ["FRAN_CONF"]


def generate_dataset(project_title):
    # P = Project(project_title=args.project_title)
    # P.create(mnemonic= "litsmall")
    # P.add_data([DS["litsmall"]])
    # P.create(mnemonic="lidc")
    # P.create(mnemonic="lidc", datasources=[DS["lidc"]])
    # P.add_data([DS["lidc"]])
    # P.maybe_store_projectwide_properties()

    parser = argparse.ArgumentParser(description="Resampler")

    parser.add_argument(
        "-t", "--project-title", help="project title", dest="project_title"
    )
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        help="number of parallel processes",
        default=1,
    )
    parser.add_argument("-p", "--plan", type=int, help="Just a number like 1, 2")

    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_known_args()[0]
    args.project_title=project_title
    # args.plan = 6
    args.num_processes = 4
    args.overwrite=False
    I = PreprocessingManager(args)
    I.resample_dataset(overwrite=args.overwrite,num_processes=args.num_processes)
    # args.num_processes = 1

    if I.plan["mode"] == "patch":
        # I.generate_TSlabelboundeddataset("lungs","/s/fran_storage/predictions/totalseg/LITS-827")
        I.generate_hires_patches_dataset(overwrite=args.overwrite)
    elif I.plan["mode"] == "lbd":
        imported_folder = I.plan.get("imported_folder", None)
        if imported_folder is None:
            I.generate_lbd_dataset(overwrite=args.overwrite,num_processes=args.num_processes)
        else:
            I.generate_TSlabelboundeddataset(
                overwrite=args.overwrite,
                num_processes=args.num_processes
            )



def build_datasets_from_yaml(yaml_path: str, project_title: str, *, num_processes=4, overwrite=False):
    """
    Read tune.yaml and generate all dataset permutations (no training).

    Automatically extracts dataset parameter permutations (src_dim*, expand_by, etc.)
    and builds the corresponding datasets.
    """
    # --- Load tuning config ---
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_params = config.get("dataset_params", {})
    if not dataset_params:
        raise ValueError("No dataset_params found in tune.yaml")

    # --- Split scalar vs list params ---
    keys, values = [], []
    for k, v in dataset_params.items():
        if isinstance(v, list):
            keys.append(k)
            values.append(v)
    if not keys:
        print("No permutations found; running single dataset build.")
        values = [[]]

    permutations = [dict(zip(keys, combo)) for combo in itertools.product(*values)] or [{}]

    # --- Build datasets for each permutation ---
    for spec in permutations:
        args = SimpleNamespace(project_title=project_title,
                               num_processes=num_processes,
                               overwrite=overwrite,
                               plan=None)
        pm = PreprocessingManagerTune(args)

        # Apply overrides from permutation
        for k, v in spec.items():
            pm.plan[k] = v

        # Optional derived parameters
        if "patch_dim0" in pm.plan and "patch_dim1" in pm.plan:
            pm.plan["patch_size"] = [pm.plan["patch_dim0"], pm.plan["patch_dim1"], pm.plan["patch_dim1"]]

        # Resample + dataset generation
        pm.resample_dataset(overwrite=overwrite, num_processes=num_processes)
        mode = pm.plan.get("mode", "lbd")
        if mode == "patch":
            pm.generate_hires_patches_dataset(overwrite=overwrite)
        elif mode == "lbd":
            if pm.plan.get("imported_folder"):
                pm.generate_TSlabelboundeddataset(overwrite=overwrite, num_processes=num_processes)
            else:
                pm.generate_lbd_dataset(overwrite=overwrite, num_processes=num_processes)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    print(f"Generated {len(permutations)} dataset permutations from {yaml_path}")
if __name__ == '__main__':
# %%
    # --- Load tuning config ---
    conf = load_tune_template()
    dataset_vars = ["spacing", "expand_by"]
    vars={}
    for var in dataset_vars:
            vars[var] = conf[var]

# %%


    project_title="litsmc"
    parser = argparse.ArgumentParser(description="Resampler")

    parser.add_argument(
        "-t", "--project-title", help="project title", dest="project_title"
    )
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        help="number of parallel processes",
        default=1,
    )
    parser.add_argument("-p", "--plan", type=int, help="Just a number like 1, 2")

    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_known_args()[0]

    args.project_title=project_title
    args.plan = 1
    args.num_processes = 4


# %%

    for i in range(3):
        spacing = vars["spacing"]["value"][i]
        for n in range (len(vars["expand_by"]["value"])):
            expand_by = vars["expand_by"]["value"][n]
            I = PreprocessingManager(args)
            I.plan["mode"] = "lbd"
            I.plan["expand_by"] = expand_by
            I.plan["spacing"] = spacing

            I.resample_dataset(overwrite=args.overwrite,num_processes=args.num_processes)
            imported_folder = I.plan.get("imported_folder", None)
            if imported_folder is None:
                I.generate_lbd_dataset(overwrite=args.overwrite,num_processes=args.num_processes)
            else:
                I.generate_TSlabelboundeddataset(
                    overwrite=args.overwrite,
                    num_processes=args.num_processes
                )
# %%
