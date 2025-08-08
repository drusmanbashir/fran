# %%
import argparse
from pathlib import Path
from fran.managers import Project, Datasource, _DS
from fran.utils.config_parsers import ConfigMaker


# %%
#SECTION:-------------------- Project creation--------------------------------------------------------------------------------------



if __name__ == '__main__':
    from fran.utils.common import *
    
    P = Project("litstmp")
# P.delete()

    DS = _DS()
    P.add_data([DS.lits_tmp])
# P.add_data([DS.totalseg])
# %%
#SECTION:-------------------- DATA FOLDER H5PY file--------------------------------------------------------------------------------------

    test =False
    ds = Datasource(folder=Path("/s/datasets_bkp/litstmp"), name="lits_tmp", alias="tmp", test=test)
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
    args.plan = "plan2"
    args.project_title = "litstmp"

    P = Project(project_title=args.project_title)

    conf = ConfigMaker(P, raytune=False, configuration_filename=None).config

    plan = conf[args.plan]
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
    I.resample_dataset(overwrite=False)
    I.R.get_tensor_folder_stats()

# %%
#SECTION:--------------------  Processing based on MODE ------------------------------------------------------------------
    if I.plan["mode"] == "patch":
        # I.generate_TSlabelboundeddataset("lungs","/s/fran_storage/predictions/totalseg/LITS-827")
        I.generate_hires_patches_dataset()
    elif I.plan["mode"] == "lbd":
        if "imported_folder" not in plan.keys():
            I.generate_lbd_dataset(overwrite=False)
        else:
            I.generate_TSlabelboundeddataset(
                imported_labels=plan["imported_labels"],
                imported_folder=plan["imported_folder"],)
# %%

# %%
# S
