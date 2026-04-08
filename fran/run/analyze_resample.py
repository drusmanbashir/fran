# %%
import ast
from fran.configs.parser import ConfigMaker, confirm_plan_analyzed
from fran.managers import Project
from fran.preprocessing.datasetanalyzers import Path, multiprocess_multiarg
from fran.preprocessing.fixed_spacing import (
    NiftiToTorchDataGenerator,
    ResampleDatasetniftiToTorch,
)
from fran.preprocessing.imported import LabelBoundedDataGeneratorImported
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.preprocessing.patch import PatchDataGenerator
from fran.utils.folder_names import folder_names_from_plan
from utilz.fileio import os, save_list, str_to_path
from utilz.helpers import ask_proceed, re
from utilz.stringz import headline

common_vars_filename = os.environ["FRAN_CONF"]


def postprocess_complete(project, plan):
    if plan["mode"] != "lbd":
        return True
    folder = Path(folder_names_from_plan(project, plan)["data_folder_lbd"])
    stats_folder = folder / "dataset_stats"
    required = [
        folder / "labels_all.json",
        stats_folder / "lesion_stats.csv",
        # stats_folder / "snapshot.gif",
    ]
    return all(pth.exists() for pth in required)


def main(args):
    P = Project(project_title=args.project_title)
    C = ConfigMaker(P)
    overwrite = args.overwrite
    if args.num_processes < 1:
        args.num_processes = 1
    print(
        f"[analyze_resample] start project={args.project_title} plan={args.plan} overwrite={args.overwrite} num_processes={args.num_processes}"
    )

    if args.plan == 0:
        # Process all plans
        plan_ids = C.plans["plan_id"].tolist()
        headline("Processing ALL Plans: {}".format(plan_ids))
        for plan_id in plan_ids:
            print(f"[analyze_resample] setting up plan {plan_id}")
            C.setup(plan_id, plan_id)
            headline(plan_id)
            plan = C.configs["plan_train"]
            completed = confirm_plan_analyzed(P, plan)
            if (
                overwrite
                or not all(completed.values())
                or not postprocess_complete(P, plan)
            ):
                print(f"[analyze_resample] processing plan {plan_id}")
                args.plan = plan_id
                process_plan(args)
            else:
                print(f"Plan {plan_id} already processed. Skipping")
    else:
        # Process specific plan
        headline("Processing plan {}".format(args.plan))
        print(f"[analyze_resample] setting up plan {args.plan}")
        C.setup(args.plan, args.plan)
        plan = C.configs["plan_train"]
        completed = confirm_plan_analyzed(P, plan)
        if (
            overwrite
            or not all(completed.values())
            or not postprocess_complete(P, plan)
        ):
            print(f"[analyze_resample] processing plan {args.plan}")
            process_plan(args)
        else:
            print(f"Plan {args.plan} already processed. Skipping")


def process_plan(args):
    print(f"[analyze_resample] creating PreprocessingManager for plan {args.plan}")
    mgr = PreprocessingManager(args)
    print(
        f"[analyze_resample] stage=resample_dataset plan={args.plan} mode={mgr.plan['mode']} num_processes={args.num_processes}"
    )
    mgr.resample_dataset(
        overwrite=args.overwrite,
        num_processes=args.num_processes,
        debug=args.debug,
    )
    print(f"[analyze_resample] stage=resample_dataset complete plan={args.plan}")
    # args.num_processes = 1

    if mgr.plan["mode"] == "pbd":
        print(
            f"[analyze_resample] stage=generate_hires_patches_dataset plan={args.plan}"
        )
        mgr.generate_hires_patches_dataset(
            overwrite=args.overwrite,
            num_processes=args.num_processes,
            debug=args.debug,
        )
        print(
            f"[analyze_resample] stage=generate_hires_patches_dataset complete plan={args.plan}"
        )
    elif mgr.plan["mode"] == "lbd":
        imported_folder = mgr.plan.get("imported_folder", None)
        if imported_folder is None:
            print(f"[analyze_resample] stage=generate_lbd_dataset plan={args.plan}")
            mgr.generate_lbd_dataset(
                overwrite=args.overwrite,
                num_processes=args.num_processes,
                debug=args.debug,
            )
            print(
                f"[analyze_resample] stage=generate_lbd_dataset complete plan={args.plan}"
            )
        else:
            print(
                f"[analyze_resample] stage=generate_TSlabelboundeddataset plan={args.plan} imported_folder={imported_folder}"
            )
            mgr.generate_TSlabelboundeddataset(
                overwrite=args.overwrite,
                num_processes=args.num_processes,
                debug=args.debug,
            )
            print(
                f"[analyze_resample] stage=generate_TSlabelboundeddataset complete plan={args.plan}"
            )
    print(f"[analyze_resample] finished plan {args.plan}")


@str_to_path(0)
def verify_dataset_integrity(folder: Path, debug=False, fix=False):
    """
    folder has subfolders images and masks
    """
    print("Verifying dataset integrity")
    subfolder = list(folder.glob("mask*"))[0]
    args = [[fn, fix] for fn in subfolder.glob("*")]
    res = multiprocess_multiarg(verify_img_label_match, args, debug=debug, io=True)  # noqa: F405
    errors = [item for item in res if re.search("mismatch", item[0], re.IGNORECASE)]
    if len(errors) > 0:
        outname = folder / ("errors.txt")
        print(f"Errors found saved in {outname}")
        save_list(errors, outname)
        res.insert(0, errors)
    else:
        print("All images and masks are verified for matching sizes and spacings.")
    return res


def user_input(inp: str, out=int):
    tmp = input(inp)
    try:
        tmp = ast.literal_eval(tmp)
        tmp = out(tmp)
    except Exception:
        tmp = None
    return tmp


class PreprocessingManager:
    def __init__(self, args, conf=None):
        P = Project(project_title=args.project_title)
        self.project = P
        if conf is None:
            C = ConfigMaker(P)
            C.setup(args.plan)
            conf = C.configs
        self.plan = conf["plan_train"]
        print("Project: {0}".format(args.project_title))

    def resample_dataset(self, overwrite=False, num_processes=1, debug=False):
        """
        Resamples dataset to target spacing and stores it in the cold_storage fixed_spacing_folder.
        Typically this will be a basis for further processing e.g., pbd, lbd dataset which will then be used in training
        """

        self.R = NiftiToTorchDataGenerator(
            project=self.project,
            plan=self.plan,
            data_folder=self.project.raw_data_folder,
        )

        self.R.setup(overwrite=overwrite, num_processes=num_processes, debug=debug)
        self.R.process()

    def generate_lbd_dataset(
        self, overwrite=False, device="cpu", num_processes=1, debug=False
    ):

        resampled_data_folder = folder_names_from_plan(self.project, self.plan)[
            "data_folder_source"
        ]

        headline(
            "LBD dataset will be based on resampled dataset output_folder {}".format(
                resampled_data_folder
            )
        )
        self.L = LabelBoundedDataGenerator(
            project=self.project,
            plan=self.plan,
            data_folder=resampled_data_folder,
        )
        self.L.setup(
            overwrite=overwrite,
            device=device,
            num_processes=num_processes,
            debug=debug,
        )
        self.L.process()

    def generate_TSlabelboundeddataset(
        self,
        device="cpu",
        overwrite=False,
        num_processes=1,
        debug=False,
    ):
        """
        requires resampled folder to exist. Crops within this folder
        """

        resampled_data_folder = folder_names_from_plan(self.project, self.plan)[
            "data_folder_source"
        ]
        self.L = LabelBoundedDataGeneratorImported(
            project=self.project,
            plan=self.plan,
            data_folder=resampled_data_folder,
        )
        self.L.setup(
            overwrite=overwrite,
            device=device,
            num_processes=num_processes,
            debug=debug,
        )
        self.L.process()

    @ask_proceed("Generating low-res whole images to localise organ of interest")
    def generate_whole_images_dataset(self):

        if not hasattr(self, "spacing"):
            self.set_spacing()
        output_shape = ast.literal_eval(
            input(
                "Enter whole image matrix shape as list/tuple/number(e.g., [128,128,96]): "
            )
        )
        if isinstance(output_shape, (int, float)):
            output_shape = [
                output_shape,
            ] * 3
        self.WholeImageTM = WholeImageTensorMaker(  # noqa: F405
            self.project,
            source_spacing=self.plan["spacing"],
            output_size=output_shape,
            num_processes=self.num_processes,
        )
        arglist_imgs, arglist_masks = self.WholeImageTM.get_args_for_resizing()
        for arglist in [arglist_imgs, arglist_masks]:
            multiprocess_multiarg(
                func=resize_and_save_tensors,  # noqa: F405
                arguments=arglist,
                num_processes=self.num_processes,
                debug=self.debug,
                io=True,
            )
        print("Now call bboxes_from_masks_folder")
        generate_bboxes_from_masks_folder(  # noqa: F405
            self.WholeImageTM.output_folder_masks, 0, self.debug, self.num_processes
        )

    def generate_hires_patches_dataset(
        self, debug=False, overwrite=False, num_processes=1
    ):

        data_folder = self.get_source_data_folder_for_patch()
        PG = PatchDataGenerator(
            project=self.project,
            plan=self.plan,
            data_folder=data_folder,
        )
        PG.setup(
            overwrite=overwrite,
            num_processes=num_processes,
            debug=debug,
        )
        PG.process(derive_bboxes=False)

    def get_source_data_folder_for_patch(self):
        src_plan = self.plan["source_plan"]
        src_plan_idx, src_plan_mode = src_plan.replace(" ", "").split(",")
        src_plan_idx = int(src_plan_idx)
        C2 = ConfigMaker(self.project)
        C2.setup(src_plan_idx)
        src_plan_full = C2.configs["plan_train"]
        data_fldrs = folder_names_from_plan(self.project, src_plan_full)
        data_folder = data_fldrs[f"data_folder_{src_plan_mode}"]
        data_foldre = Path(data_folder)
        return data_foldre

# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
if __name__ == "__main__":
    import sys
    import argparse

    from fran.utils.common import *  # noqa: F403
    from utilz.cprint import cprint

    parser = argparse.ArgumentParser(description="Resampler")

    parser.add_argument(
        "-t", "--project-title", "--project", help="project title", dest="project_title"
    )
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        help="number of parallel processes",
        default=1,
    )
    parser.add_argument(
        "-p",
        "--plan",
        "--plan-num",
        type=int,
        help="Just a number. If 0 or None selected, all plans will be analyzed.",
        default=0,
    )
    parser.add_argument("-d", "--debug", action="store_true")

    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument(
        "--help-args",
        action="store_true",
        help="Show CLI help and exit.",
    )
    args = parser.parse_known_args()[0]
# %%
    # cprint("Warning: Using args saved into file analyze_resample.py", color= "red")
    args.project_title="totalseg"
    args.plan = 2
    # args.project_title = "test"
    # args.plan = 1
    args.num_processes = 6
    args.overwrite = False
    args.debug = False

# %%
    cprint("Project: {0}".format(args.project_title), color="green")

    if args.help_args:
        parser.print_help()
        raise SystemExit(0)
    main(args)

# %%
    #
    mgr = PreprocessingManager(args)
    mgr.resample_dataset(overwrite=args.overwrite, num_processes=args.num_processes)
    mgr.plan["mode"]

# %%
    mgr.R = ResampleDatasetniftiToTorch(
        project=mgr.project,
        plan=mgr.plan,
        data_folder=mgr.project.raw_data_folder,
    )

# %%
    #
    overwrite = False
    num_processes = 8
    mgr.R.setup(overwrite=overwrite, num_processes=num_processes)
    #     mgr.R.process()
    #     mgr.resample_output_folder = mgr.R.output_folder
    # resampled_data_folder = folder_names_from_plan(mgr.project, mgr.plan)[
    #
    #        "data_folder_source"
    #    ]
# %%
    #
    #         data_folder = mgr.get_source_data_folder_for_patch()
    #         PG = PatchDataGenerator(
    #             project=mgr.project,
    #             plan=mgr.plan,
    #             data_folder=data_folder,
    #         )
    #         PG.setup(
    #             overwrite=overwrite,
    #             num_processes=mgr.num_processes,
    #             debug=debug,
    #         )
    #         PG.process(derive_bboxes=False)
    #     mgr.R.create_dataset_stats_artifacts()
    #
# %%
    #     overwrite=False
    #     num_processes=8
    #
    #     resampled_data_folder = folder_names_from_plan(mgr.project, mgr.plan)[
    #         "data_folder_source"
    #     ]
    #
# %%
    #     mgr.L = LabelBoundedDataGeneratorImported(
    #         project=mgr.project,
    #         plan=mgr.plan,
    #         data_folder=resampled_data_folder,
    #     )
# %%
# %%
    #     device='cpu'
    #     mgr.L.setup(overwrite=overwrite, device=device,num_processes=num_processes,debug=True)
    #     mgr.L.process()
# %%
    #
    #
# %%
    #     # sys.exit()
    sys.exit()
# %%
# dataset_root = Path(mgr.R.output_folder)
# lms_folder = dataset_root / "lms"
# if not lms_folder.exists():
#     print(f"Skipping dataset stats: missing labels folder {lms_folder}")
#     return
#
# stats_folder = dataset_root / "dataset_stats"
# maybe_makedirs([stats_folder])
#
# from label_analysis.dataset_stats import end2end_lms_stats_and_plots
# from utilz.overlay_grid_gif import create_nifti_overlay_grid_gif
# %%

