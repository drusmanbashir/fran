# %%
import ast
import threading

from fran.configs.parser import ConfigMaker, confirm_plan_analyzed
from fran.managers import Project
from fran.preprocessing.datasetanalyzers import Path, headline, multiprocess_multiarg
from fran.preprocessing.fixed_spacing import (
    NiftiToTorchDataGenerator,
)
from fran.preprocessing.helpers import env_flag
from fran.preprocessing.fixed_size2 import FixedSizeDataGenerator
from fran.preprocessing.imported import LabelBoundedDataGeneratorImported
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.preprocessing.patch import PatchDataGenerator
from fran.preprocessing.preprocessor import DEFAULT_HDF5_SRC_DIMS
from fran.preprocessing.regionbounded import RegionBoundedDataGenerator
from fran.utils.folder_names import FolderNames
from tqdm.auto import tqdm
from utilz.fileio import os, save_list, str_to_path
from utilz.helpers import re
from utilz.stringz import headline, info_from_filename

common_vars_filename = os.environ["FRAN_CONF"]


def postprocess_complete(project, plan):
    if plan["mode"] != "lbd":
        return True
    folder = Path(FolderNames(project, plan).folders["data_folder_lbd"])
    stats_folder = folder / "dataset_stats"
    required = [
        folder / "labels_all.json",
    ]
    if env_flag("FRAN_STORE_LABEL_STATS", True):
        required.append(stats_folder / "lesion_stats.csv")
    if env_flag("FRAN_STORE_GIFS", False):
        required.append(stats_folder / "snapshot.gif")
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
    I = PreprocessingManager(args)
    print(
        f"[analyze_resample] stage=resample_dataset plan={args.plan} mode={I.plan['mode']} num_processes={args.num_processes}"
    )
    I.resample_dataset(
        overwrite=args.overwrite,
        num_processes=args.num_processes,
        debug=args.debug,
    )
    print(f"[analyze_resample] stage=resample_dataset complete plan={args.plan}")
    # args.num_processes = 1

    if I.plan["mode"] in ["pbd", "patch"]:
        print(
            f"[analyze_resample] stage=generate_hires_patches_dataset plan={args.plan}"
        )
        I.generate_hires_patches_dataset(
            overwrite=args.overwrite,
            num_processes=args.num_processes,
            debug=args.debug,
        )
        print(
            f"[analyze_resample] stage=generate_hires_patches_dataset complete plan={args.plan}"
        )
    elif I.plan["mode"] == "lbd":
        imported_folder = I.plan.get("imported_folder", None)
        if imported_folder is None:
            print(f"[analyze_resample] stage=generate_lbd_dataset plan={args.plan}")
            I.generate_lbd_dataset(
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
            I.generate_TSlabelboundeddataset(
                overwrite=args.overwrite,
                num_processes=args.num_processes,
                debug=args.debug,
            )
            print(
                f"[analyze_resample] stage=generate_TSlabelboundeddataset complete plan={args.plan}"
            )
    elif I.plan["mode"] == "rbd":
        print(f"[analyze_resample] stage=generate_rbd_dataset plan={args.plan}")
        I.generate_rbd_dataset(
            overwrite=args.overwrite,
            num_processes=args.num_processes,
            debug=args.debug,
        )
        print(
            f"[analyze_resample] stage=generate_rbd_dataset complete plan={args.plan}"
        )
    elif I.plan["mode"] == "whole":
        print(f"[analyze_resample] stage=generate_whole_images_dataset plan={args.plan}")
        I.generate_whole_images_dataset(
            overwrite=args.overwrite,
            num_processes=args.num_processes,
            debug=args.debug,
        )
        print(
            f"[analyze_resample] stage=generate_whole_images_dataset complete plan={args.plan}"
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
    res = multiprocess_multiarg(verify_img_label_match, args, debug=debug, io=True)
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
    except:
        tmp = None
    return tmp


class StageProgressCounter:
    def __init__(self, output_folder, total: int):
        self.output_folder = Path(output_folder)
        self.total = int(total)
        self.images_folder = self.output_folder / "images"
        self.lms_folder = self.output_folder / "lms"
        self.baseline_images = self._names(self.images_folder)
        self.baseline_lms = self._names(self.lms_folder)

    def _names(self, folder):
        return {pth.name for pth in folder.glob("*.pt")}

    def completed_cases(self):
        raise NotImplementedError


class ExactCaseOutputCounter(StageProgressCounter):
    def completed_cases(self):
        baseline = self.baseline_images.intersection(self.baseline_lms)
        current = self._names(self.images_folder).intersection(self._names(self.lms_folder))
        return min(self.total, len(current - baseline))


class CaseOutputCounter(StageProgressCounter):
    def completed_cases(self):
        current = self._names(self.images_folder)
        return min(self.total, len(current - self.baseline_images))


class PatchCaseApproxCounter(StageProgressCounter):
    def completed_cases(self):
        current = self._names(self.images_folder)
        new_patch_files = len(current - self.baseline_images)
        case_ids = {info_from_filename(name, full_caseid=True)["case_id"] for name in current}
        estimated_patches_per_case = len(current) / max(1, len(case_ids))
        completed = int(new_patch_files / max(1.0, estimated_patches_per_case))
        return min(self.total, completed)


class OutputFolderProgressMonitor:
    def __init__(
        self,
        counter,
        desc: str = "Analyze/resample",
        unit: str = "case",
        poll_interval: float = 0.5,
    ):
        self.counter = counter
        self.poll_interval = poll_interval
        self.completed = 0
        self.pbar = tqdm(total=self.counter.total, desc=desc, unit=unit)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sync(self):
        completed = self.counter.completed_cases()
        delta = completed - self.completed
        if delta > 0:
            self.pbar.update(delta)
            self.completed = completed
        return self.completed

    def _run(self):
        while not self.stop_event.wait(self.poll_interval):
            self.sync()

    def start(self):
        self.sync()
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sync()
        self.pbar.close()


class PreprocessingManager:
    def __init__(self, args, conf=None):
        self.args = args
        P = Project(project_title=args.project_title)
        self.project = P
        if conf is None:
            C = ConfigMaker(P)
            C.setup(args.plan)
            conf = C.configs
        self.plan = conf["plan_train"]
        print("Project: {0}".format(args.project_title))

    def _configure_postproc_artifacts(self, generator):
        if not self.args.create_postproc_artifacts:
            generator.store_gifs = False
            generator.store_label_stats = False
        return generator

    def _process_with_output_progress(
        self,
        generator,
        counter_cls,
        desc="Analyze/resample",
        overwrite=None,
        derive_bboxes=True,
        src_dims=DEFAULT_HDF5_SRC_DIMS,
        cases_per_shard=5,
        max_shard_bytes=None,
        overwrite_hdf5_shards=False,
        hdf5_compression="gzip",
        hdf5_compression_opts=1,
    ):
        monitor = OutputFolderProgressMonitor(
            counter=counter_cls(generator.output_folder, len(generator.df)),
            desc=desc,
        ).start()
        try:
            return generator.process(
                overwrite=overwrite,
                derive_bboxes=derive_bboxes,
                src_dims=src_dims,
                cases_per_shard=cases_per_shard,
                max_shard_bytes=max_shard_bytes,
                overwrite_hdf5_shards=overwrite_hdf5_shards,
                hdf5_compression=hdf5_compression,
                hdf5_compression_opts=hdf5_compression_opts,
            )
        finally:
            monitor.stop()

    def resample_dataset(
        self,
        overwrite=False,
        num_processes=1,
        debug=False,
        src_dims=DEFAULT_HDF5_SRC_DIMS,
        cases_per_shard=5,
        max_shard_bytes=None,
        overwrite_hdf5_shards=False,
        hdf5_compression="gzip",
        hdf5_compression_opts=1,
    ):
        """
        Resamples dataset to target spacing and stores it in the rapid-access fixed_spacing folder.
        Typically this will be a basis for further processing e.g., pbd, lbd dataset which will then be used in training
        """

        self.R = NiftiToTorchDataGenerator(
            project=self.project,
            plan=self.plan,
            data_folder=self.project.raw_data_folder,
        )
        self._configure_postproc_artifacts(self.R)

        self.R.setup(overwrite=overwrite, num_processes=num_processes, debug=debug)
        self._process_with_output_progress(
            self.R,
            ExactCaseOutputCounter,
            desc="Resample",
            overwrite=overwrite,
            src_dims=src_dims,
            cases_per_shard=cases_per_shard,
            max_shard_bytes=max_shard_bytes,
            overwrite_hdf5_shards=overwrite_hdf5_shards,
            hdf5_compression=hdf5_compression,
            hdf5_compression_opts=hdf5_compression_opts,
        )

    def generate_lbd_dataset(
        self,
        overwrite=False,
        device="cpu",
        num_processes=1,
        debug=False,
        src_dims=DEFAULT_HDF5_SRC_DIMS,
        cases_per_shard=5,
        max_shard_bytes=None,
        overwrite_hdf5_shards=False,
        hdf5_compression="gzip",
        hdf5_compression_opts=1,
    ):

        resampled_data_folder = FolderNames(self.project, self.plan).folders[
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
        self._configure_postproc_artifacts(self.L)
        self.L.setup(
            overwrite=overwrite,
            device=device,
            num_processes=num_processes,
            debug=debug,
        )
        self._process_with_output_progress(
            self.L,
            CaseOutputCounter,
            desc="LBD",
            overwrite=overwrite,
            src_dims=src_dims,
            cases_per_shard=cases_per_shard,
            max_shard_bytes=max_shard_bytes,
            overwrite_hdf5_shards=overwrite_hdf5_shards,
            hdf5_compression=hdf5_compression,
            hdf5_compression_opts=hdf5_compression_opts,
        )

    def generate_TSlabelboundeddataset(
        self,
        device="cpu",
        overwrite=False,
        num_processes=1,
        debug=False,
        src_dims=DEFAULT_HDF5_SRC_DIMS,
        cases_per_shard=5,
        max_shard_bytes=None,
        overwrite_hdf5_shards=False,
        hdf5_compression="gzip",
        hdf5_compression_opts=1,
    ):
        """
        requires resampled folder to exist. Crops within this folder
        """

        resampled_data_folder = FolderNames(self.project, self.plan).folders[
            "data_folder_source"
        ]
        self.L = LabelBoundedDataGeneratorImported(
            project=self.project,
            plan=self.plan,
            data_folder=resampled_data_folder,
        )
        self._configure_postproc_artifacts(self.L)
        self.L.setup(
            overwrite=overwrite,
            device=device,
            num_processes=num_processes,
            debug=debug,
        )
        self._process_with_output_progress(
            self.L,
            CaseOutputCounter,
            desc="LBD imported",
            overwrite=overwrite,
            src_dims=src_dims,
            cases_per_shard=cases_per_shard,
            max_shard_bytes=max_shard_bytes,
            overwrite_hdf5_shards=overwrite_hdf5_shards,
            hdf5_compression=hdf5_compression,
            hdf5_compression_opts=hdf5_compression_opts,
        )

    def generate_rbd_dataset(
        self,
        overwrite=False,
        device="cpu",
        num_processes=1,
        debug=False,
        src_dims=DEFAULT_HDF5_SRC_DIMS,
        cases_per_shard=5,
        max_shard_bytes=None,
        overwrite_hdf5_shards=False,
        hdf5_compression="gzip",
        hdf5_compression_opts=1,
    ):

        resampled_data_folder = FolderNames(self.project, self.plan).folders[
            "data_folder_source"
        ]

        headline(
            "RBD dataset will be based on resampled dataset output_folder {}".format(
                resampled_data_folder
            )
        )
        self.L = RegionBoundedDataGenerator(
            project=self.project,
            plan=self.plan,
            data_folder=resampled_data_folder,
        )
        self._configure_postproc_artifacts(self.L)
        self.L.setup(
            overwrite=overwrite,
            device=device,
            num_processes=num_processes,
            debug=debug,
        )
        self._process_with_output_progress(
            self.L,
            CaseOutputCounter,
            desc="RBD",
            overwrite=overwrite,
            src_dims=src_dims,
            cases_per_shard=cases_per_shard,
            max_shard_bytes=max_shard_bytes,
            overwrite_hdf5_shards=overwrite_hdf5_shards,
            hdf5_compression=hdf5_compression,
            hdf5_compression_opts=hdf5_compression_opts,
        )

    def generate_whole_images_dataset(
        self,
        overwrite=False,
        device="cpu",
        num_processes=1,
        debug=False,
        src_dims=DEFAULT_HDF5_SRC_DIMS,
        cases_per_shard=5,
        max_shard_bytes=None,
        overwrite_hdf5_shards=False,
        hdf5_compression="gzip",
        hdf5_compression_opts=1,
    ):
        resampled_data_folder = FolderNames(self.project, self.plan).folders[
            "data_folder_source"
        ]
        headline(
            "Whole dataset will be based on resampled dataset output_folder {}".format(
                resampled_data_folder
            )
        )
        self.W = FixedSizeDataGenerator(
            project=self.project,
            plan=self.plan,
            data_folder=resampled_data_folder,
        )
        self._configure_postproc_artifacts(self.W)
        self.W.setup(
            overwrite=overwrite,
            device=device,
            num_processes=num_processes,
            debug=debug,
        )
        self._process_with_output_progress(
            self.W,
            CaseOutputCounter,
            desc="Whole",
            overwrite=overwrite,
            src_dims=src_dims,
            cases_per_shard=cases_per_shard,
            max_shard_bytes=max_shard_bytes,
            overwrite_hdf5_shards=overwrite_hdf5_shards,
            hdf5_compression=hdf5_compression,
            hdf5_compression_opts=hdf5_compression_opts,
        )

    def generate_hires_patches_dataset(
        self,
        debug=False,
        overwrite=False,
        num_processes=1,
        src_dims=DEFAULT_HDF5_SRC_DIMS,
        cases_per_shard=5,
        max_shard_bytes=None,
        overwrite_hdf5_shards=False,
        hdf5_compression="gzip",
        hdf5_compression_opts=1,
    ):

        data_folder = self.get_source_data_folder_for_patch()
        PG = PatchDataGenerator(
            project=self.project,
            plan=self.plan,
            data_folder=data_folder,
        )
        self._configure_postproc_artifacts(PG)
        PG.setup(
            overwrite=overwrite,
            num_processes=num_processes,
            debug=debug,
        )
        self._process_with_output_progress(
            PG,
            PatchCaseApproxCounter,
            desc="Patches",
            overwrite=overwrite,
            derive_bboxes=False,
            src_dims=src_dims,
            cases_per_shard=cases_per_shard,
            max_shard_bytes=max_shard_bytes,
            overwrite_hdf5_shards=overwrite_hdf5_shards,
            hdf5_compression=hdf5_compression,
            hdf5_compression_opts=hdf5_compression_opts,
        )

    def get_source_data_folder_for_patch(self):
        src_plan = self.plan["source_plan"]
        src_plan_idx, src_plan_mode = src_plan.replace(" ", "").split(",")
        src_plan_idx = int(src_plan_idx)
        C2 = ConfigMaker(self.project)
        C2.setup(src_plan_idx)
        src_plan_full = C2.configs["plan_train"]
        data_fldrs = FolderNames(self.project, src_plan_full).folders
        data_folder = data_fldrs[f"data_folder_{src_plan_mode}"]
        data_foldre = Path(data_folder)
        return data_foldre


# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
if __name__ == "__main__":
    import argparse
    import sys

    from fran.utils.common import *
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
        "--create-postproc-artifacts",
        action="store_true",
        help="Create postprocess artifacts after preprocessing. Default: off.",
    )
    parser.add_argument(
        "--help-args",
        action="store_true",
        help="Show CLI help and exit.",
    )
    args = parser.parse_known_args()[0]
# %%
    # cprint("Warning: Using args saved into file analyze_resample.py", color= "red")
    # args.project_title="tmpts"
    # args.plan = 3
    # # args.project_title = "test"
    # # args.plan = 1
    # args.num_processes = 6
    # args.overwrite = False
    # args.debug = True
    #
# %%
    cprint("Project: {0}".format(args.project_title), color="green")

    if args.help_args:
        parser.print_help()
        raise SystemExit(0)
    main(args)

# %%

    #     I.resample_dataset(overwrite=args.overwrite, num_processes=args.num_processes)
    #     I.plan["mode"]
    #
# %%
    #     I.R = ResampleDatasetniftiToTorch(
    #         project=I.project,
    #         plan=I.plan,
    #         data_folder=I.project.raw_data_folder,
    #     )
    #
# %%
    #     #
    #     overwrite = False
    #     num_processes = 8
    #     I.R.setup(overwrite=overwrite, num_processes=num_processes)
    #     I.R.process()
    #     I.resample_output_folder = I.R.output_folder
    # resampled_data_folder = FolderNames(I.project, I.plan).folders[
    #
    #        "data_folder_source"
    #    ]
# %%
    #
    #         data_folder = I.get_source_data_folder_for_patch()
    #         PG = PatchDataGenerator(
    #             project=I.project,
    #             plan=I.plan,
    #             data_folder=data_folder,
    #         )
    #         PG.setup(
    #             overwrite=overwrite,
    #             num_processes=I.num_processes,
    #             debug=debug,
    #         )
    #         PG.process(derive_bboxes=False)
    #     I.R.create_dataset_stats_artifacts()
    #
# %%
    #     overwrite=False
    #     num_processes=8
    #
    #     resampled_data_folder = FolderNames(I.project, I.plan).folders[
    #         "data_folder_source"
    #     ]
    #
# %%
    #     I.L = LabelBoundedDataGeneratorImported(
    #         project=I.project,
    #         plan=I.plan,
    #         data_folder=resampled_data_folder,
    #     )
# %%
# %%
    #     device='cpu'
    #     I.L.setup(overwrite=overwrite, device=device,num_processes=num_processes,debug=True)
    #     I.L.process()
# %%
    #
    #
# %%
    #     # sys.exit()
    sys.exit()
# %%
# dataset_root = Path(I.R.output_folder)
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
