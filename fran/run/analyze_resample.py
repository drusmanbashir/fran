# %%
import sys
import argparse
import ast
import shutil

from label_analysis.totalseg import TotalSegmenterLabels
from utilz.fileio import *
from utilz.helpers import *
from utilz.string import headline

from fran.managers import Project

from fran.data.dataregistry import DS
from fran.managers.db import  find_matching_plan
from fran.preprocessing.datasetanalyzers import *
from fran.preprocessing.fixed_spacing import ResampleDatasetniftiToTorch
from fran.preprocessing.globalproperties import GlobalProperties
from fran.preprocessing.imported import LabelBoundedDataGeneratorImported
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.preprocessing.patch import PatchDataGenerator, PatchGenerator
from fran.configs.parser import ConfigMaker
from fran.utils.folder_names import folder_names_from_plan

common_vars_filename = os.environ["FRAN_COMMON_PATHS"]

def main(args):
    P = Project(project_title=args.project_title)
    # P.create(mnemonic= "litsmall")
    # P.add_data([DS["litsmall"]])
    # P.create(mnemonic="lidc")
    # P.create(mnemonic="lidc", datasources=[DS["lidc"]])
    # P.add_data([DS["lidc"]])
    # P.maybe_store_projectwide_properties()

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
    #
    # if not "labels_all" in P.global_properties.keys():
    #     P.set_lm_groups(plan["lm_groups"])
        # P.maybe_store_projectwide_properties(overwrite=args.overwrite)



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


class PreprocessingManager:
    # Declare attributes that assimilate_args will set
    project_title: str
    num_processes: int
    overwrite: bool
    debug:bool
    # dont use getattr
    def __init__(self, args):
        self.assimilate_args(args)
        self.num_processes = args.num_processes
        P = Project(project_title=args.project_title)
        self.project = P
        C = ConfigMaker(P,  configuration_filename=None)
        C.setup(args.plan)

        conf = C.configs
        self.plan = conf["plan_train"]
        # self.plan['spacing'] = ast.literal_eval(self.plan['spacing'])

        #
        print("Project: {0}".format(self.project_title))

    def __str__(self) -> str:
        plan_details = "\n".join(
            [f"{key}: {value}" for key, value in self.plan.items()]
        )
        return f"PreprocessingManager. Project: {self.project_title}\nPlan Details:\n{plan_details}"

    def verify_dataset_integrity(self):
        verify_dataset_integrity(
            self.project.raw_data_folder, debug=self.debug, fix=not self.no_fix
        )

    def analyse_dataset(self):
        if self._analyse_dataset_questions() == True:
            self.GlobalP = GlobalProperties(
                self.project, bg_label=0, clip_range=self.clip_range
            )
            self.GlobalP.store_projectwide_properties()
            self.GlobalP.compute_std_mean_dataset(debug=self.debug)
            self.GlobalP.collate_lm_labels()

    def _analyse_dataset_questions(self):

        global_properties = load_dict(self.project.global_properties_filename)
        if not "total_voxels" in global_properties.keys():
            return True
        else:
            reanalyse = input(
                "Dataset global properties already computed. Re-analyse dataset (Y/y)?"
            )
            if reanalyse.lower() == "y":
                return True

    def resample_dataset(self, overwrite=False, num_processes=1):
        """
        Resamples dataset to target spacing and stores it in the cold_storage fixed_spacing_folder.
        Typically this will be a basis for further processing e.g., pbd, lbd dataset which will then be used in training
        """

        self.R = ResampleDatasetniftiToTorch(
            project=self.project,
            plan=self.plan,
            data_folder=self.project.raw_data_folder,
        )

        self.R.setup(overwrite=overwrite, num_processes=num_processes)
        self.R.process()
        self.resample_output_folder = self.R.output_folder

    def generate_lbd_dataset(self, overwrite=False, device="cpu",num_processes=1):

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
        self.L.setup(overwrite=overwrite, device=device, num_processes=num_processes)
        self.L.process()

    def generate_TSlabelboundeddataset(
        self,
        device="cpu",
        overwrite=False,
        num_processes=1,
    ):
        """
        requires resampled folder to exist. Crops within this folder
        """

        resampled_data_folder = folder_names_from_plan(self.project, self.plan)[
            "data_folder_source"
        ]
        # Path(imported_folder)
        self.L = LabelBoundedDataGeneratorImported(
            project=self.project,
            plan=self.plan,
            data_folder=resampled_data_folder,
        )
        self.L.setup(overwrite=overwrite, device=device,num_processes=num_processes)
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
        self.WholeImageTM = WholeImageTensorMaker(
            self.project,
            source_spacing=self.plan["spacing"],
            output_size=output_shape,
            num_processes=self.num_processes,
        )
        arglist_imgs, arglist_masks = self.WholeImageTM.get_args_for_resizing()
        for arglist in [arglist_imgs, arglist_masks]:
            res = multiprocess_multiarg(
                func=resize_and_save_tensors,
                arguments=arglist,
                num_processes=self.num_processes,
                debug=self.debug,
                io=True,
            )
        print("Now call bboxes_from_masks_folder")
        generate_bboxes_from_masks_folder(
            self.WholeImageTM.output_folder_masks, 0, self.debug, self.num_processes
        )

    def generate_hires_patches_dataset(self, debug=False, overwrite=False, mode=None):
        lbd_folder = folder_name_from_list(
            prefix="spc",
            parent_folder=self.project.lbd_folder,
            values_list=self.plan["spacing"],
        )
        patch_overlap = self.plan["patch_overlap"]
        expand_by = self.plan["expand_by_patch"]

        if mode is None:
            mode = self.mode
        # BUG: Throws file not found error, multiple bugs that need fixsing  (see #10)
        PG = PatchDataGenerator(
            self.project,
            lbd_folder,
            self.patch_size,
            patch_overlap=patch_overlap,
            expand_by=expand_by,
            mode=mode,
        )
        PG.create_patches(overwrite=overwrite, debug=debug)
        print("Generating boundingbox data")
        PG.generate_bboxes(debug=debug)

        resampled_dataset_properties_fn_org = lbd_folder / (
            "resampled_dataset_properties.json"
        )
        resampled_dataset_properties_fn_dest = (
            PG.output_folder.parent / resampled_dataset_properties_fn_org.name
        )
        if not resampled_dataset_properties_fn_dest.exists():
            shutil.copy(
                resampled_dataset_properties_fn_org,
                resampled_dataset_properties_fn_dest,
            )

    def create_patches_output_folder(self, fixed_spacing_folder, patch_size):

        patches_fldr_name = "dim_{0}_{1}_{2}".format(*patch_size)
        output_folder = (
            self.project.patches_folder / fixed_spacing_folder.name / patches_fldr_name
        )
        # maybe_makedirs(output_folder)
        return output_folder

    def assimilate_args(self, args):
        for key, value in vars(args).items():
            setattr(self, key, value)

    def maybe_change_default_spacing(self, vals):
        def _accept_defaults():
            print("Accepting defaults")

        try:
            if isinstance(vals, str):
                vals = ast.literal_eval(vals)
            if all([isinstance(vals, (list, tuple)), len(vals) == 3]):
                self.R.spacing = vals
            elif isinstance(vals, (int, float)):
                vals = [
                    vals,
                ] * 3
                self.R.spacing = vals
            else:
                _accept_defaults()
        except:
            _accept_defaults()

    def get_resampling_config(self, spacing):
        resamping_config_fn = self.project.fixed_spacing_folder / (
            "resampling_configs.json"
        )
        resampling_configs = load_dict(resamping_config_fn)
        for config in resampling_configs:
            if spacing == config["spacing"]:
                return config
        raise ValueError(
            "No resampling config found for this spacing: {0}. \nAll configs are:\n{1}".format(
                spacing, resampling_configs
            )
        )

    @property
    def resampling_configs(self):
        return self.get_resampling_configs()


def do_resempling(R, args):
    dim0 = input("Change dim0 to (press enter to leave unchanged)")
    dim1 = input("Change dim2 to (press enter to leave unchanged)")
    dim2 = input("Change dim3 to (press enter to leave unchanged)")
    spacing = [
        float(a) if len(a) > 0 else b for a, b in zip([dim0, dim1, dim2], R.spacing)
    ]
    R.spacing = spacing
    R.resample_cases(debug=False, overwrite=args.overwrite, multiprocess=True)


def do_low_res(proj_defaults):
    low_res_shape = get_list_input(
        text="Enter low-res image shape (e.g., '128,128,128')", fnc=str_to_list_int
    )
    stage0_files = list(Path(proj_defaults.stage0_folder / "volumes").glob("*.pt"))

    stage1_subfolder = (
        proj_defaults.stage1_folder
        / str(low_res_shape).strip("[]").replace(", ", "_")
        / "volumes"
    )
    maybe_makedirs(stage1_subfolder)

    args = [[fn, stage1_subfolder, low_res_shape, False] for fn in stage0_files]
    multiprocess_multiarg(resample_img_mask_tensors, args, debug=False,io=True)



# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
if __name__ == "__main__":
    from fran.utils.common import *

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
    parser.add_argument("-d", "--debug", action="store_true")

    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_known_args()[0]
# %%
    # args.project_title="lidc"
    # args.plan = 1
    # args.num_processes = 8
    # args.overwrite=True
    # args.debug=True
    #
#python  analyze_resample.py -t nodes -p 6 -n 4 -o


# %%
#     resampled_data_folder = folder_names_from_plan(I.project, I.plan)[
#         "data_folder_source"
#     ]
#     
#     headline(
#         "LBD dataset will be based on resampled dataset output_folder {}".format(
#             resampled_data_folder
#         )
#     )
#     I.L = LabelBoundedDataGenerator(
#         project=I.project,
#         plan=I.plan,
#         data_folder=resampled_data_folder,
#     )
# # %%
#     overwrite=False
#     num_processes=4
#     device="cpu"
#     I.L.setup(overwrite=overwrite, device=device, num_processes=num_processes)
#     I.L.process()
# # %%
    main(args)
    # sys.exit()
# %%
