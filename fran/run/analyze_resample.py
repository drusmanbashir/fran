# %%
import argparse
import ast
import shutil
from fran.preprocessing.imported import LabelBoundedDataGeneratorImported
from fran.utils.config_parsers import ConfigMaker
from fran.managers import Project

from label_analysis.totalseg import TotalSegmenterLabels
from fran.preprocessing.datasetanalyzers import *
from fran.preprocessing.fixed_spacing import ResampleDatasetniftiToTorch
from fran.preprocessing.globalproperties import GlobalProperties
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.preprocessing.patch import PatchDataGenerator, PatchGenerator
from utilz.fileio import *
from utilz.helpers import *
#CODE: implement database such that instead of creating a new copy of lbd / patch etc for every plan, it checks if a similar plan with similar specs exists already, and reuses it For example, patch_size may change in a new plan with LBD mode, but that doesnt needaa new copy of LBD
common_vars_filename = os.environ["FRAN_COMMON_PATHS"]


@str_to_path(0)
def verify_dataset_integrity(folder: Path, debug=False, fix=False):
    """
    folder has subfolders images and masks
    """
    print("Verifying dataset integrity")
    subfolder = list(folder.glob("mask*"))[0]
    args = [[fn, fix] for fn in subfolder.glob("*")]
    res = multiprocess_multiarg(verify_img_label_match, args, debug=debug)
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
    # dont use getattr
    def __init__(self, args):
        self.assimilate_args(args)
        P = Project(project_title=args.project_title)
        self.project = P
        conf = ConfigMaker(P, raytune=False, configuration_filename=None).config

        # args.overwrite=False
        self.plan = conf[self.plan]
        self.plan_name = args.plan
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

    def resample_dataset(self, overwrite=False):
        """
        Resamples dataset to target spacing and stores it in the cold_storage fixed_spacing_folder. 
        Typically this will be a basis for further processing e.g., pbd, lbd dataset which will then be used in training
        """

        self.R = ResampleDatasetniftiToTorch(
            project=self.project,
            spacing=self.plan["spacing"],
            data_folder=self.project.raw_data_folder,
        )
        self.R.setup(overwrite)
        self.R.process()

    def generate_lbd_dataset(self, overwrite=False):
        self.L = LabelBoundedDataGenerator(
            project=self.project,
            plan=self.plan,
            folder_suffix=self.plan_name,
        )
        self.L.setup(overwrite=overwrite)
        self.L.process()
    def generate_TSlabelboundeddataset(
        self,
        imported_labels,
        imported_folder,
        merge_imported_labels=False,
        lm_group="lm_group1",
    ):
        """
        requires resampled folder to exist. Crops within this folder
        """
        imported_folder = Path(imported_folder)

        TSL = TotalSegmenterLabels()
        if imported_labels == "lungs":
            imported_labelsets = TSL.labels("lung", "right"), TSL.labels("lung", "left")
            remapping = TSL.create_remapping(imported_labelsets, [8, 9])
        self.L = LabelBoundedDataGeneratorImported(
            project=self.project,
            expand_by=self.plan["expand_by"],
            spacing=self.plan["spacing"],
            lm_group=lm_group,
            imported_folder=imported_folder,
            imported_labelsets=imported_labelsets,
            merge_imported_labels=merge_imported_labels,
            remapping=remapping,
        )

        self.L.setup()
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
    multiprocess_multiarg(resample_img_mask_tensors, args, debug=False)


# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR>
if __name__ == "__main__":
    from fran.utils.common import *

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

    args.project_title = "nodes"
    P = Project(project_title=args.project_title)

    conf = ConfigMaker(P, raytune=False, configuration_filename=None).config

    plans = conf[args.plan]
# %%
    if not "labels_all" in P.global_properties.keys():
        P.set_lm_groups(plans["lm_groups"])
        P.maybe_store_projectwide_properties(overwrite=True)

# %%
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
        if "imported_folder" not in plans.keys():
            I.generate_lbd_dataset(overwrite=False)
        else:
            I.generate_TSlabelboundeddataset()
# %%

# %%
# SECTION:-------------------- TSL dataset Imported labels-------------------------------------------------------------------------------------- <CR>
# this section uses imported labels from TSL and integrates those into the dataset. 
    assert "imported_folder"  in plans.keys(),"Skip this section, there are no imported labels"
    imported_folder = plans["imported_folder"]
    imported_folder = Path(imported_folder)
    imported_labels = plans["imported_labels"]
    imported_labels = imported_labels.split(".")[1]
    merge_imported_labels = plans["merge_imported_labels"]
    lm_group = plans["lm_groups"]

    TSL = TotalSegmenterLabels()

    if imported_labels == "all":
        imported_labelsets = TSL.labels("lung", "right"), TSL.labels("lung", "left")
        remapping = None

    elif imported_labels == "lungs":
        imported_labelsets = TSL.labels("lung", "right"), TSL.labels("lung", "left")
        remapping = TSL.create_remapping(imported_labelsets, [8, 9])
    else:
        raise NotImplementedError
# %%
    I.L = LabelBoundedDataGeneratorImported(
        project=I.project,
        expand_by=I.plan["expand_by"],
        spacing=I.plan["spacing"],
        lm_group=lm_group,
        imported_folder=imported_folder,
        merge_imported_labels=merge_imported_labels,
        remapping=remapping,
        folder_suffix=I.plan_name,
        fg_indices_exclude=None,
    )

# %%
    I.L.setup()
    I.L.process()

    # I.L.get_tensor_folder_stats()
    # I.L.generate_bboxes()
# %%
# SECTION:-------------------- Troubleshooting-------------------------------------------------------------------------------------- <CR>

    overwrite=False
    I.L = LabelBoundedDataGenerator(
        project=I.project,
        plan=I.plan,
        folder_suffix=I.plan_name,
    )
# %%
    I.L.setup(overwrite=overwrite)
    I.L.process()
# %%
    overwrite = False
    L = LabelBoundedDataGenerator(
        project=I.project,
        expand_by=I.plan["expand_by"],
        spacing=I.plan["spacing"],
        lm_group="lm_group1",
        mask_label=1,
        folder_suffix=I.plan_name,
        fg_indices_exclude=None,
    )
    L.setup(overwrite=overwrite)
    lm_fn = data[0]["lm"]
    data = L.ds.create_data_dicts()

    L.process()
    L.ds[1]

    L.ds.set_transforms("LT,D,E,C,Ind")
# %%
    ds = L.ds
    for dici in data:
        # dici = data[1]
        try:
            print(dici["image"])
            dici = L.ds.LT(dici)
            im, lm = dici["image"], dici["lm"]
            print(lm.unique())
            print("=" * 100)
        except:
            pass
        # ImageMaskViewer([im,lm])
# %%
    dici = L.ds.D(dici)
    dici = L.ds.E(dici)
    dici = L.ds.C(dici)

# %%

    R = I.R
    dici = R.ds.data[0]
    dici = L(dici)

    dici["image"].meta
# %%
    ca = R.ds[0]
# %%

    dici = R(dici)
    dici = L(dici)

    lm = dici["lm"]
# %%

# %%
    overwrite = True
    debug = True
    PG.create_patches(overwrite=overwrite, debug=debug)

# %%
    bb = PG.fixed_sp_bboxes[0]
    args = [
        PG.dataset_properties,
        PG.output_folder,
        PG.patch_size,
        bb,
        patch_overlap,
        PG.expand_by,
        PG.mode,
    ]
# %%
    P = PatchGenerator(
        PG.dataset_properties,
        PG.output_folder,
        PG.patch_size,
        bb,
        patch_overlap,
        expand_by,
        "fg",
    )
# %%
    P.create_patches_from_all_bboxes()

# %%
    PG.generate_bboxes(debug=debug)
# %%
    # I.verify_dataset_integrity()

    # I.analyse_dataset()

# %%
    im1 = "/home/ub/tmp/imgs/litq_72b_20170224_old.pt"
    im2 = "/s/fran_storage/datasets/preprocessed/fixed_spacing/lilun3/spc_074_074_160/images/litq_72b_20170224.pt"
    im1 = torch.load(im1)
    im2 = torch.load(im2)
    ImageMaskViewer([im1, im2], dtypes=["image", "image"])
# %%

    spacing_ind = 0
    patch_overlap = 0.25
    expand_by = 20
    patches_config, patches_output_folder = I.set_patches_config(
        spacing_ind, patch_overlap, expand_by
    )
    PG = PatchDataGenerator(
        I.project, I.fixed_spacing_folder, I.patch_size, **patches_config
    )
    print("Generating boundingbox data")
    PG.generate_bboxes(debug=debug)
# %%

    patch_overlap = 0.25
    expand_by = 20
    patches_config, patches_output_folder = I.set_patches_config(
        0, patch_overlap, expand_by
    )
# %%

    resampling_configs = I.get_resampling_configs()
    spacing_config = resampling_configs[spacing_ind]

    value = spacing_config["spacing"]
# %%
    folder_name_from_list(
        prefix="spc",
        parent_folder=I.project.lbd_folder,
        values_list=value,
    )
# %%
    spacing_ind = 1
    patch_overlap = 0.25
    expand_by = 0
    patches_config, patches_output_folder = I.set_patches_config(
        spacing_ind, patch_overlap, expand_by
    )

    if mode is None:
        mode = I.mode
        PG = PatchDataGenerator(
            I.project, I.lbd_output_folder, I.patch_size, **patches_config, mode=mode
        )

        # ii = "/s/fran_storage/datasets/preprocessed/fixed_spacing/lax/spc_080_080_150/images/lits_5.pt"
        # torch.load(ii).dtype
# %%

        I.get_resampling_configs()
        I.R.create_dl()
        I.R.process()
# %%
