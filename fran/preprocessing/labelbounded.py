# %%
import itertools as il
from pathlib import Path

import pandas as pd
import ray
from fastcore.basics import GetAttr
from fran.configs.parser import is_excel_None
from fran.preprocessing.helpers import (
    create_dataset_stats_artifacts,
    infer_dataset_stats_window,
)
from fran.preprocessing.preprocessor import (
    Preprocessor,
    generate_bboxes_from_lms_folder,
    store_label_count,
)
from fran.preprocessing.rayworker_base import RayWorkerBase
from fran.utils.folder_names import folder_names_from_plan
from utilz.cprint import cprint
from utilz.fileio import maybe_makedirs, np, save_json, tr
from utilz.helpers import pp, resolve_device
from utilz.stringz import headline, info_from_filename

MIN_SIZE = 32  # min size in a single dimension of any image

# plain, testable class (NOT a Ray actor)

import pandas as pd


class _LBDSamplerWorkerBase(RayWorkerBase):
    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder,
        crop_to_label=None,
        device="cpu",
        debug=False,
        tfms_keys="LoadT,Chan,Dev,Crop,Remap,Labels,Indx",
    ):
        super().__init__(
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
            crop_to_label=crop_to_label,
            device=device,
            debug=debug,
            tfms_keys=tfms_keys,
            remapping_key = "remapping_lbd",
        )

    def _create_data_dict(self, row):
        data = {
            "image": row["image"],
            "lm": row["lm"],
            "ds": row["ds"],
            "remapping": row["remapping"],
        }
        return data

    @property
    def indices_subfolder(self):
        fg_indices_exclude = self.plan.get("fg_indices_exclude")
        if fg_indices_exclude is None:
            fg_indices_exclude = []
        elif isinstance(fg_indices_exclude, int):
            fg_indices_exclude = [fg_indices_exclude]
        if len(fg_indices_exclude) > 0:
            indices_subfolder = "indices_fg_exclude_{}".format(
                "".join([str(x) for x in fg_indices_exclude])
            )
        else:
            indices_subfolder = "indices"
        return self.output_folder / indices_subfolder


@ray.remote(num_cpus=1)
class LBDSamplerWorkerImpl(_LBDSamplerWorkerBase):
    pass


class LBDSamplerWorkerLocal(_LBDSamplerWorkerBase):
    pass


class LabelBoundedDataGenerator(Preprocessor, GetAttr):
    # CODE: Preprocessor and downstream classes need thorough review and re-writing. E.g., where does expand_by go, in preprocessor or labelboudned?
    """
    Label-bounded data generator for preprocessing medical imaging data with automatic folder management.

    This class generates preprocessed image and label data by applying transformations and cropping
    based on label boundaries. It automatically manages input and output folder structures based
    on spacing parameters and configuration.

    **Algorithm for Setting Input/Output Folders:**

    1. **Input Data Folder Logic:**
       - If `data_folder` is provided: Uses the specified path directly
       - If `data_folder` is None: Automatically generates path using spacing parameters
         - Creates folder name: "{fixed_spacing_folder}/spc_{spacing[0]}_{spacing[1]}_{spacing[2]}"
         - Example: "/project/data/fixed_spacing/spc_1.5_1.5_1.5"

    2. **Output Folder Logic:**
       - If `output_folder` is provided: Uses the specified path directly
       - If `output_folder` is None: Automatically generates path under project's lbd_folder
         - Base path: "{project.lbd_folder}/spc_{spacing[0]}_{spacing[1]}_{spacing[2]}"

    3. **Folder Structure Created:**
       - `{output_folder}/images/` - Processed image files (.pt format)
       - `{output_folder}/lms/` - Processed label/mask files (.pt format)
       - `{output_folder}/indices/` - Foreground/background indices (.pt format)
       - `{output_folder}/indices_fg_exclude_{labels}/` - When fg_indices_exclude is specified

    The automatic folder naming ensures consistent organization based on processing parameters
    and prevents conflicts between different spacing configurations or processing plans.
    """

    _default = "project"

    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder=None,
        crop_to_label=None,
    ) -> None:
        """
        Initialize the LabelBoundedDataGenerator.

        Args:
            project: Project instance containing paths and configuration
            plan: Processing plan dictionary containing spacing and other parameters
            data_folder: Path to input data folder. If None, auto-generated from spacing
            output_folder: Path to output folder. If None, auto-generated under project.lbd_folder
            mask_label: Specific label value to use for cropping. If None, uses all labels >0
        """

        existing_fldr = folder_names_from_plan(project, plan)["data_folder_lbd"]
        existing_fldr = Path(existing_fldr)
        if existing_fldr.exists():
            headline(
                "Plan folder already exists: {}.\nWill use existing folder to add data".format(
                    existing_fldr
                )
            )
            output_folder = existing_fldr
        self.plan = plan
        self.lm_group = self.plan.get("lm_group")
        # self.remapping = self.create_remapping_dict(plan["remapping"])

        if is_excel_None(self.lm_group):
            self.lm_group = "lm_group1"
        self.remapping_key = "remapping_lbd"
        Preprocessor.__init__(
            self,
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
        )
        self.actor_cls = LBDSamplerWorkerImpl
        self.local_worker_cls = LBDSamplerWorkerLocal

    def create_data_df(self):
        Preprocessor.create_data_df(self)
        remapping = self.plan.get(self.remapping_key)
        self.df = self.df.assign(remapping=[remapping] * len(self.df))

    def set_input_output_folders(self, data_folder, output_folder):
        self.data_folder = Path(data_folder)
        if output_folder is None:
            lbd_subfolder = folder_names_from_plan(self.project, self.plan)[
                "data_folder_lbd"
            ]
            self.output_folder = Path(lbd_subfolder)
        else:
            self.output_folder = Path(output_folder)

        cprint(f"Data folder is {self.data_folder}", color="yellow")

    def create_output_folders(self):
        maybe_makedirs(
            [
                self.output_folder / ("lms"),
                self.output_folder / ("images"),
                self.indices_subfolder,
            ]
        )

    def process(self, derive_bboxes=True):
        return super().process(derive_bboxes=derive_bboxes)

    def postprocess_results(self, **process_kwargs):
        derive_bboxes = process_kwargs["derive_bboxes"]
        ts = self.results_df.shape
        if derive_bboxes and ts[-1] == 4:  # only store if entire dset is processed
            self._store_dataset_properties()
            generate_bboxes_from_lms_folder(
                self.output_folder / ("lms"),
                num_processes=getattr(self, "num_processes", 1),
            )
        elif derive_bboxes == False:
            print("No bboxes generated")
        else:
            print(
                "self.results  shape is {0}. Last element should be 4 , is {1}. therefore".format(
                    ts, ts[-1]
                )
            )
            print(
                "since some files skipped, dataset stats are not being stored. run self.get_tensor_folder_stats and generate_bboxes_from_lms_folder separately"
            )

        self.store_label_count()
        self.results_df.to_csv(
            self.output_folder / "resampled_dataset_properties.csv", index=False
        )
        create_dataset_stats_artifacts(
            output_folder=self.output_folder,
            gif=self.store_gifs,
            label_stats=self.store_label_stats,
            gif_window=infer_dataset_stats_window(self.project),
        )

    def store_label_count(self):
        try:
            labels_all = self.results_df["labels"].sum()
            labels_all = set(labels_all)
            labels_all = list(labels_all)
            out_fn = self.output_folder / "labels_all.json"
            save_json(labels_all, out_fn)
        except:
            store_label_count(self.output_folder, num_processes=6)

    def setup(self, num_processes=8, device="cpu", overwrite=True, debug=False):
        self.setup_workers(
            overwrite=overwrite,
            num_processes=num_processes,
            device=device,
            debug=debug,
        )

    def create_properties_dict(self):
        resampled_dataset_properties = Preprocessor.create_properties_dict(self)
        ignore_keys = [
            "remapping_train",
            "mode",
            "spacing",
            "samples_per_file",
            "remapping_train",
        ]
        for key in self.plan.keys():
            if not key in ignore_keys:
                resampled_dataset_properties[key] = self.plan[key]
        return resampled_dataset_properties

    def get_case_ids_lm_group(self, lm_group):
        dsrcs = self.global_properties[lm_group]["ds"]
        cids = []
        for dsrc in dsrcs:
            cds = self.df["case_id"][self.df["ds"] == dsrc].to_list()
            cids.extend(cds)
        return cids

    @property
    def indices_subfolder(self):
        fg_indices_exclude = self.plan.get("fg_indices_exclude")
        if fg_indices_exclude is None:
            fg_indices_exclude = []
        elif isinstance(fg_indices_exclude, int):
            fg_indices_exclude = [fg_indices_exclude]
        if len(fg_indices_exclude) > 0:
            indices_subfolder = "indices_fg_exclude_{}".format(
                "".join([str(x) for x in fg_indices_exclude])
            )
        else:
            indices_subfolder = "indices"
        indices_subfolder = self.output_folder / indices_subfolder
        return indices_subfolder


class FGBGIndicesLBD(LabelBoundedDataGenerator):
    """
    Outputs FGBGIndices only. No images of lms are created.
    Use this generator when LBD images and lms are already created, but a new set of FG indices is required.
    """

    def __init__(self, project, data_folder, fg_indices_exclude: list = None) -> None:
        self.project = project
        self.data_folder = data_folder
        self.fg_indices_exclude = fg_indices_exclude
        self.data_folder = Path(data_folder)
        self.output_folder = Path(data_folder)

    def register_existing_files(self):
        self.existing_output_fnames = {p.name for p in self.indices_subfolder.glob("*pt")}
        print("Output folder: ", self.output_folder)
        print(
            "Index files fully processed in a previous session: ",
            len(self.existing_output_fnames),
        )

    def setup(self, device="cpu", batch_size=4, overwrite=False):
        device = resolve_device(device)
        print("Processing on ", device)
        self.register_existing_files()
        print("Overwrite:", overwrite)

        if not overwrite:
            self.remove_completed_cases()

        print("Cases remaining to process: ", len(self.df))
        # Setup file lists and transforms
        if len(self.df) > 0:
            self.image_files = sorted(self.data_folder.glob("images/*.pt"))
            self.lm_files = sorted(self.data_folder.glob("lms/*.pt"))
            self.transforms = self.create_transforms(device)
        else:
            print("No cases to process.")

    def create_output_folders(self):
        maybe_makedirs(self.indices_subfolde)


if __name__ == "__main__":
    from fran.configs.parser import ConfigMaker

    # %%
    # SECTION:-------------------- setup-------------------------------------------------------------------------------------- <CR> <CR>
    from fran.managers import Project
    from fran.utils.common import *

    project_title = "lidc"
    P = Project(project_title=project_title)
    # P.maybe_store_projectwide_properties()
    # spacing = [1.5, 1.5, 1.5]

    C = ConfigMaker(P)
    C.setup(8)
    C.plans
    conf = C.configs
    print(conf["model_params"])

    plan = conf["plan_train"]
    pp(plan)
    spacing = plan["spacing"]
    # plan["remapping_imported"][0]
    existing_fldr = folder_names_from_plan(P, plan).get("data_folder_source", None)
    # %%

    num_processes = 4
    L = LabelBoundedDataGenerator(project=P, plan=plan, data_folder=existing_fldr)

    # %%
    overwrite = False
    num_processes = 5
    debug_ = False
    L.setup(
        overwrite=overwrite, device="cpu", num_processes=num_processes, debug=debug_
    )
    L.process()
    # %%
    # %%
    L.mini_dfs = L.split_dataframe_for_workers(L.df, num_processes)
    mini_df = L.mini_dfs[0].iloc[:3]
    # %%
    overwrite = False
    LL = LBDSamplerWorkerImpl(
        project=L.project,
        plan=L.plan,
        data_folder=L.data_folder,
        output_folder=L.output_folder,
    )
    LL.process(mini_df)
    # %%
    # %%
    row = mini_df.iloc[1]
    data = {
        "image": row["image"],
        "lm": row["lm"],
        "remapping": row["remapping"],
    }
    # %%
    data["image"]

    # Apply transforms
    data = LL.transforms(data)
    image = data["image"]
    lm = (data["lm"],)
    lm_fg_indices = data["lm_fg_indices"]
    lm_bg_indices = data["lm_bg_indices"]
    # Get metadata and indices
    # Process the case
    assert image.shape == lm.shape, "mismatch in shape"
    assert image.dim() == 4, "images should be cxhxwxd"
    assert image.numel() > MIN_SIZE**3, f"image size is too small {image.shape}"

    inds = {
        "lm_fg_indices": lm_fg_indices,
        "lm_bg_indices": lm_bg_indices,
        "meta": image.meta,
    }

    LL.save_indices(inds, LL.indices_subfolder)
    LL.save_pt(image[0], "images")
    LL.save_pt(lm[0], "lms")
    LL.extract_image_props(image)
    results = {
        "case_id": row.get("case_id"),
        "ok": True,
        "shape": list(image.shape),
    }

    d
    # %%

    # %%
    # SECTION:-------------------- TS-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR>

    # for img_file, lm_file in zip(I.L.image_files, I.L.lm_files):
    row = L.df.iloc[0]
    img_file, lm_file = L.image_files[1], L.lm_files[1]
    # Load and process single case
    remapping = row["remapping"]
    src = list(remapping.keys())
    dest = list(remapping.values())

    # %%
    row = L.df.iloc[0]
    data = {
        "image": img_file,
        "lm": lm_file,
        "remapping": row["remapping"],
    }

    # %%
    # Apply transforms

    # self.tfms_keys = "LoadT,Chan,Dev,Crop,Remap,Indx"
    data = L.transforms_dict["LoadT"](data)
    data = L.transforms_dict["Chan"](data)
    data = L.transforms_dict["Dev"](data)
    data = L.transforms_dict["Crop"](data)
    data = L.transforms_dict["Remap"][data["ds"]](data)
    data = L.transforms(data)

    # Get metadata and indices
    fg_indices = L.get_foreground_indices(data["lm"])
    bg_indices = L.get_background_indices(data["lm"])
    coords = L.get_foreground_coords(data["lm"])

    # Process the case
    L.process_single_case(
        data["image"],
        data["lm"],
        fg_indices,
        bg_indices,
        coords["start"],
        coords["end"],
    )

    # %%
    for index, row in L.df.iterrows():
        print(row)
        tr()
        # %%
        remap = I.L.plan["remapping"]
        I.L.df = I.L.df.assign(remapping=[remap] * len(I.L.df))

    # %%
    L.results_df = pd.DataFrame(il.chain.from_iterable(L.results))
    # %%

    derive_bboxes = False
    ts = L.results_df.shape
    if derive_bboxes and ts[-1] == 4:  # only store if entire dset is processed
        L._store_dataset_properties()
        generate_bboxes_from_lms_folder(L.output_folder / ("lms"))
    elif derive_bboxes == False:
        print("No bboxes generated")
    else:
        print(
            "L.results  shape is {0}. Last element should be 4 , is {1}. therefore".format(
                ts, ts[-1]
            )
        )
        print(
            "since some files skipped, dataset stats are not being stored. run L.get_tensor_folder_stats and generate_bboxes_from_lms_folder separately"
        )
    # %%
    add_plan_to_db(
        L.project, L.plan, db_path=L.project.db, data_folder_lbd=L.output_folder
    )

    # %%

    add_plan_to_db(
        L.project, I.L.plan, db_path=I.L.project.db, data_folder_lbd=I.L.output_folder
    )
    # %%
    add_plan_to_db(
        L.project,
        L.plan,
        db_path=L.project.db,
        data_folder_source=L.data_folder,
        data_folder_lbd=L.output_folder,
    )
    # %%

    output_fldr = L.output_folder

    store_label_count(output_fldr)
# %%
