# %%
import pandas as pd
import itertools as il
from pathlib import Path

import pandas as pd
from fastcore.basics import GetAttr, store_attr
from utilz.fileio import *
from utilz.helpers import *
from utilz.imageviewers import *
from utilz.string import headline, info_from_filename

from fran.preprocessing.preprocessor import (Preprocessor,
                                             generate_bboxes_from_lms_folder)
from fran.preprocessing.rayworker_base import RayWorkerBase
from fran.utils.config_parsers import (ConfigMaker, is_excel_None)
from fran.utils.folder_names import folder_names_from_plan

MIN_SIZE = 32  # min size in a single dimension of any image

# plain, testable class (NOT a Ray actor)

import pandas as pd



@ray.remote(num_cpus=1)
class LBDSamplerWorkerImpl(RayWorkerBase):

    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder,
        crop_to_label=None,
        device="cpu",
        tfms_keys  = "LT,E,D,C,R,Ind",
    ):
        super().__init__(
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
            crop_to_label=crop_to_label,
            device=device,
            tfms_keys=tfms_keys
        )


    def _create_data_dict(self,row):

        data = {
            "image": row["image"],
            "lm": row["lm"],
            "remapping": row["remapping"],
        }
        return data
    #     super().__init__(
    #         project=project,
    #         plan=plan,
    #         data_folder=data_folder,
    #         output_folder=output_folder,
    #     )
    #
    #     self.crop_to_label = crop_to_label  # redundant
    #     self.image_key = "image"
    #     self.lm_key = "lm"
    #     self.tnsr_keys = [self.image_key, self.lm_key]
    #     self.create_transforms(device=device)
    #     self.set_transforms(tfms_keys)
    #
    # def _process_row(self, row: pd.Series) -> Dict[str, Any]:
    #     data = {
    #         "image": row["image"],
    #         "lm": row["lm"],
    #         "remapping": row["remapping"],
    #     }
    #
    #     # Apply transforms
    #     data = self.transforms(data)
    #     image = data["image"]
    #     lm = (data["lm"])
    #     lm_fg_indices = data["lm_fg_indices"]
    #     lm_bg_indices = data["lm_bg_indices"]
    #     # Get metadata and indices
    #     # Process the case
    #     assert image.shape == lm.shape, "mismatch in shape"
    #     assert image.dim() == 4, "images should be cxhxwxd"
    #     assert image.numel() > MIN_SIZE**3, f"image size is too small {image.shape}"
    #
    #     inds = {
    #         "lm_fg_indices": lm_fg_indices,
    #         "lm_bg_indices": lm_bg_indices,
    #         "meta": image.meta,
    #     }
    #
    #     self.save_indices(inds, self.indices_subfolder)
    #     self.save_pt(image[0], "images")
    #     self.save_pt(lm[0], "lms")
    #     self.extract_image_props(image)
    #     results = {
    #         "case_id": row.get("case_id"),
    #         "ok": True,
    #         "shape": list(image.shape),
    #     }
    #     return results
    # def set_transforms(self, keys_tr: str):
    #     self.transforms = self.tfms_from_dict(keys_tr)
    #
    # def tfms_from_dict(self, keys: str):
    #     keys = keys.replace(" ", "").split(",")
    #     tfms = []
    #     for key in keys:
    #         tfm = self.transforms_dict[key]
    #         tfms.append(tfm)
    #     tfms = Compose(tfms)
    #     return tfms
    #
    # def set_input_output_folders(self, data_folder, output_folder):
    #     self.data_folder = data_folder
    #     self.output_folder = output_folder
    #
    # def _create_data_dicts_from_df(self, df):
    #     """Create data dictionaries from DataFrame."""
    #     data = []
    #     for index in range(len(df)):
    #         row = df.iloc[index]
    #         dici = self._dici_from_df_row(row, remapping)
    #         data.append(dici)
    #
    # def create_transforms(self, device):
    #     if self.plan["expand_by"]:
    #         margin = [int(self.plan["expand_by"] / sp) for sp in self.plan["spacing"]]
    #     else:
    #         margin = 0
    #     if self.crop_to_label is None:
    #         select_fn = is_positive
    #     else:
    #         select_fn = lambda lm: lm == self.crop_to_label
    #     # Transform attributes in alphabetical order
    #     self.C = CropForegroundd(
    #         keys=self.tnsr_keys,
    #         source_key=self.lm_key,
    #         select_fn=select_fn,
    #         allow_smaller=True,
    #         margin=margin,
    #     )
    #     self.D = ToDeviced(device=device, keys=self.tnsr_keys)
    #     self.E = EnsureChannelFirstd(keys=self.tnsr_keys, channel_dim="no_channel")
    #     self.Ind = FgBgToIndicesd2(
    #         keys=[self.lm_key],
    #         image_key=self.image_key,
    #         ignore_labels=self.plan["fg_indices_exclude"],
    #         image_threshold=-2600,
    #     )
    #     self.LT = LoadTorchd(keys=[self.image_key, self.lm_key])
    #     if self.plan["remapping_lbd"] is not None:
    #         self.R = MapLabelValueD(
    #             keys=[self.lm_key],
    #             orig_labels=self.plan["remapping_lbd"][0],
    #             target_labels=self.plan[remapping_lbd][1],
    #         )
    #     else:
    #         self.R = DummyTransform(keys=[self.lm_key])
    #     # )
    #     self.transforms_dict = {
    #         "C": self.C,
    #         "D": self.D,
    #         "E": self.E,
    #         "Ind": self.Ind,
    #         "LT": self.LT,
    #         "R": self.R,
    #     }
    #
    # def process(self, mini_df):
    #     outs = []
    #     for i,row in mini_df.iterrows():
    #         try:
    #             outs.append(self._process_row(row))
    #         except Exception as e:
    #             
    #             img_fn = row.get("image")
    #             print(f"[{self.__class__.__name__}] error: {img_fn}: {e}")
    #             outs.append({"case_id": row.get("case_id"), "ok": False, "err": str(e)})
    #     return outs
    #
    #

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

        existing_fldr = folder_names_from_plan(project, plan).get("data_folder_lbd"
        )
        existing_fldr=Path(existing_fldr)
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

    def create_data_df(self):
        Preprocessor.create_data_df(self)
        remapping = self.plan.get(self.remapping_key)
        self.df = self.df.assign(remapping=[remapping] * len(self.df))

    def set_input_output_folders(self, data_folder, output_folder):
        self.data_folder = Path(data_folder)
        if output_folder is None:
            lbd_subfolder = folder_names_from_plan(self.project,self.plan)["data_folder_lbd"]
            self.output_folder = self.project.lbd_folder / (lbd_subfolder)
        else:
            self.output_folder = Path(output_folder)

    def create_output_folders(self):
        maybe_makedirs(
            [
                self.output_folder / ("lms"),
                self.output_folder / ("images"),
                self.indices_subfolder,
            ]
        )

    def process(self, derive_bboxes=False):
        if not hasattr(self, "df") or len(self.df) == 0:
            print("No data loader created. No data to be processed")
            return 0
        self.create_output_folders()
        self.results = []
        self.shapes = []
        # self.results= pd.DataFrame(self.results).values

        self.results = ray.get(
            [
                actor.process.remote(mini_df)
                for actor, mini_df in zip(self.actors, self.mini_dfs)
            ]
        )

        self.results_df = pd.DataFrame(il.chain.from_iterable(self.results))
        ts = self.results_df.shape
        if derive_bboxes and ts[-1] == 4:  # only store if entire dset is processed
            self._store_dataset_properties()
            generate_bboxes_from_lms_folder(self.output_folder / ("lms"))
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
        # add_plan_to_db(self.project,
        #     self.plan, db_path=self.project.db, data_folder_lbd=self.output_folder
        # )

    def setup(self, num_processes=8, device="cpu", overwrite=True):

        self.create_data_df()
        self.register_existing_files()
        print("Overwrite:", overwrite)
        if overwrite == False:
            self.remove_completed_cases()
        if len(self.df) > 0:
            self.mini_dfs = np.array_split(self.df, num_processes)

            self.n_actors = min(len(self.df), int(num_processes))
            # (Optionally) initialise Ray if not already
            if not ray.is_initialized():
                try:
                    ray.init(ignore_reinit_error=True)
                except Exception as e:
                    print("Ray init warning:", e)

            actor_kwargs = dict(
                project=self.project,
                plan=self.plan,
                data_folder=self.data_folder,
                output_folder=self.output_folder,
                device=device,
            )
            self.actors = [
                self.actor_cls.remote(**actor_kwargs)
                for _ in range(self.n_actors)
            ]
            # self.mini_dfs = list_to_chunks(self.df, num_processes)

    # def process_files(self, force_store_props=False):
    #     """Process files without using DataLoader"""
    #     self.create_output_folders()
    #     self.results = []
    #     self.shapes = []
    #     # for img_file, lm_file in pbar(zip(self.image_files, self.lm_files), desc="Processing files", total=len(self.image_files)):
    #     for index, row in pbar(self.df.iterrows(), total=len(self.df)):
    #         try:
    #             # Load and process single case
    #             data = {
    #                 "image": row["image"],
    #                 "lm": row["lm"],
    #                 "remapping": row["remapping"],
    #             }
    #
    #             # Apply transforms
    #             data = self.transforms(data)
    #
    #             # Get metadata and indices
    #             # Process the case
    #             self.process_single_case(
    #                 data["image"],
    #                 data["lm"],
    #                 data["lm_fg_indices"],
    #                 data["lm_bg_indices"],
    #             )
    #
    #         except Exception as e:
    #             print(f"Error processing {row['image'].name}: {str(e)}")
    #             continue
    #
    #     self.results_df = pd.DataFrame(self.results)
    #     # self.results= pd.DataFrame(self.results).values
    #     ts = self.results_df.shape
    #     if (
    #         ts[-1] == 4 or force_store_props == True
    #     ):  # only store if entire dset is processed
    #         self._store_dataset_properties()
    #         generate_bboxes_from_lms_folder(self.output_folder / ("lms"))
    #     else:
    #         print(
    #             "self.results  shape is {0}. Last element should be 4 , is {1}. therefore".format(
    #                 ts, ts[-1]
    #             )
    #         )
    #         print(
    #             "since some files skipped, dataset stats are not being stored. Either:\na) set force_store_props to True, or\nb) run self.get_tensor_folder_stats and generate_bboxes_from_lms_folder separately"
    # )

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

        # def process_single_case(self, image, lm, fg_inds, bg_inds):
        #     """Process a single case and save results"""
        #     assert image.shape == lm.shape, "mismatch in shape"
        #     assert image.dim() == 4, "images should be cxhxwxd"
        #     assert image.numel() > MIN_SIZE**3, f"image size is too small {image.shape}"
        #
        #     inds = {
        #         "lm_fg_indices": fg_inds,
        #         "lm_bg_indices": bg_inds,
        #         "meta": image.meta,
        #     }
        #
        #     self.save_indices(inds, self.indices_subfolder)
        #     self.save_pt(image[0], "images")
        #     self.save_pt(lm[0], "lms")
        self.extract_image_props(image)

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
        if len(fg_indices_exclude) > 0:
            indices_subfolder = "indices_fg_exclude_{}".format(
                "".join([str(x) for x in self.plan["fg_indices_exclude"]])
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
        store_attr()
        self.data_folder = Path(data_folder)
        self.output_folder = Path(data_folder)

    def register_existing_files(self):
        self.existing_files = list(
            (self.output_folder / self.indices_subfolder).glob("*pt")
        )
        self.existing_case_ids = [
            info_from_filename(f.name, full_caseid=True)["case_id"]
            for f in self.existing_files
        ]
        self.existing_case_ids = set(self.existing_case_ids)
        print("Case ids processed in a previous session: ", len(self.existing_case_ids))

    def setup(self, device="cpu", batch_size=4, overwrite=False):
        device = resolve_device(device)
        print("Processing on ", device)
        self.register_existing_files()
        print("Overwrite:", overwrite)

        if not overwrite:
            self.remove_completed_cases()

        # Setup file lists and transforms
        if len(self.df) > 0:
            self.image_files = sorted(self.data_folder.glob("images/*.pt"))
            self.lm_files = sorted(self.data_folder.glob("lms/*.pt"))
            self.transforms = self.create_transforms(device)
        else:
            print("No cases to process.")

    def create_output_folders(self):
        maybe_makedirs(self.indices_subfolder)


if __name__ == "__main__":
# %%
# %%
#SECTION:-------------------- setup--------------------------------------------------------------------------------------

    from fran.managers import Project
    from fran.utils.common import *

    project_title = "totalseg"
    P = Project(project_title=project_title)
    # P.maybe_store_projectwide_properties()
    # spacing = [1.5, 1.5, 1.5]

    C = ConfigMaker(P, raytune=False, configuration_filename=None)
    C.setup(3)
    C.plans
    conf = C.configs
    print(conf["model_params"])

    plan = conf["plan_train"]
    pp(plan)
    spacing = plan["spacing"]
    # plan["remapping_imported"][0]
    existing_fldr = folder_names_from_plan(P, plan).get("data_folder_lbd", None)
# %%

# %%
    num_processes=16
    L = LabelBoundedDataGenerator(
        project=P,
        plan=plan,
        data_folder="/r/datasets/preprocessed/lidc/fixed_spacing/spc_080_080_150_ldc"
    )

# %%
    L.setup(overwrite=False, device="cpu")
# %%
    L.mini_dfs = np.array_split(L.df, num_processes)
    mini_df = L.mini_dfs[0].iloc[:3]
# %%
    overwrite = False
    LL = LBDSamplerWorkerImpl(project=L.project, plan=L.plan, data_folder=L.data_folder, output_folder=L.output_folder)
    LL.process(mini_df )
# %%
# %%
    row = mini_df.iloc[1]
    data = {
        "image": row["image"],
        "lm": row["lm"],
        "remapping": row["remapping"],
    }
# %%
    data['image']

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
# SECTION:-------------------- TS-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR>

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

    # self.tfms_keys = "LT,E,D,C,R,Ind"
    data = L.transforms_dict["LT"](data)
    data = L.transforms_dict["E"](data)
    data = L.transforms_dict["D"](data)
    data = L.transforms_dict["C"](data)
    data = L.transforms_dict["R"](data)
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
    add_plan_to_db(L.project,
        L.plan, db_path=L.project.db, data_folder_lbd=L.output_folder
    )

# %%

    add_plan_to_db(L.project,
            I.L.plan, db_path=I.L.project.db, data_folder_lbd=I.L.output_folder
        )
# %%
    add_plan_to_db(L.project,
        L.plan, db_path=L.project.db, data_folder_source=L.data_folder,data_folder_lbd=L.output_folder
    )
# %%
