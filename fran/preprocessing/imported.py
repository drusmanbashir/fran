# %%
import ipdb
import numpy as np
import ray
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from utilz.imageviewers import ImageMaskViewer

from fran.preprocessing.preprocessor import generate_bboxes_from_lms_folder
from fran.preprocessing.rayworker_base import RayWorkerBase
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import BBoxFromPTd
from fran.transforms.misc_transforms import (ApplyBBox, LabelRemapSITKd,
                                             MergeLabelmapsd, Recastd)
from fran.transforms.spatialtransforms import ResizeToTensord

tr = ipdb.set_trace

from pathlib import Path
from typing import Any, Dict, Optional, Union

from fastcore.all import store_attr
from label_analysis.totalseg import TotalSegmenterLabels
from utilz.helpers import find_matching_fn, pbar
from utilz.string import info_from_filename

from fran.preprocessing.labelbounded import (LabelBoundedDataGenerator,
                                             LBDSamplerWorkerImpl)
from fran.utils.config_parsers import ConfigMaker

@ray.remote(num_cpus=4)
class LBDImportedSamplerWorkerImpl(RayWorkerBase):
    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder,
        crop_to_label=None,
        device="cpu",
    ):
        imported_folder = plan["imported_folder"]
        self.imported_folder = Path(imported_folder)
        merge_imported_labels = plan["merge_imported_labels"]
        if merge_imported_labels == True:
            tfms_keys = "R,LS,LT,D,E,Rz,M,B,A,Ind"
        else:
            tfms_keys = "R,LS,LT,D,E,Rz,B,A,Ind"
        self.lm_imported_key = "lm_imported"
        super().__init__(
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
            crop_to_label=crop_to_label,
            device=device,
            tfms_keys=tfms_keys,
        )

    def _create_data_dict(self, row):


        data = {
                "image": row["image"],
                "lm": row["lm"],
            "lm_imported": row["lm_imported"],
                "remapping": row["remapping"],
            "remapping_imported": row["remapping_imported"],
            }
        return data

    def create_transforms(self, device):
        super().create_transforms(device=device)
        self.A = ApplyBBox(keys=[self.lm_key, self.image_key], bbox_key="bounding_box")
        self.B = BBoxFromPTd(
            keys=[self.lm_imported_key],
            spacing=self.plan["spacing"],
            expand_by=self.plan["expand_by"],
        )

        self.E = EnsureChannelFirstd(
            keys=[self.lm_imported_key, self.image_key, self.lm_key],
            channel_dim="no_channel",
        )
        self.LS = LoadSITKd(keys=[self.lm_imported_key], image_only=True)
        self.M = MergeLabelmapsd(
            keys=[self.lm_imported_key, self.lm_key],
            meta_key=self.lm_key,
            key_output=self.lm_key,
        )

        self.R = LabelRemapSITKd(
            keys=[self.lm_imported_key], remapping_key="remapping_imported"
        )  # This loads and remaps sitk image. Meta filename is lost!

        self.Re = Recastd(keys=[self.lm_imported_key])
        self.Rz = ResizeToTensord(
            keys=[self.lm_imported_key], key_template_tensor=self.lm_key, mode="nearest"
        )
        self.transforms_dict.update(
            {
                "A": self.A,
                "E": self.E,
                "B": self.B,
                "LS": self.LS,
                "M": self.M,
                "R": self.R,
                "Re": self.Re,
                "Rz": self.Rz,
            }
        )

class LabelBoundedDataGeneratorImported(LabelBoundedDataGenerator):
    """
    Label-bounded data generator that works with imported label files.

    This class extends LabelBoundedDataGenerator to work with external/imported label files
    (e.g., from TotalSegmenter) alongside the original dataset labels. It processes
    fixed_spacing_folder data and uses imported labels tocrop images accordingly.

    The class can either merge the imported labels with existing ones so they are used in training as well or it can simply use them for cropping images.

    Args:
        project: Project instance containing dataset configuration
        plan: Processing plan configuration dictionary
        imported_folder (str/Path): Path to folder containing imported label files
            existing labels. Defaults to False.
        folder_suffix (str, optional): Suffix to append to output folder name
        data_folder (str/Path, optional): Input data folder path
        output_folder (str/Path, optional): Output folder path for processed data
        mask_label (optional): Specific label to use for masking
        remapping (dict, optional): Label remapping dictionary for imported labels

    Attributes:
        imported_folder (Path): Path to imported label files
        df (DataFrame): Extended dataframe with 'imported' column for file matching

    Example:
        >>> from fran.managers import Project
        >>> P = Project(project_title="my_project")
        >>> generator = LabelBoundedDataGeneratorImported(
        ...     project=P,
        ...     plan=plan_config,
        ...     imported_folder="/path/to/totalseg/predictions",
        ...     
        ...     remapping={1: 1, 2: 2}  # liver labels
        ... )
        >>> generator.setup(overwrite=True)
        >>> generator.process()

    Note:
        - Imported files are matched to cases using case_id extracted from filenames
        - A single image/label pair is generated per case after boundary-based cropping
        - Supports various imported label formats (TotalSegmenter, custom annotations, etc.)
    """

    _default = "project"

    def __init__(
        self,
        project,
        plan: Dict[str, Any],
        data_folder: Optional[Union[str, Path]] = None,
        output_folder: Optional[Union[str, Path]] = None,
        mask_label: Optional[Any] = None,
        device="cpu",
    ) -> None:

        super().__init__(
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
            crop_to_label=mask_label,
        )
        imported_folder = plan["imported_folder"]
        self.imported_folder = Path(imported_folder)
        self.actor_cls = LBDImportedSamplerWorkerImpl
    def create_data_df(self) -> None:
        """
        Create data DataFrame with imported file matching.
        Extends the parent's create_data_df method by adding an 'imported' column
        that maps case_ids to their corresponding imported label files.

        Raises:
            FileNotFoundError: If imported_folder doesn't exist
            ValueError: If no imported files are found or no case_ids can be matched
        """
        super().create_data_df()

        # Validate imported folder exists
        if not self.imported_folder.exists():
            raise FileNotFoundError(
                f"Imported folder not found: {self.imported_folder}"
            )
        # Get imported files
        imported_fns = list(self.imported_folder.glob("*"))
        if not imported_fns:
            raise ValueError(
                f"No files found in imported folder: {self.imported_folder}"
            )

        unmatched_images = []
        matched_files = []
        for fn in pbar(self.df.image):
            try:
                matching = find_matching_fn(fn, imported_fns, tags=["all"])
                matched_files.append(matching)
            except Exception as e:
                print(f"Warning: No match found for {fn.name}: {e}")
                unmatched_images.append((self.df.image.name, "unknown"))

        self.df["lm_imported"] = matched_files
        self.df["remapping_imported"] = [self.plan["remapping_imported"]] * len(self.df)

        nan_mask = self.df["lm_imported"].isna()
        if nan_mask.any():
            missing_count = nan_mask.sum()
            print(f"Warning: {missing_count}/{len(self.df)} cases lack imported files:")
            missing_image_fns = self.df.loc[nan_mask, "image"].tolist()
            print(f"  Missing image_fns: {missing_image_fns}")

        else:
            print("✓ All case_ids successfully matched with imported files!")

    def create_properties_dict(self) -> Dict[str, Any]:
        """
        Create properties dictionary with imported label information.

        Extends the parent's create_properties_dict method by adding information
        about imported labels, their folder location, and merge settings.

        Returns:
            dict: Combined properties dictionary containing both parent properties
                and imported label-specific information including:
                - imported_folder: Path to imported label files
                - imported_labels: Dictionary of imported label mappings
                - merge_imported_labels: Boolean flag for label merging
        """
        resampled_dataset_properties = super().create_properties_dict()
        if self.plan["remapping_imported"] is None:
            tr()
            labels = None
        else:
            labels = {
                k[0]: k[1] for k in self.remapping.items() if self.remapping[k[0]] != 0
            }
        additional_props = {
            "imported_folder": str(self.imported_folder),
            "imported_labels": labels,
            "merge_imported_labels": self.merge_imported_labels,
        }
        return resampled_dataset_properties | additional_props

    #
    # def setup(self, num_processes=16, device="cpu", overwrite=True):
    #
    #     self.create_data_df()
    #     self.register_existing_files()
    #     print("Overwrite:", overwrite)
    #     if overwrite == False:
    #         self.remove_completed_cases()
    #     if len(self.df) > 0:
    #         self.mini_dfs = np.array_split(self.df, num_processes)
    #
    #         self.n_actors = min(len(self.df), int(num_processes))
    #         # (Optionally) initialise Ray if not already
    #         if not ray.is_initialized():
    #             try:
    #                 ray.init(ignore_reinit_error=True)
    #             except Exception as e:
    #                 print("Ray init warning:", e)
    #
    #         actor_kwargs = dict(
    #             project=self.project,
    #             plan=self.plan,
    #             data_folder=self.data_folder,
    #             output_folder=self.output_folder,
    #             device=device,
    #             crop_to_label=None,
    #         )
    #         self.actors = [
    #             self.actor_cls.remote(**actor_kwargs)
    #             for _ in range(self.n_actors)
    #         ]

if __name__ == "__main__":
# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>

    from fran.managers import Project
    from fran.utils.common import *

    project_title = "lidc"
    P = Project(project_title=project_title)
    # P.maybe_store_projectwide_properties()
    # spacing = [1.5, 1.5, 1.5]

    C = ConfigMaker(P, raytune=False, configuration_filename=None)
    C.setup(3)
# %%
    C.plans
    conf = C.configs
    print(conf["model_params"])

    plan = conf["plan_train"]
    pp(plan)
    spacing = plan["spacing"]
    # plan["remapping_imported"][0]
# %%
# SECTION:-------------------- Imported labels-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>
    L = LabelBoundedDataGeneratorImported(
        project=P,
        plan=plan,
        data_folder="/r/datasets/preprocessed/lidc/fixed_spacing/spc_080_080_150_ldc",
    )

# %%
    overwrite=True
    L.setup(overwrite=overwrite)

    L.process()
# %%
# %%
    num_processes=16
    L.create_data_df()
    L.register_existing_files()
    L.mini_dfs = np.array_split(L.df, num_processes)
# %%

    mini_df = L.mini_dfs[0]
    # mini_df = mini_df.iloc[:3]
# %%
    LL = LBDImportedSamplerWorkerImpl(project=L.project, plan=L.plan, data_folder=L.data_folder, output_folder=L.output_folder)
    outs = LL.process(mini_df )
# %%
    row=mini_df.iloc[0]
    dici = {"A": 12}
    dici.update({"A": 13, "B":15})

    data = {
            "image": row["image"],
            "lm": row["lm"],
        "lm_imported": row["lm_imported"],
            "remapping": row["remapping"],
        "remapping_imported": row["remapping_imported"],
        }

# %%
        # Apply transforms
    # data = LL.transforms(data)
    data2 = LL.transforms_dict["R"](data)
    data2  = LL.transforms_dict["LS"](data2)
    data2  = LL.transforms_dict["LT"](data2)
    data2  = LL.transforms_dict["D"](data2)
    data2  = LL.transforms_dict["E"](data2)
    data3 = LL.transforms_dict["Rz"](data2)
    data2['lm_imported'].shape
    data2['lm'].shape
# %%
# %%
