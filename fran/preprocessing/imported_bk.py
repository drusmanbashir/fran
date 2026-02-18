# %%
import ipdb
from utilz.imageviewers import ImageMaskViewer

from fran.preprocessing.preprocessor import generate_bboxes_from_lms_folder
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
from utilz.stringz import info_from_filename

from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.configs.parser import ConfigMaker


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
        merge_imported_labels (bool, optional): Whether to merge imported labels with
            existing labels. Defaults to False.
        folder_suffix (str, optional): Suffix to append to output folder name
        data_folder (str/Path, optional): Input data folder path
        output_folder (str/Path, optional): Output folder path for processed data
        mask_label (optional): Specific label to use for masking
        remapping (dict, optional): Label remapping dictionary for imported labels

    Attributes:
        imported_folder (Path): Path to imported label files
        merge_imported_labels (bool): Flag for label merging behavior
        df (DataFrame): Extended dataframe with 'imported' column for file matching

    Example:
        >>> from fran.managers import Project
        >>> P = Project(project_title="my_project")
        >>> generator = LabelBoundedDataGeneratorImported(
        ...     project=P,
        ...     plan=plan_config,
        ...     imported_folder="/path/to/totalseg/predictions",
        ...     merge_imported_labels=True,
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
            mask_label=mask_label,
        )
        imported_folder = plan["imported_folder"]
        self.imported_folder = Path(imported_folder)
        self.merge_imported_labels = plan["merge_imported_labels"]
        if self.merge_imported_labels == True:
            self.tfms_keys = "R,LS,LT,D,E,Rz,M,B,A,Ind"
        else:
            self.tfms_keys = "R,LS,LT,D,E,Rz,B,A,Ind"
        self.lm_imported_key = "lm_imported"
        self.tnsr_keys = self.image_key, self.lm_key ,self.lm_imported_key

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
                matching = find_matching_fn(fn, imported_fns, tags=["all"])[0]
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

    def create_transforms(self, device):
        super().create_transforms(device=device)
        self.A = ApplyBBox(keys=[self.lm_key, self.image_key], bbox_key="bounding_box")
        self.B = BBoxFromPTd(
            keys=[self.lm_imported_key],
            spacing=self.plan["spacing"],
            expand_by=self.plan["expand_by"],
        )

        self.LS = LoadSITKd(keys=[self.lm_imported_key], image_only=True)
        self.M = MergeLabelmapsd(
            keys=[self.lm_imported_key, self.lm_key], meta_key=self.lm_key, key_output=self.lm_key
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
                "B": self.B,
                "LS": self.LS,
                "M": self.M,
                "R": self.R,
                "Re": self.Re,
                "Rz": self.Rz,
            }
        )

    def process_files(self):
        """Process files without using DataLoader"""
        self.create_output_folders()
        self.results = []
        self.shapes = []

        # for img_file, lm_file in pbar(zip(self.image_files, self.lm_files), desc="Processing files", total=len(self.image_files)):
        for index, row in pbar(self.df.iterrows(), total=len(self.df)):
            data = row.to_dict()
            try:
                # Load and process single case
                # data = {
                #     "image": row['image'],
                #     "lm": row['lm'],
                #     "lm_imported": row['lm_imported'],
                # }

                # Apply transforms
                data = self.transforms(data)

                # Get metadata and indices
                # Process the case
                self.process_single_case(
                    data["image"],
                    data["lm"],
                    data["lm_fg_indices"],
                    data["lm_bg_indices"],
                )

            except Exception as e:
                print(f"Error processing {row['image'].name}: {str(e)}")
                continue

        self.results_df = pd.DataFrame(self.results)
        # self.results= pd.DataFrame(self.results).values
        ts = self.results_df.shape
        if ts[-1] == 4:  # only store if entire dset is processed
            self._store_dataset_properties()
            generate_bboxes_from_lms_folder(self.output_folder / ("lms"))
        else:
            print(
                "self.results  shape is {0}. Last element should be 4 , is {1}. therefore".format(
                    ts, ts[-1]
                )
            )
            print(
                "since some files skipped, dataset stats are not being stored. run self.get_tensor_folder_stats and generate_bboxes_from_lms_folder separately"
            )

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
        if self.plan["remapping_imported"] is None :
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



if __name__ == "__main__":
# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR>

    from fran.managers import Project
    from fran.utils.common import *

    project_title = "lidc"
    P = Project(project_title=project_title)
    # P.maybe_store_projectwide_properties()
    # spacing = [1.5, 1.5, 1.5]


    C = ConfigMaker(P,  configuration_filename=None)
    C.setup(3)
# %%
    C.plans
    conf = C.configs
    print(conf["model_params"])

    plan = conf['plan_train']
    pp(plan)
    spacing = plan["spacing"]
    # plan["remapping_imported"][0]
# %%
# SECTION:-------------------- Imported labels-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>
    L = LabelBoundedDataGeneratorImported(
        project=P,
        plan=plan,
        data_folder = "/r/datasets/preprocessed/lidc/fixed_spacing/spc_080_080_150_ldc"


    )

# %%
    overwrite = True
    L.setup(overwrite=overwrite)
    L.process()

# %%
    if not "labels_all" in P.global_properties.keys():
        P.set_lm_groups(plan["lm_groups"])
        P.maybe_store_projectwide_properties(overwrite=False)
    lm_group = P.global_properties["lm_group1"]
    TSL = TotalSegmenterLabels()
    imported_folder = "/s/fran_storage/predictions/totalseg/LITS-1088"
    imported_labelsets = lm_group["imported_labelsets"]
    imported_labelsets = [TSL.get_labels("liver", localiser=True)]
    remapping = TSL.create_remapping(
        imported_labelsets,
        [
            1,
        ]
        * len(imported_labelsets),
        localiser=True,
    )
    merge_imported_labels = True
# %%

    L = LabelBoundedDataGeneratorImported(
        project=P,
        data_folder="",
        output_folder="/s/xnat_shadow/crc/sampling/tensors/lbd",
        plan=plan,
        imported_folder=imported_folder,
        merge_imported_labels=merge_imported_labels,
        remapping_imported=remapping,
        folder_suffix=plan["plan_name"],
    )
# %%
    L.setup(overwrite=True)
    L.process()
# %%
# SECTION:-------------------- TROUBLESHOOTING-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
    L.create_output_folders()
    L.results = []
    L.shapes = []

    L.results = []
    # Load and process single case
# %%

    row = L.df.iloc[0]
    data = row.to_dict()
    try:
        # Load and process single case
        # data = {
        #     "image": row['image'],
        #     "lm": row['lm'],
        #     "lm_imported": row['lm_imported'],
        # }

        # Apply transforms
        data = L.transforms(data)

        # Get metadata and indices
        # Process the case
        L.process_single_case(
            data["image"],
            data["lm"],
            data["lm_fg_indices"],
            data["lm_bg_indices"],
        )

    except Exception as e:
        print(f"Error processing {row['image'].name}: {str(e)}")

# %%
    row = L.df.iloc[0]
    data = row.to_dict()
# %%
    if L.merge_imported_labels == True:
        L.set_transforms("R,LS,LT,D,E,Rz,M,B,A,Ind")
    else:
        L.set_transforms("R,LS,LT,D,E,Rz,B,A,Ind")

# %%

    # Apply transforms
    data2 = L.transforms(data)

    # Get metadata and indices
# %%
    img = data2["image"][0]
    lm = data2["lm"][0]
    img.shape
    ImageMaskViewer([img, lm])
# %%
    # Process the case
    L.process_single_case(
        data2["image"],
        data2["lm"],
        data2["lm_fg_indices"],
        data2["lm_bg_indices"],
    )
# %%
    row = L.df.iloc[10]
    img_fn = row["image"]
    find_matching_fn(img_fn, Path(L.imported_folder)[0])

# %%
    dici = L.transforms_dict["R"](data)
    dici = L.transforms_dict["LS"](dici)
    dici = L.transforms_dict["LT"](dici)
    dici = L.transforms_dict["D"](dici)
    dici = L.transforms_dict["E"](dici)
    dici = L.transforms_dict["Rz"](dici)
# %%
    dici = L.transforms_dict["M"](dici)
    dici = L.transforms_dict["B"](dici)
    dici = L.transforms_dict["A"](dici)
    dici = L.transforms_dict["Ind"](dici)
# %%
    dici["lm"].meta

# %%
# %%
    """Create data dictionaries from DataFrame."""
    data2 = []
    for index in range(len(L.ds.df)):
        row = L.ds.df.iloc[index]
        ds = row.get("ds")
        remapping = L.ds._get_ds_remapping(ds)
        dici = L.ds._dici_from_df_row(row, remapping)
        data2.append(dici)
# %%
    L = I.L
    imported_folder = L.ds.imported_folder
    masks_folder = L.ds.data_folder / "lms"
    images_folder = L.ds.data_folder / "images"
    lm_fns = list(masks_folder.glob("*.pt"))
    img_fns = list(images_folder.glob("*.pt"))
    imported_files = list(imported_folder.glob("*"))
    data2 = []
    for cid in L.ds.case_ids:
        lm_fn = L.ds.case_id_file_match(cid, lm_fns)
        img_fn = L.ds.case_id_file_match(cid, img_fns)
        imported_fn = L.ds.case_id_file_match(cid, imported_files)
        dici = {
            "lm": lm_fn,
            "image": img_fn,
            "lm_imported": imported_fn,
            "remapping": L.ds.remapping,
        }
        data2.append(dici)
# %%
    imported_folder = plan["imported_folder"]
    imported_folder = Path(imported_folder)

    merge_imported_labels = plan["merge_imported_labels"]
    TSL = TotalSegmenterLabels()
# %%
    I.L = LabelBoundedDataGeneratorImported(
        project=I.project,
        plan=I.plan,
        folder_suffix=I.plan_name,
        # expand_by=I.plan["expand_by"],
        # spacing=I.plan["spacing"],
        imported_folder=imported_folder,
        merge_imported_labels=merge_imported_labels,
        remapping_imported=remapping,
    )

# %%

    imported_fns = list(L.imported_folder.glob("*"))
    if not imported_fns:
        raise ValueError(f"No files found in imported folder: {L.imported_folder}")

    print(f"Found {len(imported_fns)} imported files to match")

    # Create case_id to index mapping for O(1) lookups
    case_id_to_idx = {case_id: idx for idx, case_id in enumerate(L.df["case_id"])}

    # Initialize imported column efficiently
    L.df["imported"] = None

    # Process imported files with optimized matching
    matched_files = {}
    unmatched_images = []

    for fn in imported_fns:
        try:
            case_id = info_from_filename(fn.name, full_caseid=True)["case_id"]
            if case_id in case_id_to_idx:
                matched_files[case_id] = fn
            else:
                unmatched_images.append((fn.name, case_id))
        except Exception as e:
            print(f"Warning: Could not extract case_id from {fn.name}: {e}")
            unmatched_images.append((fn.name, "unknown"))

        # L.df.to_csv("/s/xnat_shadow/crc/tensors/ldb/info.csv")

# %%

        L = I.L

        if not L.imported_folder.exists():
            raise FileNotFoundError(f"Imported folder not found: {L.imported_folder}")

        # Get imported files
        imported_fns = list(L.imported_folder.glob("*"))

# %%
        if not imported_fns:
            raise ValueError(f"No files found in imported folder: {L.imported_folder}")

        print(f"Found {len(imported_fns)} imported files to match")

        # Create case_id to index mapping for O(1) lookups
        case_id_to_idx = {case_id: idx for idx, case_id in enumerate(L.df["case_id"])}

        # Initialize imported column efficiently
        L.df["imported"] = None

        # Process imported files with optimized matching
# %%
        unmatched_images = []
        matched_files = []
        for fn in pbar(L.df.image):
            try:
                # case_id = info_from_filename(fn.name, full_caseid=True)['case_id']
                matching = find_matching_fn(fn, imported_fns, tags=["all"])[0]
                matched_files.append(matching)

                # print(case_id)
                if case_id in case_id_to_idx:
                    matched_files[case_id] = fn
                else:
                    unmatched_images.append((fn.name, case_id))
            except Exception as e:
                print(f"Warning: No match found for {fn.name}: {e}")
                unmatched_images.append((L.df.image.name, "unknown"))

        L.df.imported = matched_files
        matched_count = len(matched_files)
        # Check for cases without imported files using vectorized operation
        nan_mask = L.df["imported"].isna()
        if nan_mask.any():
            missing_count = nan_mask.sum()
            print(f"Warning: {missing_count}/{len(L.df)} cases lack imported files:")
            missing_image_fns = L.df.loc[nan_mask, "image"].tolist()
            print(f"  Missing image_fns: {missing_image_fns}")

        else:
            print("✓ All case_ids successfully matched with imported files!")


# %%
