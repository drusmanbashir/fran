# %%

MIN_SIZE = 32  # min size in a single dimension of any image

import ipdb
import ray
from fran.data.dataregistry import DS, DatasetRegistry
from fran.preprocessing.rayworker_base import RayWorkerBase
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import BBoxFromPTd
from fran.transforms.misc_transforms import (
    ApplyBBoxd,
    DummyTransform,
    LabelRemapSITKd,
    MergeLabelmapsd,
    RecastToFloatd,
)
from fran.transforms.spatialtransforms import ResizeToTensord
from monai.transforms.utility.dictionary import EnsureChannelFirstd

tr = ipdb.set_trace

from pathlib import Path
from typing import Any, Dict, Optional, Union

from fran.configs.parser import parse_excel_datasources, parse_excel_remapping
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from tqdm.auto import tqdm as pbar
from utilz.helpers import find_matching_fn


def resolve_relative_path(pth: str) -> str:
    from fran.utils.common import COMMON_PATHS

    DS = DatasetRegistry()
    pth2 = pth.split("/")
    str_out = ""
    for sub_path in pth2:
        if "$" in sub_path:
            sub_path = sub_path.replace("$", "")
            rel_path = COMMON_PATHS[sub_path]
            str_out += f"{rel_path}/"
        elif "DS." in sub_path:
            ds = sub_path.split(".")[1]
            ds = DS[ds]
            fldr_path = ds.folder
            fldr_path = str(fldr_path)
            str_out += f"{fldr_path}/"

        else:
            str_out += f"{sub_path}/"
    return str_out


class _LBDImportedSamplerWorkerBase(RayWorkerBase):
    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder,
        crop_to_label=None,
        device="cpu",
        debug=False,
    ):
        """
        if both remmapping and remapping_imported are present, remapping imported is applied to the imported lm first,
        then it is cropped, then remapping is kept or discarded, and finally remapping is applied to the lm
        """

        imported_folder = plan["imported_folder"]
        self.imported_folder = Path(imported_folder)
        merge_imported_labels = plan["merge_imported_labels"]
        if merge_imported_labels:
            tfms_keys = (
                "RemapI,LoadS,LoadT,Dev,Chan,Cast,Rsz,Merg,BBox,AppBx,Remap,Labels,Indx"
            )
        else:
            tfms_keys = (
                "RemapI,LoadS,LoadT,Dev,Chan,Cast,Rsz,BBox,AppBx,Remap,Labels,Indx"
            )
        self.lm_imported_key = "lm_imported"
        super().__init__(
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
            crop_to_label=crop_to_label,
            device=device,
            debug=debug,
            tfms_keys=tfms_keys,
            remapping_key="remapping_lbd",
        )

    def _create_data_dict(self, row):
        data = {
            "image": row["image"],
            "lm": row["lm"],
            "ds": row["ds"],
            "lm_imported": row["lm_imported"],
            "remapping": row["remapping"],
            "remapping_imported": row["remapping_imported"],
        }
        return data

    def create_sitk_remapping_per_ds(self, remapping_key, lm_key) -> dict:
        dss = self.plan["datasources"]
        dss = parse_excel_datasources(dss)
        rem = self.plan[remapping_key]
        rem = parse_excel_remapping(rem)
        Remapping_tfms = {}
        for rez, ds in zip(rem, dss):
            ds_name = DS[ds].name
            if rez is not None:
                R = LabelRemapSITKd(keys=[lm_key], remapping=rez)
            else:
                R = DummyTransform(keys=[lm_key])
            Remapping_tfms[ds_name] = R
        return Remapping_tfms

    def create_transforms(self, device):
        super().create_transforms(device=device)
        self.A = ApplyBBoxd(keys=[self.lm_key, self.image_key], bbox_key="bounding_box")
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

        self.RemapI = self.create_sitk_remapping_per_ds(
            "remapping_imported", self.lm_imported_key
        )

        self.Re = RecastToFloatd(keys=[self.lm_imported_key])
        self.Rz = ResizeToTensord(
            keys=[self.lm_imported_key], key_template_tensor=self.lm_key, mode="nearest"
        )
        self.transforms_dict.update(
            {
                "AppBx": self.A,
                "Chan": self.E,
                "BBox": self.B,
                "LoadS": self.LS,
                "Merg": self.M,
                "RemapI": self.RemapI,
                "Cast": self.Re,
                "Rsz": self.Rz,
            }
        )

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


@ray.remote(num_cpus=4)
class LBDImportedSamplerWorkerImpl(_LBDImportedSamplerWorkerBase):
    pass


class LBDImportedSamplerWorkerLocal(_LBDImportedSamplerWorkerBase):
    pass


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


    _default = "project"
    """

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
        datasources_ = plan.get("datasources")
        datasources = datasources_.replace(" ", "").split(",")
        imported_folder_ = plan.get("imported_folder", "")
        imported_folder__ = imported_folder_.replace(" ", "").split(",")
        dicis = []

        remappings = plan.get("remapping_imported") or [None] * len(datasources)
        for ds, imported_folder, remapping in zip(
            datasources, imported_folder__, remappings
        ):
            dici = {}
            dici["ds"] = DS[ds].name
            imported_folder = resolve_relative_path(imported_folder)
            dici["imported_folder"] = imported_folder
            dici["remapping_imported"] = remapping
            dicis.append(dici)
        self.ds_imported_folder_remapping = dicis
        self.actor_cls = LBDImportedSamplerWorkerImpl
        self.local_worker_cls = LBDImportedSamplerWorkerLocal

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

    def set_remapping_per_ds(self):
        df = self.df.copy()
        for dici in self.ds_imported_folder_remapping:
            ds = dici["ds"]
            imported_folder = dici["imported_folder"]
            imported_folder = Path(imported_folder)
            remapping_imported = dici["remapping_imported"]
            mask = df["ds"] == ds
            idx = df.index[mask]
            # df = df.loc[idx]

            if len(str(imported_folder)) < 3:
                pass
            else:
                if not imported_folder.exists():
                    raise FileNotFoundError(
                        f"Imported folder not found: {imported_folder}"
                    )

                df.loc[idx, "imported_folder"] = imported_folder
                df.loc[idx, "remapping_imported"] = [remapping_imported] * len(idx)

                # df.loc[idx]["imported_folder"]
                # Validate imported folder exists
                # Get imported files
                imported_fns = list(imported_folder.glob("*"))
                unmatched_images = []
                matched_files = []
                for fn in pbar(df.loc[idx, "image"]):
                    try:
                        fn = Path(fn)
                        matching = find_matching_fn(
                            fn.name, imported_fns, tags=["all"]
                        )[0]
                        matched_files.append(matching)
                    except Exception as e:
                        print(f"Warning: No match found for {fn.name}: {e}")
                        unmatched_images.append(fn)
                if len(matched_files) != len(idx):
                    raise ValueError(
                        f"Failed to match all case_ids with imported files: {unmatched_images}"
                    )

                df.loc[idx, "lm_imported"] = matched_files
                nan_mask = df.loc[idx, "lm_imported"].isna()
                if nan_mask.any():
                    missing_count = nan_mask.sum()
                    print(
                        f"Warning: {missing_count}/{len(df)} cases lack imported files:"
                    )
                    missing_image_fns = df.loc[idx, "image"].loc[nan_mask].tolist()
                    print(f"  Missing image_fns: {missing_image_fns}")

                else:
                    print(
                        f"✓ All case_ids successfully matched with imported files for datasource {ds}!"
                    )
        self.df = df.copy()

    def create_properties_dict(self) -> Dict[str, Any]:
        props = super().create_properties_dict()
        imported_info = {
            "merge_imported_labels": bool(
                self.plan.get("merge_imported_labels", False)
            ),
            "remapping_imported": [
                d.get("remapping_imported") for d in self.ds_imported_folder_remapping
            ],
        }
        return props | imported_info


if __name__ == "__main__":
    from pprint import pp

    import numpy as np
    from utilz.helpers import set_autoreload

    set_autoreload()

    from fran.configs.parser import ConfigMaker

# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR>
    from fran.managers import Project
    from fran.preprocessing.preprocessor import store_label_count
    from fran.utils.common import *
    from fran.utils.folder_names import folder_names_from_plan

    project_title = "kits2"
    P = Project(project_title=project_title)
    # P.maybe_store_projectwide_properties()
    # spacing = [1.5, 1.5, 1.5]

    C = ConfigMaker(P)
    C.setup(9)

# %%

    conf = C.configs
    print(conf["model_params"])

    plan = conf["plan_train"]
    pp(plan)
    spacing = plan["spacing"]
    imported_fldr = plan["imported_folder"]

# %%
# SECTION:-------------------- Imported labels-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>
    resampled_data_folder = folder_names_from_plan(P, plan)["data_folder_source"]

# %%
    L = LabelBoundedDataGeneratorImported(
        project=P, plan=plan, data_folder=resampled_data_folder
    )

# %%
    num_processes = 6
    debug = False
    debug = True
    overwrite = False
    L.setup(overwrite=overwrite, num_processes=num_processes, debug=debug)
    L.process()
    store_label_count(L.output_folder, 16)
# %%
    num_processes = 16
    L.create_data_df()
    L.register_existing_files()
    L.mini_dfs = np.array_split(L.df, num_processes)
# %%

    mini_df = L.mini_dfs[0]
    # mini_df = mini_df.iloc[:3]
# %%
    LL = LBDImportedSamplerWorkerImpl.remote(
        project=L.project,
        plan=L.plan,
        data_folder=L.data_folder,
        output_folder=L.output_folder,
    )
    outs = LL.process(mini_df)
# %%
    row = mini_df.iloc[0]
    dici = {"AppBx": 12}
    dici.update({"AppBx": 13, "BBox": 15})

# %%
    data = {
        "image": row["image"],
        "lm": row["lm"],
        "lm_imported": row["lm_imported"],
        "remapping": row["remapping"],
        "remapping_imported": row["remapping_imported"],
    }

# %%
    # Apply transforms
    data = LL.transforms(data)
    data2 = LL.transforms_dict["Remap"][data["ds"]](data)
    data2 = LL.transforms_dict["LoadS"](data2)
    data2 = LL.transforms_dict["LoadT"](data2)
    data2 = LL.transforms_dict["Dev"](data2)
    data2 = LL.transforms_dict["Chan"](data2)
    data3 = LL.transforms_dict["Rsz"](data2)
    data2["lm"].shape
    data2["lm_imported"].shape

# %%
