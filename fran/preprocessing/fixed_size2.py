from pathlib import Path

import pandas as pd
import ray
import torch
from fran.preprocessing.helpers import (
    create_dataset_stats_artifacts,
    infer_dataset_stats_window,
)
from fran.preprocessing.preprocessor import Preprocessor, get_tensor_stats, store_label_count
from fran.preprocessing.rayworker_base import RayWorkerBase
from fran.transforms.imageio import LoadTorchd
from fran.transforms.misc_transforms import ChangeDtyped, GetLabelsd
from fran.utils.folder_names import folder_names_from_plan
from monai.transforms.spatial.dictionary import Resized
from monai.transforms.utility.dictionary import EnsureChannelFirstd

from utilz.fileio import maybe_makedirs


class _FixedSizeWorkerBase(RayWorkerBase):
    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder,
        device="cpu",
        debug=False,
    ):
        super().__init__(
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
            device=device,
            debug=debug,
            tfms_keys="LoadT,Chan,Remap,Resize,LmDType,Labels",
            remapping_key="remapping_whole",
        )

    def _create_data_dict(self, row: pd.Series):
        return {
            "image": row["image"],
            "lm": row["lm"],
            "ds": row["ds"],
            "remapping": row["remapping"],
        }

    def create_transforms(self, device="cpu"):
        self.image_key = "image"
        self.lm_key = "lm"
        self.LoadT = LoadTorchd(keys=[self.image_key, self.lm_key])
        self.Chan = EnsureChannelFirstd(
            keys=[self.image_key, self.lm_key], channel_dim="no_channel"
        )
        self.Remap = self.create_monai_remapping_per_ds("remapping_whole", self.lm_key)
        self.Resize = Resized(
            keys=[self.image_key, self.lm_key],
            spatial_size=self.plan["patch_size"],
            mode=["linear", "nearest"],
        )
        self.LmDType = ChangeDtyped(keys=[self.lm_key], target_dtype=torch.uint8)
        self.Labels = GetLabelsd(lm_key=self.lm_key)
        self.transforms_dict = {
            "LoadT": self.LoadT,
            "Chan": self.Chan,
            "Remap": self.Remap,
            "Resize": self.Resize,
            "LmDType": self.LmDType,
            "Labels": self.Labels,
        }
        self.transforms = self.tfms_from_dict(self.tfms_keys)

    def _process_row(self, row: pd.Series):
        data = self._create_data_dict(row)
        data = self.apply_transforms(data)
        image = data["image"]
        lm = data["lm"]
        labels = data["lm_labels"]
        assert image.shape == lm.shape, "mismatch in shape"
        assert image.dim() == 4, "images should be cxhxwxd"
        self.save_pt(image[0], "images")
        self.save_pt(lm[0], "lms")
        stats = get_tensor_stats(image[0])
        stats["case_id"] = row["case_id"]
        stats["ok"] = True
        stats["labels"] = labels
        return stats


@ray.remote(num_cpus=1)
class FixedSizeWorkerImpl(_FixedSizeWorkerBase):
    pass


class FixedSizeWorkerLocal(_FixedSizeWorkerBase):
    pass


class FixedSizeDataGenerator(Preprocessor):
    _default = "project"

    def __init__(self, project, plan, data_folder=None, output_folder=None):
        existing_fldr = folder_names_from_plan(project, plan)["data_folder_whole"]
        existing_fldr = Path(existing_fldr)
        if existing_fldr.exists():
            print(
                "Plan folder already exists:  {}.\nWill use existing folder to add data".format(
                    existing_fldr
                )
            )
            output_folder = existing_fldr
        self.remapping_key = "remapping_whole"
        super().__init__(
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
        )
        self.actor_cls = FixedSizeWorkerImpl
        self.local_worker_cls = FixedSizeWorkerLocal

    def set_input_output_folders(self, data_folder, output_folder):
        folders = folder_names_from_plan(self.project, self.plan)
        if data_folder is None:
            data_folder = folders["data_folder_source"]
        self.data_folder = Path(data_folder)
        if output_folder is None:
            output_folder = folders["data_folder_whole"]
        self.output_folder = Path(output_folder)

    def create_output_folders(self):
        maybe_makedirs(
            [
                self.output_folder / ("lms"),
                self.output_folder / ("images"),
            ])

    def create_data_df(self):
        Preprocessor.create_data_df(self)
        remapping = self.plan.get(self.remapping_key)
        self.df = self.df.assign(remapping=[remapping] * len(self.df))

    def setup(self, overwrite=False, num_processes=8, device="cpu", debug=False):
        self.create_output_folders()
        self.setup_workers(
            overwrite=overwrite,
            num_processes=num_processes,
            device=device,
            debug=debug,
        )

    def postprocess_results(self, **process_kwargs):
        if len(self.results_df) == 0:
            print("No results generated")
            return
        if self.results_df["ok"].all():
            self._store_dataset_properties()
        else:
            print("Some files failed; dataset properties not stored from process results")
        store_label_count(
            self.output_folder, num_processes=getattr(self, "num_processes", 1)
        )
        create_dataset_stats_artifacts(
            output_folder=self.output_folder,
            gif=self.store_gifs,
            label_stats=self.store_label_stats,
            gif_window=infer_dataset_stats_window(self.project),
        )


# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>
if __name__ == "__main__":
    import torch
    from utilz.helpers import set_autoreload

    set_autoreload()

    from fran.configs.parser import ConfigMaker
    from fran.managers import Project
    from fran.utils.common import *  # noqa: F403

    P = Project(project_title="totalseg")
    C = ConfigMaker(P)
    C.setup(2)
    conf = C.configs
    # P.maybe_store_projectwide_properties()

# %%
    plan = conf["plan_train"]
# %%

    F = FixedSizeDataGenerator(
        project=P,
        plan=plan,


    )
# %%
    debug_ = False
    overwrite_ = False
    F.setup(overwrite=overwrite_, debug=debug_)

# %%
    F.process()
    F.run_postprocess_only()

    # pd.set_option("display.width", 200)  # total line width before
#
