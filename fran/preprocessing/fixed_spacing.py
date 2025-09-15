from pathlib import Path
import ipdb
import ray
import itertools as il

from fran.inference.base import list_to_chunks
from fran.utils.config_parsers import create_remapping
from fran.utils.folder_names import folder_names_from_plan

tr = ipdb.set_trace

import numpy as np
import pandas as pd
import torch
from monai.transforms.compose import Compose
from monai.transforms.spatial.dictionary import Spacingd
from monai.transforms.utility.dictionary import (EnsureChannelFirstd,
                                                 FgBgToIndicesd, ToDeviceD,
                                                 ToDeviced)
from utilz.fileio import load_dict, maybe_makedirs, save_dict, save_json
from utilz.helpers import multiprocess_multiarg, pbar
from utilz.string import strip_extension

from fran.preprocessing import bboxes_function_version
from fran.preprocessing.preprocessor import Preprocessor
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import ToCPUd
from fran.transforms.intensitytransforms import NormaliseClipd
from fran.transforms.misc_transforms import (ChangeDtyped, DictToMeta,
                                             FgBgToIndicesd2, LabelRemapd,
                                             Recastd)
from fran.transforms.spatialtransforms import ResizeToTensord
from fran.utils.common import *


def generate_bboxes_from_lms_folder(
    masks_folder, bg_label=0, debug=False, num_processes=16
):
    label_files = masks_folder.glob("*pt")
    arguments = [
        [x, bg_label] for x in label_files
    ]  # 0.2 factor for thresholding as kidneys are small on low-res imaging and will be wiped out by default threshold 3000
    bboxes = multiprocess_multiarg(
        func=bboxes_function_version,
        arguments=arguments,
        num_processes=num_processes,
        debug=debug,
    )
    bbox_fn = masks_folder.parent / ("bboxes_info")
    print("Storing bbox info in {}".format(bbox_fn))
    save_dict(bboxes, bbox_fn)




@ray.remote(num_cpus=4)
class NiftiResampler(Preprocessor):
    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder,
        half_precision=False,
        clip_center=False,
        store_label_inds=False,
        mean_std_mode="dataset",
        device="cuda",
    ):

        super().__init__(
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
        )
        self.half_precision = half_precision
        self.store_label_inds = store_label_inds
        self.clip_center = clip_center
        self.device = device
        self.set_normalization_values(mean_std_mode)
        self.create_transforms()
    def set_input_output_folders(self, data_folder, output_folder):
        self.data_folder = data_folder
        self.output_folder = output_folder

    def _create_data_dicts_from_df(self, df):
        """Create data dictionaries from DataFrame."""
        remapping= self.plan["remapping_source"]
        data = []
        for index in range(len(df)):
            row = df.iloc[index]
            dici = self._dici_from_df_row(row, remapping)
            data.append(dici)
        return data

    def _get_ds_remapping(self, ds):
        if ds:
            try:
                remapping = get_ds_remapping(ds, self.global_properties)
                return remapping
            except:
                return None
        return None

    def _dici_from_df_row(self, row, remapping):
        img_fname = row["image"]
        mask_fname = row["lm"]
        dici = {
            "image": img_fname,
            "lm": mask_fname,
            "remapping_source": remapping,
        }
        return dici

    def _create_data_dicts_from_folder(self):
        """Create data dictionaries from data_folder structure."""
        data_folder = Path(self.data_folder)
        masks_folder = data_folder / "lms"
        images_folder = data_folder / "images"

        img_fns = list(images_folder.glob("*"))
        data = []

        for img_fn in img_fns:
            lm_fn = find_matching_fn(img_fn.name, masks_folder, "case_id")

            remapping = None

            dici = {
                "image": str(img_fn),
                "lm": str(lm_fn),
                "remapping_imported": remapping,
            }
            data.append(dici)
        assert len(data) > 0, "No data found in data folder"
        return data

    def create_transforms(self,device='cpu'):
        self.LS = LoadSITKd(keys=["image", "lm"], image_only=True)
        self.Rem = LabelRemapd(keys=["lm"], remapping_key="remapping_source")
        # self.RemI = LabelRemapd(keys=["lm"], remapping_key="remapping_imported")
        self.Re = Recastd(keys=["image", "lm"])

        self.Ind = FgBgToIndicesd2(
            keys=["lm"], image_key="image", image_threshold=-2600
        )
        self.Ai = DictToMeta(
            keys=["image"], meta_keys=["image_fname"], renamed_keys=["filename"]
        )
        self.T = ToDeviceD(keys=["image", "lm"], device=self.device)
        self.Am = DictToMeta(
            keys=["lm"],
            meta_keys=[
                "lm_fname",
                "remapping_source",
                "lm_fg_indices",
                "lm_bg_indices",
            ],
            renamed_keys=[
                "filename",
                "remapping_source",
                "lm_fg_indices",
                "lm_bg_indices",
            ],
        )
        self.E = EnsureChannelFirstd(
            keys=["image", "lm"], channel_dim="no_channel"
        )  # funny shape output mismatch
        self.SpI = Spacingd(
            keys=["image"], pixdim=self.plan.get("spacing"), mode="trilinear"
        )
        self.SpL = Spacingd(
            keys=["lm"], pixdim=self.plan.get("spacing"), mode="nearest"
        )

        self.Rz = ResizeToTensord(
            keys=["lm"], key_template_tensor="image", mode="nearest"
        )

        # Sm = Spacingd(keys=["lm"], pixdim=self.spacing,mode="nearest")
        self.N = NormaliseClipd(
            keys=["image"],
            clip_range=self.global_properties["intensity_clip_range"],
            mean=self.mean,
            std=self.std,
        )
        self.Ch = ChangeDtyped(keys=["lm"], target_dtype=torch.uint8)

        tfms = [
            self.LS,
            self.Rem,
            self.T,
            self.Re,
            self.Ind,
            self.E,
            self.SpI,
            self.SpL,
            self.Rz,
            self.Ch,
        ]
        if self.clip_center == True:
            tfms.extend([self.N])
        if self.half_precision == True:
            self.H = HalfPrecisiond(keys=["image"])
            tfms.extend([self.H])
        self.transform = Compose(tfms)

    def set_normalization_values(self, mean_std_mode):
        if mean_std_mode == "dataset":
            self.mean = self.global_properties["mean_dataset_clipped"]
            self.std = self.global_properties["std_dataset_clipped"]
        else:
            self.mean = self.global_properties["mean_fg"]
            self.std = self.global_properties["std_fg"]

    def process(self, mini_df):
        data = self._create_data_dicts_from_df(mini_df)
        self.results = []
        for dici in data:
            print("Processing image file: {0}".format(dici["image"]))
            dici = self.transform(dici)
            inds = {
                "lm_fg_indices": dici["lm_fg_indices"],
                "lm_bg_indices": dici["lm_bg_indices"],
                "meta": dici["image"].meta,
            }
            self.save_indices(inds, self.indices_subfolder)
            self.save_pt(dici["image"][0], "images")
            self.save_pt(dici["lm"][0], "lms")
            self.extract_image_props(dici["image"][0])
        return  self.results


class ResampleDatasetniftiToTorch(Preprocessor):
    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder=None,
        half_precision=False,
        clip_center=False,
    ):

        try:
            existing_fldr = folder_names_from_plan(project, plan)['data_folder_source']
            existing_fldr = Path(existing_fldr)
            if existing_fldr.exists():
                print(
                    "Plan folder already exists:  {}.\nWill use existing folder to add data".format(
                    existing_fldr
                )
            )
            output_folder = existing_fldr
        except:
            pass
        self.clip_center = clip_center
        self.half_precision = half_precision
        self.remapping_key = "remapping_source"
        super().__init__(
            project, plan, output_folder=output_folder, data_folder=data_folder
        )

    def create_data_df(self):
        Preprocessor.create_data_df(self)
        remapping = self.plan.get(self.remapping_key)
        self.df = self.df.assign(remapping=[remapping] * len(self.df))

    def setup(
        self, overwrite=False, mean_std_mode="dataset", num_processes=8, device="cpu"
    ):
        self.create_data_df()
        self.register_existing_files()
        print("Overwrite:", overwrite)
        if overwrite == False:
            self.remove_completed_cases()
        if len(self.df) > 0:

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
                clip_center=self.clip_center,
                half_precision=self.half_precision,
                mean_std_mode=mean_std_mode,
                device=device,
            )
            self.actors = [
                NiftiResampler.remote(**actor_kwargs) for _ in range(self.n_actors)
            ]
            # self.mini_dfs = list_to_chunks(self.df, num_processes)

            self.mini_dfs = np.array_split(self.df, num_processes)

            # self.mini_dfs = divide(num_processes, self.df)

    def process(
        self,
    ):
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

        self.results_df= pd.DataFrame(il.chain.from_iterable(self.results))
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
        # add_plan_to_db(self.project,
        #     self.plan, db_path=self.project.db, data_folder_source=self.output_folder
        # )

    def generate_bboxes_from_masks_folder(
        self, bg_label=0, debug=False, num_processes=8
    ):
        masks_folder = self.output_folder / ("lms")
        print("Generating bbox info from {}".format(masks_folder))
        generate_bboxes_from_lms_folder(
            masks_folder,
            bg_label,
            debug,
            num_processes,
        )

    def update_specsfile(self):
        lbd_output_folder = self.project.lbd_folder / (self.output_folder.name)
        specs = {
            "plan": self.plan,
            "output_folder": str(self.output_folder),
            "lbd_output_folder": str(lbd_output_folder),
        }
        specs_file = self.output_folder.parent / ("resampling_configs")

        try:
            saved_specs = load_dict(specs_file)
            matches = [specs == dic for dic in saved_specs]
            if not any(matches):
                saved_specs.append(specs)
                save_dict(saved_specs, specs_file)
        except:
            saved_specs = [specs]
            save_dict(saved_specs, specs_file)

    #
    # def get_tensor_folder_stats(self, debug=True):
    #     img_filenames = (self.output_folder / ("images")).glob("*")
    #     args = [[img_fn] for img_fn in img_filenames]
    #     results = multiprocess_multiarg(get_tensorfile_stats, args, debug=debug,io=True)
    #     self.results = pd.DataFrame(results).values
    #     self._store_dataset_properties()

    def set_input_output_folders(self, data_folder, output_folder):
        self.data_folder = Path(data_folder)
        if output_folder is not None:
            self.output_folder = Path(output_folder)
        else:
            src_subfolder = folder_names_from_plan(self.project,self.plan)["data_folder_source"]
            self.output_folder = self.project.fixed_spacing_folder / (src_subfolder)


class FGBGIndicesResampleDataset(ResampleDatasetniftiToTorch):
    def __init__(self, project, plan, half_precision=False):
        super().__init__(project, plan, half_precision)

    def register_existing_files(self):
        self.existing_files = list(self.indices_subfolder.glob("*"))

    def process(
        self,
    ):
        if not hasattr(self, "dl"):
            print("No data loader created. No data to be processed")
            return 0
        print("resampling dataset to spacing: {0}".format(self.plan.get("spacing")))
        self.create_output_folders()
        for batch in pbar(self.dl):
            self.process_batch(batch)

    def process_batch(self, batch):
        images, lms, fg_inds, bg_inds = (
            batch["image"],
            batch["lm"],
            batch["lm_fg_indices"],
            batch["lm_bg_indices"],
        )
        for (
            image,
            lm,
            fg_ind,
            bg_ind,
        ) in zip(
            images,
            lms,
            fg_inds,
            bg_inds,
        ):
            assert image.shape == lm.shape, "mismatch in shape".format(
                image.shape, lm.shape
            )
            assert image.dim() == 4, "images should be cxhxwxd"
            inds = {
                "lm_fg_indices": fg_ind,
                "lm_bg_indices": bg_ind,
                "meta": image.meta,
            }
            self.save_indices(inds, self.indices_subfolder)


# %%
if __name__ == "__main__":
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- P = Project("nodes") <CR>
    from fran.managers import Project
    from fran.preprocessing.fixed_spacing import ResampleDatasetniftiToTorch
    from fran.utils.common import *
    from fran.utils.config_parsers import ConfigMaker

# %%
    # chunkify = lambda l, n: [l[i : i + n] for i in range(0, len(l), n)]
    # aa = chunkify(Rs.df,16)

    P = Project("nodes")
    # P._create_plans_table()
    # P.add_data([DS.totalseg])
    C = ConfigMaker(P, raytune=False, configuration_filename=None)
    C.setup(6)
    C.plans
    plan = C.configs["plan_train"]
    conf = C.configs

# %%

    plan = conf["plan_train"]
    pp(plan)
    plan["mode"]

    folder_names_from_plan(P, plan)
# %%
    # add_plan_to_db(plan,"/r/datasets/preprocessed/totalseg/lbd/spc_100_100_100_plan5",P.db)
    Rs = ResampleDatasetniftiToTorch(P, plan, P.raw_data_folder)
    Rs.output_folder.exists()

    overwrite = False
    n_processes = 12
    Rs.setup(num_processes=n_processes, overwrite=overwrite)

# %%

    num_processes=12
    Rs.create_data_df()
    Rs.register_existing_files()
    mini_df = Rs.mini_dfs[0]
    Rs.mini_dfs = np.array_split(Rs.df, num_processes)
# %%
    Rs.process()
    #
    rr = Rs.results

    import itertools as il
# %%
    dds = list_to_chunks(Rs.df, n_processes)
    mini_df = Rs.df.iloc[:10]

    mean_std_mode = "dataset"
    actor_kwargs = dict(
        project=Rs.project,
        plan=Rs.plan,
        data_folder=Rs.data_folder,
        output_folder=Rs.output_folder,
        clip_center=Rs.clip_center,
        half_precision=Rs.half_precision,
        mean_std_mode=mean_std_mode,
    )

# %%

    ts = Rs.results_df.shape
    if ts[-1] == 4:  # only store if entire dset is processed
        Rs._store_dataset_properties()
        generate_bboxes_from_lms_folder(Rs.output_folder / ("lms"))
    else:
        print(
            "Rs.results  shape is {0}. Last element should be 4 , is {1}. therefore".format(
                ts, ts[-1]
            )
        )
        print(
            "since some files skipped, dataset stats are not being stored. run Rs.get_tensor_folder_stats and generate_bboxes_from_lms_folder separately"
        )
    add_plan_to_db(
        Rs.plan, db_path=Rs.project.db, data_folder_source=Rs.output_folder
    )
# %%
    N = NiftiResampler(**actor_kwargs)
    # for mini_df in dds:
    N.process(mini_df)

# %%
    remapping = create_remapping(N.plan, "remapping_source", as_dict=True)
    data = []
    for index in range(len(df)):
        row = df.iloc[index]
        dici = N._dici_from_df_row(row, remapping)
        data.append(dici)
# %%
    maybe_makedirs(N.indices_subfolder)
    dici = data[0]
    for dici in data:
        dici = N.transform(dici)
        inds = {
            "lm_fg_indices": dici["lm_fg_indices"],
            "lm_bg_indices": dici["lm_bg_indices"],
            "meta": dici["image"].meta,
        }
        N.save_indices(inds, N.indices_subfolder)
        N.save_pt(dici["image"][0], "images")
        N.save_pt(dici["lm"][0], "lms")
        N.extract_image_props(image)
# %%
# SECTION:-------------------- ResampleDatasetniftiToTorch-------------------------------------------------------------------------------------- <CR> <CR>
    spacing = [1, 1, 1]
    project = P
    overwrite = False
    Rs = ResampleDatasetniftiToTorch(
        project,
        plan,
        data_folder="/s/xnat_shadow/crc/sampling/nifti",
        output_folder="/s/xnat_shadow/crc/sampling/tensors/fixed_spacing/",
    )
    Rs.setup(overwrite=overwrite)
    Rs.process()
# %%
    F = FGBGIndicesResampleDataset(project, spacing=[0.8, 0.8, 1.5])
    F.setup()
    F.process()
    # R.register_existing_files()

# %%

# %%
    L = LoadSITKd(keys=["image", "lm"], image_only=True)
    R = LabelRemapd(keys=["lm"], remapping_key="remapping")
    T = ToDeviced(keys=["image", "lm"], device=Rs.ds.device)
    Re = Recastd(keys=["image", "lm"])

    Ind = FgBgToIndicesd2(keys=["lm"], image_key="image", image_threshold=-2600)
    Ai = DictToMeta(
        keys=["image"], meta_keys=["image_fname"], renamed_keys=["filename"]
    )
    Am = DictToMeta(
        keys=["lm"],
        meta_keys=["lm_fname", "remapping", "lm_fg_indices", "lm_bg_indices"],
        renamed_keys=["filename", "remapping", "lm_fg_indices", "lm_bg_indices"],
    )
    E = EnsureChannelFirstd(
        keys=["image", "lm"], channel_dim="no_channel"
    )  # funny shape output mismatch
    Si = Spacingd(keys=["image"], pixdim=Rs.ds.spacing, mode="trilinear")
    Sl = Spacingd(keys=["lm"], pixdim=Rs.ds.spacing, mode="nearest")
    Rz = ResizeToTensord(keys=["lm"], key_template_tensor="image", mode="nearest")

    # Sm = Spacingd(keys=["lm"], pixdim=Rs.ds.spacing,mode="nearest")
    N = NormaliseClipd(
        keys=["image"],
        clip_range=Rs.ds.global_properties["intensity_clip_range"],
        mean=Rs.ds.mean,
        std=Rs.ds.std,
    )
    Ch = ChangeDtyped(keys=["lm"], target_dtype=torch.uint8)

    # tfms = [R, L, T, Re, Ind, Ai, Am, E, Si, Rz,Ch]
    tfms = [L, R, T, Re, Ind, E, Si, Rz, Ch]
# %%
    dici = Rs.ds[0]
    dici["lm"].meta
    dici = Rs.ds.data[0]

# %%
    dici = L(dici)

    dici = R(dici)
    dici = T(dici)
    dici = Re(dici)
    dici = Ind(dici)
    dici = E(dici)
    dici = Si(dici)
    dici = Rz(dici)
    dici = Ch(dici)

# %%
    print(dici["image"].meta["filename_or_obj"], dici["lm"].meta["filename_or_obj"])

    L = LoadSITKd(keys=["image", "lm"], image_only=True)
    T = ToDeviced(keys=["image", "lm"], device=Rx.device)
    Re = Recastd(keys=["image", "lm"])

    Ind = FgBgToIndicesd(keys=["lm"], image_key="image", image_threshold=-2600)
    Ai = DictToMeta(
        keys=["image"], meta_keys=["image_fname"], renamed_keys=["filename"]
    )
    Am = DictToMeta(
        keys=["lm"],
        meta_keys=["lm_fname", "remapping", "lm_fb_indices", "lm_fg_indices"],
        renamed_keys=["filename", "remapping", "lm_fb_indices", "lm_fg_indices"],
    )
    E = EnsureChannelFirstd(
        keys=["image", "lm"], channel_dim="no_channel"
    )  # funny shape output mismatch
    Si = Spacingd(keys=["image"], pixdim=Rx.spacing, mode="trilinear")
    Rz = ResizeDynamicd(keys=["lm"], key_spatial_size="image", mode="nearest")

    # Sm = Spacingd(keys=["lm"], pixdim=Rx.spacing,mode="nearest")
    N = NormaliseClipd(
        keys=["image"],
        clip_range=Rx.global_properties["intensity_clip_range"],
        mean=Rx.mean,
        std=Rx.std,
    )

    tf = Compose([R, L, T, Re, Ind])
    dici = tf(dici)
# %%
    I.Resampler.shapes = np.array(I.Resampler.shapes)
    fn_dict = I.Resampler.output_folder / "info.json"

    dici = {
        "median_shape": np.median(I.Resampler.shapes, 0).tolist(),
        "spacing": I.Resampler.spacing,
    }
    save_dict(dici, fn_dict)

    resampled_dataset_properties["median_shape"] = np.median(
        I.Resampler.shapes, 0
    ).tolist()
# %%
    tnsr = torch.load(
        "/s/fran_storage/datasets/preprocessed/fixed_spacing/litsmc/spc_080_080_150/lms/drli_001ub.nii.gz"
    )
    fn = Path(tnsr.meta["filename"])
    fn_name = strip_extension(fn.name) + ".pt"
    fn_out = fn.parent / (fn_name)
    generate_bboxes_from_lms_folder(R.output_folder / ("lms"))
# %%

    existing_fnames = [fn.name for fn in R.existing_files]
    df = R.df.copy()
    rows_new = []
# %%
    for i in range(len(df)):
        row = df.loc[i]
        df_fname = Path(row.lm_symlink)
        df_fname = strip_extension(df_fname.name) + ".pt"
        if df_fname in existing_fnames:
            df.drop([i], inplace=True)
            # rows_new.append(row)

# %%
        L.shapes = np.array(L.shapes)
        resampled_dataset_properties = dict()
        resampled_dataset_properties["median_shape"] = np.median(L.shapes, 0).tolist()
        resampled_dataset_properties["dataset_spacing"] = L.spacing
        resampled_dataset_properties["dataset_max"] = L.results["max"].max().item()
        resampled_dataset_properties["dataset_min"] = L.results["min"].min().item()
        resampled_dataset_properties["dataset_median"] = np.median(
            L.results["median"]
        ).item()

# %%
    df = pd.DataFrame(np.arange(12).reshape(3, 4), columns=["A", "B", "C", "D"])
    df.drop(["A", "B"], axis=1)
    df
    dici = load_dict()
    save_json(
        resampled_dataset_properties,
        "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150_plan3/resampled_dataset_properties.json",
    )
# %%
    dl = I.R.dl
    iteri = iter(dl)
    batch = next(iteri)
# %%
    U = ToCPUd(keys=["image", "lm", "lm_fg_indices", "lm_bg_indices"])
    batch = U(batch)
    images, lms, fg_inds, bg_inds = (
        batch["image"],
        batch["lm"],
        batch["lm_fg_indices"],
        batch["lm_bg_indices"],
    )
    for (
        image,
        lm,
        fg_ind,
        bg_ind,
    ) in zip(
        images,
        lms,
        fg_inds,
        bg_inds,
    ):
        assert image.shape == lm.shape, "mismatch in shape".format(
            image.shape, lm.shape
        )
        assert image.dim() == 4, "images should be cxhxwxd"

        inds = {
            "lm_fg_indices": fg_ind,
            "lm_bg_indices": bg_ind,
            "meta": image.meta,
        }
        I.R.save_indices(inds, I.R.indices_subfolder)
        I.R.save_pt(image[0], "images")
        I.R.save_pt(lm[0], "lms")
        I.R.extract_image_props(image)


# %%
