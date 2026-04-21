# %%
from pathlib import Path

import ipdb
import pandas as pd
import ray
import torch
from fran.preprocessing import bboxes_function_version
from fran.preprocessing.helpers import (
    create_dataset_stats_artifacts,
    infer_dataset_stats_window,
)
from fran.preprocessing.preprocessor import (
    Preprocessor,
    get_tensor_stats,
    store_label_count,
)
from fran.preprocessing.rayworker_base import RayWorkerBase
from fran.transforms.imageio import LoadSITKd
from fran.transforms.intensitytransforms import NormaliseClipd
from fran.transforms.misc_transforms import (
    ChangeDtyped,
    DictToMetad,
    HalfPrecisiond,
    RecastToFloatd,
)
from fran.transforms.spatialtransforms import ResizeToTensord
from fran.utils.folder_names import FolderNames
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from tqdm.auto import tqdm as pbar
from utilz.fileio import save_dict
from utilz.helpers import find_matching_fn, multiprocess_multiarg


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


class _NiftiResamplerBase(RayWorkerBase):
    remapping_key = "remapping_source"

    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder,
        half_precision=False,
        clip_center=False,
        mean_std_mode="dataset",
        device="cuda",
        debug=False,
    ):

        self.project = project
        self.global_properties = self.project.global_properties
        self.half_precision = half_precision
        self.clip_center = clip_center
        self.device = device
        self.set_normalization_values(mean_std_mode)
        tfms_keys = "LoadS,Chan,Orient,Remap,Dev,Cast,SpImg,SpLm,Rsz,LmDType,Indx"
        if self.clip_center == True:
            tfms_keys += ",Norm"
        if self.half_precision == True:
            tfms_keys += ",Half"
        super().__init__(
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
            device=device,
            debug=debug,
            tfms_keys=tfms_keys,
        )

    def set_input_output_folders(self, data_folder, output_folder):
        self.data_folder = data_folder
        self.output_folder = output_folder

    @classmethod
    def _create_data_dict(cls, row):
        return {
            "image": row["image"],
            "lm": row["lm"],
            cls.remapping_key: row[cls.remapping_key],
            "ds": row["ds"],
        }

    def _create_data_dicts_from_folder(self):
        """Create data dictionaries from data_folder structure."""
        data_folder = Path(self.data_folder)
        masks_folder = data_folder / "lms"
        images_folder = data_folder / "images"

        img_fns = list(images_folder.glob("*"))
        data = []

        for img_fn in img_fns:
            lm_fn = find_matching_fn(img_fn.name, masks_folder, "case_id")[0]

            remapping = None

            dici = {
                "image": str(img_fn),
                "lm": str(lm_fn),
                "remapping_imported": remapping,
            }
            data.append(dici)
        assert len(data) > 0, "No data found in data folder"
        return data

    def create_transforms(self, device="cpu"):
        super().create_transforms(device=device)
        self.LS = LoadSITKd(keys=["image", "lm"], image_only=True)
        self.Or = Orientationd(keys=["image", "lm"], axcodes="RAS")
        self.Re = RecastToFloatd(keys=["image", "lm"])
        self.Ai = DictToMetad(  # CODE: REDUNDANT?
            keys=["image"], meta_keys=["image_fname"], renamed_keys=["filename"]
        )
        self.Am = DictToMetad(  # CODE: REDUNDANT?
            keys=["lm"],
            meta_keys=[
                "lm_fname",
                self.remapping_key,
                "lm_fg_indices",
                "lm_bg_indices",
            ],
            renamed_keys=[
                "filename",
                self.remapping_key,
                "lm_fg_indices",
                "lm_bg_indices",
            ],
        )
        self.SpI = Spacingd(
            keys=["image"], pixdim=self.plan.get("spacing"), mode="trilinear"
        )
        self.SpL = Spacingd(
            keys=["lm"], pixdim=self.plan.get("spacing"), mode="nearest"
        )

        self.Rsz = ResizeToTensord(
            keys=["lm"], key_template_tensor="image", mode="nearest"
        )

        # Sm = Spacingd(keys=["lm"], pixdim=self.spacing,mode="nearest")
        self.N = NormaliseClipd(
            keys=["image"],
            clip_range=self.global_properties["intensity_clip_range"],
            mean=self.mean,
            std=self.std,
        )
        self.LmDType = ChangeDtyped(keys=["lm"], target_dtype=torch.uint8)
        self.H = HalfPrecisiond(keys=["image"])
        add_transforms_dict = {
            "LoadS": self.LS,
            "Orient": self.Or,
            "Cast": self.Re,
            "SpImg": self.SpI,
            "SpLm": self.SpL,
            "Rsz": self.Rsz,
            "Norm": self.N,
            "LmDType": self.LmDType,
            "Half": self.H,
        }
        self.transforms_dict.update(add_transforms_dict)

    def _process_row(self, row: pd.Series):
        data = self._create_data_dict(row)
        data = self.apply_transforms(data)
        image = data["image"]
        lm = data["lm"]
        lm_fg_indices = data["lm_fg_indices"]
        lm_bg_indices = data["lm_bg_indices"]

        assert image.shape == lm.shape, "mismatch in shape"
        assert image.dim() == 4, "images should be cxhxwxd"

        inds = {
            "lm_fg_indices": lm_fg_indices,
            "lm_bg_indices": lm_bg_indices,
            "meta": image.meta,
        }
        self.save_indices(inds, self.indices_subfolder)
        self.save_pt(image[0], "images")
        self.save_pt(lm[0], "lms")
        return get_tensor_stats(image[0])

    def set_normalization_values(self, mean_std_mode):
        if mean_std_mode == "dataset":
            self.mean = self.global_properties["mean_dataset_clipped"]
            self.std = self.global_properties["std_dataset_clipped"]
        else:
            self.mean = self.global_properties["mean_fg"]
            self.std = self.global_properties["std_fg"]

    @property
    def indices_subfolder(self):
        return self.output_folder / "indices"


@ray.remote(num_cpus=4)
class NiftiResampler(_NiftiResamplerBase):
    pass


class NiftiResamplerLocal(_NiftiResamplerBase):
    pass


class NiftiToTorchDataGenerator(Preprocessor):
    actor_cls = NiftiResampler
    local_worker_cls = NiftiResamplerLocal
    remapping_key = "remapping_source"
    subfolder_key = "data_folder_source"

    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder=None,
        half_precision=False,
        clip_center=False,
    ):

        existing_fldr = FolderNames(project, plan).folders[self.subfolder_key]
        existing_fldr = Path(existing_fldr)
        if existing_fldr.exists():
            print(
                "Plan folder already exists:  {}.\nWill use existing folder to add data".format(
                    existing_fldr
                )
            )
            output_folder = existing_fldr
        self.clip_center = clip_center
        self.half_precision = half_precision
        super().__init__(
            project, plan, output_folder=output_folder, data_folder=data_folder
        )

    def extra_worker_kwargs(self, **setup_kwargs):
        return {
            "clip_center": self.clip_center,
            "half_precision": self.half_precision,
            "mean_std_mode": setup_kwargs["mean_std_mode"],
        }

    def should_use_ray(self):
        return (self.num_processes > 1) and (not getattr(self, "debug", False))

    def setup(
        self,
        overwrite=False,
        mean_std_mode="dataset",
        num_processes=8,
        device="cpu",
        debug=False,
    ):
        super().setup(
            overwrite=overwrite,
            num_processes=num_processes,
            device=device,
            debug=debug,
            mean_std_mode=mean_std_mode,
        )

    def create_data_df(self):
        Preprocessor.create_data_df(self)
        remapping = self.plan.get(self.remapping_key)
        self.df = self.df.assign(remapping=[remapping] * len(self.df))

    def postprocess_results(self, **process_kwargs):
        ts = self.results_df.shape
        if ts[-1] == 4:  # only store if entire dset is processed
            self._store_dataset_properties()
            generate_bboxes_from_lms_folder(
                self.output_folder / ("lms"),
                num_processes=getattr(self, "num_processes", 1),
            )
        else:
            print(
                "self.results  shape is {0}. Last element should be 4 , is {1}. therefore".format(
                    ts, ts[-1]
                )
            )
            print(
                "since some files skipped, dataset stats are not being stored. run self.get_tensor_folder_stats and generate_bboxes_from_lms_folder separately"
            )
        store_label_count(
            self.output_folder, num_processes=getattr(self, "num_processes", 1)
        )
        create_dataset_stats_artifacts(
            lms_folder=self.output_folder / "lms",
            gif=self.store_gifs,
            label_stats=self.store_label_stats,
            gif_window=infer_dataset_stats_window(self.project),
        )

    def generate_bboxes_from_lms_folder(self, bg_label=0, debug=False, num_processes=8):
        masks_folder = self.output_folder / ("lms")
        print("Generating bbox info from {}".format(masks_folder))
        generate_bboxes_from_lms_folder(
            masks_folder,
            bg_label,
            debug,
            num_processes,
        )

    def set_input_output_folders(self, data_folder, output_folder):
        self.data_folder = Path(data_folder)
        if output_folder is not None:
            self.output_folder = Path(output_folder)
        else:
            src_subfolder = FolderNames(self.project, self.plan).folders[
                self.subfolder_key
            ]
            self.output_folder = src_subfolder

    @property
    def indices_subfolder(self):
        indices_subfolder = self.output_folder / "indices"
        return indices_subfolder


class FGBGIndicesResampleDataset(NiftiToTorchDataGenerator):
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


ResampleDatasetniftiToTorch = NiftiToTorchDataGenerator


# %%
if __name__ == "__main__":
    import numpy as np

# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- P = Project("nodes") <CR> <CR> <CR> <CR>
    from fran.configs.parser import ConfigMaker, parse_nested_remapping
    from fran.inference.base import list_to_chunks
    from fran.managers import Project
    from utilz.fileio import load_dict
    from utilz.helpers import set_autoreload

    set_autoreload()

    from fran.preprocessing.fixed_spacing import ResampleDatasetniftiToTorch
    from fran.transforms.inferencetransforms import ToCPUd
    from fran.transforms.fg_indices import FgBgToIndicesd2, LabelRemapd
    from fran.utils.common import *
    from monai.transforms.utility.dictionary import (
        EnsureChannelFirstd,
        FgBgToIndicesd,
        ToDeviced,
    )
    from utilz.fileio import maybe_makedirs, save_json
    from utilz.stringz import strip_extension
# %%
    # chunkify = lambda l, n: [l[i : i + n] for i in range(0, len(l), n)]
    # aa = chunkify(Rs.df,16)

    # P = Project("test")
    P = Project("kits23")

    # P._create_plans_table()
    # P.add_data([DS.totalseg])
    C = ConfigMaker(P)
    C.setup(2)
    C.plans
    plan = C.configs["plan_train"]
    conf = C.configs

# %%

    plan = conf["plan_train"]
    plan["mode"]

    FolderNames(P, plan).folders
    print(P.global_properties)
# %%
    # add_plan_to_db(plan,"/r/datasets/preprocessed/totalseg/lbd/spc_100_100_100_plan5",P.db)
    Rs = ResampleDatasetniftiToTorch(P, plan, P.raw_data_folder)
# %%

    Rs.output_folder.exists()

    debug_ = False
    overwrite = True
    n_processes = 2
    Rs.setup(num_processes=n_processes, overwrite=overwrite, debug=debug_)
    Rs.process()

# %%
    create_dataset_stats_artifacts(
        lms_folder=Rs.output_folder,
        gif=True,
        label_stats=True,
        gif_window=infer_dataset_stats_window(Rs.project),
    )
    # Rs.process()
# %%
# SECTION:-------------------- TS--------------------------------------------------------------------------------------# %%

    num_processes = 1
    Rs.create_data_df()
    Rs.register_existing_files()
    mini_df = Rs.mini_dfs[0]
    Rs.mini_dfs = Rs.split_dataframe_for_workers(Rs.df, num_processes)
# %%
    Rs.process()
    #
    rr = Rs.results

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
    add_plan_to_db(Rs.plan, db_path=Rs.project.db, data_folder_source=Rs.output_folder)
# %%
    N = NiftiResampler(**actor_kwargs)
    # for mini_df in dds:
    N.process(mini_df)

# %%
    remapping = parse_nested_remapping(N.plan, "remapping_source", as_dict=True)
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
# SECTION:-------------------- ResampleDatasetniftiToTorch-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR>
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
    Re = RecastToFloatd(keys=["image", "lm"])

    Ind = FgBgToIndicesd2(keys=["lm"], image_key="image", image_threshold=-2600)
    Ai = DictToMetad(
        keys=["image"], meta_keys=["image_fname"], renamed_keys=["filename"]
    )
    Am = DictToMetad(
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
    Re = RecastToFloatd(keys=["image", "lm"])

    Ind = FgBgToIndicesd(keys=["lm"], image_key="image", image_threshold=-2600)
    Ai = DictToMetad(
        keys=["image"], meta_keys=["image_fname"], renamed_keys=["filename"]
    )
    Am = DictToMetad(
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

    df = Rs.df
    datasources = Rs.plan.get("datasources")
    datasources = datasources.split(",")
    remappings = Rs.plan.get(Rs.remapping_key)
    assert len(remappings) == len(datasources), (
        f"There should be a unique remapping for each datasource.\n Got {len(datasources)} datasources and {len(remappings)} remappingss"
    )
# %%
    for ds, remapping in zip(datasources, remappings):
        print(remapping)
        # Use .at or .loc with a list to assign the entire dictionary object
        mask = df["ds"] == ds
        Rs.df.loc[mask, "remapping"] = [remapping] * mask.sum()

# %%

    datasources["lms"] = Rs.plan["datasources"]["lms"]
    Rs.plan["datasources"] = datasources


# %%

