# %%

import os
import shutil
from pathlib import Path

import ipdb
from monai.transforms.utility.dictionary import SplitDimD
from monai.transforms import SqueezeDimD
import pandas as pd
import ray
import SimpleITK as sitk
import torch
from utilz.cprint import cprint
from label_analysis.totalseg import TotalSegmenterLabels
from monai.transforms.spatial.dictionary import GridPatchd
from utilz.fileio import *
from utilz.fileio import load_dict, maybe_makedirs, save_dict
from utilz.helpers import *
from utilz.helpers import multiprocess_multiarg
from utilz.imageviewers import *
from utilz.stringz import headline

from fran.configs.parser import ConfigMaker
from fran.preprocessing import bboxes_function_version
from fran.preprocessing.helpers import sanitize_meta_for_monai
from fran.preprocessing.helpers import to_even
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.preprocessing.rayworker_base import RayWorkerBase
from fran.utils.folder_names import folder_names_from_plan

tr = ipdb.set_trace
MIN_SIZE = 32

from fran.preprocessing.preprocessor import Preprocessor, store_labels_info


class _PBDSamplerWorkerBase(RayWorkerBase):
    def __init__(
        self, project, plan, data_folder, output_folder, device="cpu", debug=False
    ):
        super().__init__(
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
            device=device,
            debug=debug,
            tfms_keys="LoadT,Chan,Dev,Grid,Split,Labels, Indx, Sq"
        )
        self.output_folder = Path(self.output_folder)

    def create_transforms(self, device="cpu"):
        super().create_transforms(device=device)

        patch_overlap = self.plan.get("patch_overlap", 0.20)
        patch_size = self.plan["patch_size"]
        self.G = GridPatchd(
            keys=self.tnsr_keys, patch_size=patch_size, overlap=patch_overlap
        )
        self.Sq = SqueezeDimD(keys=self.tnsr_keys, dim=0)
        self.Split = SplitDimD(keys=self.tnsr_keys, dim=0, list_output=True,keepdim=False)
        self.transforms_dict["Grid"] = self.G
        self.transforms_dict["Split"] = self.Split
        self.transforms_dict["Sq"] = self.Sq

    def _create_data_dict(self, row):
        data = {
            "image": row["image"],
            "lm": row["lm"],
        }
        return data


    def _process_row(self, row: pd.Series):
        case_id = row["case_id"]
        data_ = self._create_data_dict(row)
        data = self.apply_transforms(data_)
        row_results = []
        for n in range(len(data)):
            patch = data[n]
            assert patch["image"].shape == patch["lm"].shape, "mismatch in shape"
            assert patch["image"].dim() == 3, "images should be n_patchesxhxwxd"
            assert (
                patch["image"].numel() > MIN_SIZE**3
            ), f"image size is too small {patch['image'].shape}"
            image = patch["image"]
            lm = patch["lm"]
            inds = {"lm_fg_indices": patch['lm_fg_indices'],"lm_bg_indices":patch['lm_bg_indices'] }

            fn_name = self.create_patch_fname(image.meta, n)
            labels = patch['lm_labels']
            has_fg = any([lb > 0 for lb in labels])

            self.save_indices_patch(inds,fn_name, "indices")
            self.save_pt_patch(lm, fn_name, "lms")
            self.save_pt_patch(image, fn_name, "images")
            results = {
                "case_id": case_id,
                "fn_name": fn_name,
                "ok": True,
                "shape": list(image.shape),
                "has_fg": has_fg,
                "n_fg": len(patch['lm_fg_indices']),
                "n_bg": len(patch['lm_bg_indices']),
                "labels": patch['lm_labels'],
            }

            row_results.append(results)
        return row_results

    def save_pt_patch(
        self, tnsr, fn_name, subfolder, contiguous=True, suffix: str = None
    ):
        if contiguous == True:
            tnsr = tnsr.contiguous()
        if hasattr(tnsr, "meta") and isinstance(tnsr.meta, dict):
            tnsr.meta = sanitize_meta_for_monai(dict(tnsr.meta))
        fn = self.output_folder / subfolder / fn_name
        try:
            torch.save(tnsr, fn)
        except OSError as e:
            # get filesystem info
            try:
                usage = shutil.disk_usage(os.path.dirname(fn))
                fsinfo = f"Total={usage.total//(1024**3)}G, Used={usage.used//(1024**3)}G, Free={usage.free//(1024**3)}G"
            except Exception:
                fsinfo = "disk usage unavailable"

            print(f"[ERROR] Failed saving to {fn}")
            print(f"[ERROR] Filesystem info: {fsinfo}")

            raise RuntimeError(f"Quota exceeded at path: {fn}") from e

    def save_indices_patch(self, indices_dict, fn_name, subfolder):
        if isinstance(subfolder, Path):
            fn = subfolder / fn_name
        else:
            fn = self.output_folder / subfolder / fn_name
        torch.save(indices_dict, fn)

    def create_patch_fname(self, meta, n):
        fn_name = meta["filename_or_obj"]
        fn_name = Path(fn_name).name
        fn_name = fn_name.replace(".pt", f"_{n}.pt")
        return fn_name


@ray.remote(num_cpus=1)
class PBDSamplerWorkerImpl(_PBDSamplerWorkerBase):
    pass


class PBDSamplerWorkerLocal(_PBDSamplerWorkerBase):
    pass


class PatchDataGenerator(LabelBoundedDataGenerator, Preprocessor):

    def __init__(self, project, plan, data_folder, output_folder=None, patch_overlap = 0.2):

        existing_fldr = folder_names_from_plan(project, plan).get("data_folder_pbd")
        existing_fldr = Path(existing_fldr)
        if existing_fldr.exists():
            headline(
                "Plan folder already exists: {}.\nWill use existing folder to add data".format(
                    existing_fldr
                )
            )
            output_folder = existing_fldr
        self.plan = plan

        Preprocessor.__init__(
            self,
            project=project,
            plan=plan,
            data_folder=data_folder,
            output_folder=output_folder,
        )
        self.actor_cls = PBDSamplerWorkerImpl
        self.local_worker_cls = PBDSamplerWorkerLocal
        self.remapping_key = None

    def create_data_df(self):
        Preprocessor.create_data_df(self)

    def set_input_output_folders(self, data_folder, output_folder):
        if data_folder is None:
            data_folder = folder_names_from_plan(self.project, self.plan)[
                "data_folder_lbd"
            ]
        self.data_folder = Path(data_folder)
        if output_folder is None:
            pbd_subfolder = folder_names_from_plan(self.project, self.plan)[
                "data_folder_pbd"
            ]
            self.output_folder = Path(pbd_subfolder)
        else:
            self.output_folder = Path(output_folder)
        cprint(f"Data folder is {self.data_folder}", color="yellow")

    def generate_bboxes(self, num_processes=24, debug=False):
        lms_folder = self.output_folder / "lms"
        lm_filenames = list(lms_folder.glob("*pt"))
        bg_label = 0
        arguments = [[x, bg_label] for x in lm_filenames]
        res_cropped = multiprocess_multiarg(
            func=bboxes_function_version,
            arguments=arguments,
            num_processes=num_processes,
            debug=debug,
        )

        stats_outfilename = (lms_folder.parent) / ("bboxes_info.json")
        print("Saving bbox stats to {}".format(stats_outfilename))
        save_dict(res_cropped, stats_outfilename)

    def process(self, derive_bboxes=False):
        if not hasattr(self, "df") or len(self.df) == 0:
            print("No data loader created. No data to be processed")
            return 0
        self.create_output_folders()
        if getattr(self, "use_ray", False):
            self.results = ray.get(
                [
                    actor.process.remote(mini_df)
                    for actor, mini_df in zip(self.actors, self.mini_dfs)
                ]
            )
        else:
            self.results = [self.local_worker.process(self.mini_dfs[0])]

        # PBD worker returns nested lists: actor -> case -> patch dict.
        flat_rows = []

        def _flatten(obj):
            if isinstance(obj, dict):
                flat_rows.append(obj)
                return
            if isinstance(obj, (list, tuple)):
                for x in obj:
                    _flatten(x)

        _flatten(self.results)
        self.results_df = pd.DataFrame(flat_rows)

        if derive_bboxes:
            has_patch_rows = len(self.results_df) > 0
            if has_patch_rows:
                self.generate_bboxes(
                    num_processes=getattr(self, "num_processes", 1),
                    debug=self.debug,
                )
            else:
                print("No patch rows produced; skipping bbox generation")
        else:
            print("No bboxes generated")

        self.results_df.to_csv(self.output_folder / "resampled_dataset_properties.csv", index=False)
        self.store_labels_info()
        self.create_dataset_stats_artifacts()



# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
if __name__ == "__main__":

    from fran.managers import Project
    from fran.utils.common import *

    project_title = "lidc"
    P = Project(project_title=project_title)
    # P.maybe_store_projectwide_properties()
    # spacing = [1.5, 1.5, 1.5]

    C = ConfigMaker(P)
    C.setup(6)
    C.plans
    conf = C.configs
    print(conf["model_params"])

    plan = conf["plan_train"]
    pp(plan)
# %%
    spacing = plan["spacing"]
    # plan["remapping_imported"][0]
    existing_fldrs = folder_names_from_plan(P, plan)

    fldr_pbd = existing_fldrs["data_folder_pbd"]

    src_plan = plan["source_plan"]

    src_plan_idx, src_plan_mode = src_plan.replace(" ", "").split(",")
    src_plan_idx = int(src_plan_idx)
    C2 = ConfigMaker(P)
    C2.setup(src_plan_idx)
    src_plan_full = C2.configs["plan_train"]
    data_fldrs = folder_names_from_plan(P, src_plan_full)
    data_folder = data_fldrs[f"data_folder_{src_plan_mode}"]
    data_foldre = Path(data_folder)
    patch_size = plan["patch_size"]
    patch_overlap = plan["patch_overlap"]
    deb = False
# %%
# SECTION:-------------------- PATCHGENERATOR-------------------------------------------------------------------------------------- <CR> <CR> <CR>
    img_fldr = data_foldre / "images"

    lm_fldr = data_foldre / "lms"

    output_folder = fldr_pbd
    P = PatchDataGenerator(
        project=P, plan=plan, output_folder=output_folder, data_folder=data_folder
    )
# %%
    P.setup()
    P.process()
# %%
    P2 = _PBDSamplerWorkerBase(
        project=P,
        plan=plan,
        data_folder=data_folder,
        output_folder=output_folder,
        device="cpu",
    )

# %%
    row = P.df.iloc[0]
    data = P2._create_data_dict(row)

    data2 = P2.apply_transforms(data)

    data = data2
# %%
# %%
    data2= P2.apply_transforms_compose(data)
    data2[0]['image'].shape
# %%

    C = P2.transforms_dict["Chan"]
    CR = P2.transforms_dict["Crop"]
    LoadT = P2.transforms_dict["LoadT"]
    Dev = P2.transforms_dict["Dev"]
    Remap = P2.transforms_dict["Remap"]
    Indx = P2.transforms_dict["Indx"]
    Grid = P2.transforms_dict["Grid"]
    Split = P2.transforms_dict["Split"]
    Split = SplitDimD(keys=["image", "lm"], dim=0, keepdim=False, list_output=True)
# %%
    dici = LoadT(data)
    dici2 = C(data)
    dic2 = Grid(dici2)
    dic2["image"].shape
    dic2["image"][1].meta

    dic3 = Indx(dic2)
    dic4 = Split(dic3)
    dic5 = P2.transforms_dict["Labels"](dic4)

# %%

# %%

    data = P2.apply_transforms(data)
    image = data["image"]
    lm = data["lm"]
    lm_fg_indices = data["lm_fg_indices"]
    lm_bg_indices = data["lm_bg_indices"]
    # Get metadata and indices

    assert image.shape == lm.shape, "mismatch in shape"
    assert image.dim() == 4, "images should be cxhxwxd"
    assert image.numel() > MIN_SIZE**3, f"image size is too small {image.shape}"
    inds = {
        "lm_fg_indices": lm_fg_indices,
        "lm_bg_indices": lm_bg_indices,
        "meta": image.meta,
    }

# %%
    img_fn = list(img_fldr.glob("*"))[0]

    lm_fn = img_fn.parent.parent / "lms" / img_fn.name

    img = torch.load(img_fn, weights_only=False)
    lm = torch.load(lm_fn, weights_only=False)
    img.shape
    img2 = img.unsqueeze(0)
    lm2 = lm.unsqueeze(0)

    dici = {"image": img2, "lm": lm2}
# %%
    G = GridPatchd(keys=["image", "lm"], patch_size=patch_size, overlap=patch_overlap)
    patch_overlap = 0.25
# %%
    dic2 = G(dici)
    dic2["image"].shape
# %%
    data_folder = "/s/xnat_shadow/lidc2"
    PG = PatchDataGenerator(
        project=P,
        plan=plan,
        data_folder=data_folder,
    )
# %%

# %%
    PG.setup(overwrite=True)
    PG.process(debug=deb)
# %%
    # PG.create_patches(overwrite=overwrite,debug=debug)
    lmg = "lm_group1"

    P.global_properties[lmg]

    imported_folder = Path("/s/fran_storage/predictions/totalseg/LITS-827/")
    TSL = TotalSegmenterLabels()
    imported_labelsets = TSL.labels("lung", "right"), TSL.labels("lung", "left")
    imported_labelsets = TSL.labels("all")
    P.imported_labels(lmg, imported_folder, imported_labelsets)
    # remapping = TSL.create_remapping(imported_labelsets, [8, 9])
# %%
# SECTION:-------------------- PATCHGENERATOR Single module-------------------------------------------------------------------------------------- <CR> <CR> <CR>
# %%

    patch_overlap = 0.25
    args = [
        [
            PG.dataset_properties,
            PG.output_folder,
            PG.patch_size,
            bb,
            patch_overlap,
            PG.expand_by,
            PG.mode,
        ]
        for bb in PG.fixed_sp_bboxes
    ]
# %%
    argi = args[0]
    info = argi[3]

    patch_overlap = [int(PG.patch_overlap * ps) for ps in PG.patch_size]
    patch_overlap = [to_even(ol) for ol in patch_overlap]
    expand_by = PG.expand_by
    mode = PG.mode
# %%
    output_folder = Path("/r/datasets/tmp")
    dataset_properties = PG.dataset_properties
    maybe_makedirs(output_folder)
    Pr = PatchGenerator(
        dataset_properties,
        output_folder,
        patch_size,
        info,
        patch_overlap,
        expand_by,
        mode,
    )
# %%
    Pr.process()

# %%

# %%
# SECTION:-------------------- ROUGH-------------------------------------------------------------------------------------- <CR> <CR> <CR>
    img_fn = "/r/datasets/tmp/images/lits_108_0.pt"
    lm_fn = "/r/datasets/tmp/lms/lits_108_0.pt"

    img = torch.load(img_fn)
    lm = torch.load(lm_fn)

    ImageMaskViewer([img, lm])

    inds_fldr = Path("/r/datasets/tmp/indices/")
    indices_fn = list(inds_fldr.glob("*pt"))
# %%
    for fn in indices_fn:
        inds = torch.load(fn)
        print(fn)
        print("FG", len(inds["lm_fg_indices"]))
# %%
    #
    # case_ids=PG.case_ids,
    # expand_by=20,
    # spacing=PG.spacing,
    # data_folder=PG.fixed_spacing_subfolder,
    # imported_folder=imported_folder,
    # remapping=self.remapping,
# %%
    # def set_transforms(self, keys_tr: str = "R,L1,L2,Re,E,Rz,M,B,A"):
    I.setup()
    dici = I[2]
    print(dici.keys())

# %%
    im = dici["image"]
    lm = dici["lm_imported"]
    ImageMaskViewer([im[0], lm[0]])
# %%

# %%
    dsrcs = P.global_properties[lmg]["ds"]
    ind = 0
    dsrc = dsrcs[ind]

    cids = P.df["case_id"][P.df["ds"] == dsrc].to_list()

    spacing = [0.8, 0.8, 1.5]
    fixed_folder = P.fixed_spacing_folder / ("spc_080_080_150/lms")
    fixed_files = list(fixed_folder.glob("*.pt"))

    fixed_files_out = []
# %%
    for cid in cids:
        fn = [fn for fn in fixed_files if cid in fn.name][0]
        fixed_files_out.append(fn)

# %%
    fn = fixed_files_out[-1]
    lm_pt = torch.load(fn)
    print(lm_pt.shape)
    imported_fn = find_matching_fn(fn, imported_folder.glob("*")[0])
    lm_imp = sitk.ReadImage(imported_fn)
    print(lm_imp.GetSpacing())
    dici = {"lm": lm_pt, "lm_imported": lm_imp, "remapping": remapping}

# %%
# %%
    fixed_spacing_folder = Path(
        "/home/ub/datasets/preprocessed/lidc2/fixed_spacing/spc_080_080_150"
    )
    patch_size = [192, 192, 120]
    patch_overlap = 0.2

    PG = PatchDataGenerator(
        P, fixed_spacing_folder, patch_size, patch_overlap, expand_by=10
    )
# %%

    patch_overlap = [int(PG.patch_overlap * ps) for ps in PG.patch_size]
    patch_overlap = [to_even(ol) for ol in patch_overlap]
    maybe_makedirs(PG.output_folder)
    PG.save_patches_config()
    if overwrite == False:
        PG.remove_completed_cases()
    args = [
        [
            PG.dataset_properties,
            PG.output_folder,
            PG.patch_size,
            bb,
            patch_overlap,
            PG.expand_by,
        ]
        for bb in PG.fixed_sp_bboxes
    ]

# %%
    dici = C(dici)
    lm_out = dici["lm"][dici["lm_imported"].meta["bounding_box"]]
    ImageMaskViewer([lm_out[0], dici["lm_imported"][0]], data_types=["lm", "lm"])
    view_sitk(dici["lm_imported"], lm_imp, data_types=["lm", "lm"])
    config = {
        "spacing": PI.spacing,
        "expand_by": PI.expand_by,
        "imported_folder": PI.imported_folder,
        "imported_labelsets": PI.imported_labelsets,
    }

# %%

    lm2 = merge_pt(dici["lm_imported"][0], dici["lm"])[0]
    ImageMaskViewer([lm_out[0], dici["lm_imported"][0]], data_types=["lm", "lm"])
    lm = sitk.ReadImage(fn)
    lm_out = merge(lm_imp, lm)

    view_sitk(lm, lm_out, data_types=["lm", "lm"])
# %%

    P = PatchDataGenerator(
        dataset_properties,
        output_folder,
        output_patch_size,
        info,
        patch_overlap,
        expand_by,
    )
    P.create_patches_from_all_bboxes()
    output_shape = [128, 128, 96]
    overs = 0.25
    fixed_folder = P.fixed_spacing_folder / ("spc_080_080_150/images")
    fixed_files = list(fixed_folder.glob("*.pt"))
    dataset_properties = load_dict(
        Path(
            "/s/fran_storage/datasets/preprocessed/fixed_spacings/lits/spc_080_080_150/resampled_dataset_properties.json"
        )
    )
    output_patch_size = [192, 192, 196]
    output_folder = Path("/home/ub/tmp")
    dici_fn = Path(
        "/s/fran_storage/datasets/preprocessed/fixed_spacings/lits/spc_080_080_150/bboxes_info.pkl"
    )
    inf = load_dict(dici_fn)

    info = inf[0]
# %%
# %%
    I.transform = C

    dici = I.data[0]
    dici = C(dici)
    print(dici["bounding_box"])
# %%
    dici = R(dici)
    dici = L1(dici)
    dici = L2(dici)
    dici = e(dici)
    dici = E(dici)
# %%
    bb = dici["bounding_box"]
    print(bb)
# %%
    P.shift_bboxes_by,
    P.output_patch_size,
    P.img.shape,
    P.add_to_bbox,

# %%SECTION:-------------------- ROUGH-------------------------------------------------------------------------------------- <CR> <CR>


# %%
