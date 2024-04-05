# %%
import torch
from fran.utils.dictopts import DictToAttr
import torchio as tio
from label_analysis.merge import merge, merge_pt
from label_analysis.totalseg import TotalSegmenterLabels
from fran.transforms.spatialtransforms import PadDeficitImgMask
from fran.utils.fileio import load_dict, maybe_makedirs, save_dict
from fran.utils.helpers import multiprocess_multiarg
from fran.utils.string import info_from_filename, strip_extension

if "get_ipython" in globals():
    print("setting autoreload")
    from IPython import get_ipython

    ipython = get_ipython()
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
from pathlib import Path

import ipdb
import numpy as np
import SimpleITK as sitk
from fastcore.basics import GetAttr,  store_attr

import ipdb
tr = ipdb.set_trace

from fran.preprocessing.datasetanalyzers import bboxes_function_version


def contains_bg_only( bbox_stats):
        all_fg_bbox = [bb for bb in bbox_stats if bb["label"] == "all_fg"][0]
        bboxes = all_fg_bbox["bounding_boxes"]
        if len(bboxes) == 1:
            return True
        elif bboxes[0] != bboxes[1]:
            return False
        else:
            tr()


class PatchDataGenerator(GetAttr):
    _default = "project"

    def __init__(
        self, project, fixed_spacing_folder, patch_size, patch_overlap=0.25, expand_by=None,mode='fgbg'
    ):
        store_attr()
        patches_fldr_name = "dim_{0}_{1}_{2}".format(*self.patch_size)
        self.output_folder = (
                self.patches_folder
                / self.fixed_spacing_folder.name
                / patches_fldr_name
            )


        fixed_sp_bboxes_fn = fixed_spacing_folder / ("bboxes_info")
        self.fixed_sp_bboxes = load_dict(fixed_sp_bboxes_fn)

        dataset_properties_fn = self.fixed_spacing_folder/ (
            "resampled_dataset_properties.json"
        )
        assert (
            dataset_properties_fn.exists()
        ), "Dataset properties file does not exist. Has the Resampling been run to create folder {}?".format(
            self.fixed_spacing_folder
        )
        self.dataset_properties = load_dict(dataset_properties_fn)
        self.create_output_folders()
        self.register_existing_files()

        self.patches_config_fn = self.output_folder / ("patches_config.json")

        if self.patches_config_fn.exists():
            print(
                "Patches configs already exist. Overriding given values with those from file"
            )
            patches_config = load_dict(self.patches_config_fn)
            print(patches_config)
            self.patch_overlap = patches_config["patch_overlap"]
            self.expand_by = patches_config["expand_by"]

    def register_existing_files(self):
        self.existing_files = list((self.output_folder / ("lms")).glob("*pt"))
        self.existing_case_ids = [
            info_from_filename(f.name, full_caseid=True)["case_id"]
            for f in self.existing_files
        ]
        self.existing_case_ids = set(self.existing_case_ids)

    def remove_completed_cases(self):
        all_cases = set([bb["case_id"] for bb in self.fixed_sp_bboxes])
        new_case_ids = all_cases.difference(self.existing_case_ids)
        print(
            "Total cases {0}.Found {1} new cases".format(
                len(all_cases), len(new_case_ids)
            )
        )
        self.fixed_sp_bboxes = [
            bb for bb in self.fixed_sp_bboxes if bb["case_id"] in new_case_ids
        ]

    def create_patches(self, overwrite=True, debug=False):
        patch_overlap = [int(self.patch_overlap * ps) for ps in self.patch_size]
        patch_overlap = [to_even(ol) for ol in patch_overlap]
        maybe_makedirs(self.output_folder)
        self.save_patches_config()
        if overwrite == False:
            self.remove_completed_cases()
        args = [
            [
                self.dataset_properties,
                self.output_folder,
                self.patch_size,
                bb,
                patch_overlap,
                self.expand_by,
                self.mode
            ]
            for bb in self.fixed_sp_bboxes
        ]

        res = multiprocess_multiarg(
            patch_generator_wrapper, args, debug=debug, progress_bar=True
        )

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

    def save_patches_config(self):
        patches_config = {
            "patch_overlap": self.patch_overlap,
            "expand_by": self.expand_by,
        }
        save_dict(patches_config, self.patches_config_fn)

    def create_output_folders(self):
        maybe_makedirs([self.output_folder / ("lms"), self.output_folder / ("images")])


class PatchGenerator(DictToAttr):
    def __init__(
        self,
        dataset_properties: dict,
        output_folder,
        output_patch_size,
        info,
        patch_overlap=(0, 0, 0),
        expand_by=None,
        mode:str='fg'
    ):
        """
        generates function from 'fg' bbox associated with the given case
        expand_by is specified in mm, i.e., 30 = 30mm.
        spacings are essential to compute number of array elements to add in case expand_by is required. Default: None
        """
        assert mode in ['fg','fgbg'], "Labels should be fg or fgbg"
        if mode == 'fgbg':
            assert expand_by in [0,None], "Cannot expand bbox if entire image is being patched"
        store_attr("output_folder,output_patch_size,info,patch_overlap")
        self.output_masks_folder = output_folder / ("lms")
        self.output_imgs_folder = output_folder / ("images")
        self.lm_fn = info["filename"]
        self.img_fn = Path(str(self.lm_fn).replace("lms", "images"))
        self.assimilate_dict(dataset_properties)
        bbs = info["bbox_stats"]

        b = [b for b in bbs if b["label"] == "all_fg"][0]

        if mode=='fg':
            self.bboxes = b["bounding_boxes"][1:]
        else:
            self.bboxes = b["bounding_boxes"][0]
            self.bboxes = [self.bboxes]
        if expand_by:
            self.add_to_bbox = [int(expand_by / sp) for sp in self.dataset_spacing]
        else:
            self.add_to_bbox = [
                0.0,
            ] * 3

    def load_img_lm_padding(self):
        lm = torch.load(self.lm_fn)
        img = torch.load(self.img_fn)
        self.img, self.lm, self.padding = PadDeficitImgMask(
            patch_size=self.output_patch_size,
            input_dims=3,
            pad_values=[self.dataset_min, 0],
            return_padding_array=True,
        ).encodes([img, lm])
        self.shift_bboxes_by = list(self.padding[::2])
        self.shift_bboxes_by.reverse()

    def maybe_expand_bbox(self, bbox):
        bbox_new = []
        for s, shift, ps, imsize, exp_by in zip(
            bbox,
            self.shift_bboxes_by,
            self.output_patch_size,
            self.img.shape,
            self.add_to_bbox,
        ):
            s = slice(
                int(np.maximum(0, s.start + shift - exp_by)),
                int(np.minimum(imsize, s.stop + shift + exp_by)),
            )
            sz = int(s.stop - s.start)
            ps_larger_by = np.maximum(0, ps - sz)
            start_tentative = int(s.start - np.ceil(ps_larger_by / 2))
            stop_tentative = int(s.stop + np.floor(ps_larger_by / 2))
            shift_back = np.minimum(0, imsize - stop_tentative)
            shift_forward = abs(np.minimum(0, start_tentative))
            shift_final = shift_forward + shift_back
            start_new = start_tentative + shift_final
            stop_new = stop_tentative + shift_final
            s_new = slice(start_new, stop_new, None)
            bbox_new.append(s_new)
        return bbox_new

    def create_grid_sampler_from_patchsize(self, bbox_final):
        img_f = self.img[tuple(bbox_final)].unsqueeze(0)
        lm_f = self.lm[tuple(bbox_final)].unsqueeze(0)
        img_tio = tio.ScalarImage(tensor=img_f)
        lm_tio = tio.ScalarImage(tensor=lm_f)
        subject = tio.Subject(image=img_tio, lm=lm_tio)
        self.grid_sampler = tio.GridSampler(
            subject=subject,
            patch_size=self.output_patch_size,
            patch_overlap=self.patch_overlap,
        )

    def create_patches_from_grid_sampler(self):
        for i, a in enumerate(self.grid_sampler):
            nm = strip_extension(self.lm_fn.name)
            out_fname = nm + "_" + str(i) + ".pt"
            out_mask_fname = self.output_masks_folder / out_fname
            out_img_fname = self.output_imgs_folder / out_fname
            print("Saving to files {0} and {1}".format(out_img_fname, out_mask_fname))
            img = a["image"][tio.DATA].squeeze(0)
            lm = a["lm"][tio.DATA].squeeze(0)
            img = img.contiguous()
            lm = lm.contiguous().to(torch.uint8)
            torch.save(img, out_img_fname)
            torch.save(lm, out_mask_fname)

    def create_patches_from_all_bboxes(self):
        self.load_img_lm_padding()
        for bbx in self.bboxes:
            bbox_new = self.maybe_expand_bbox(bbx)
            self.create_grid_sampler_from_patchsize(bbox_new)
            self.create_patches_from_grid_sampler()


def to_even(input_num, lower=True):
    np.fnc = np.subtract if lower == True else np.add
    output_num = np.fnc(input_num, input_num % 2)
    return int(output_num)


def patch_generator_wrapper(
    dataset_properties, output_folder, patch_size, info, patch_overlap, expand_by, mode
):
    P = PatchGenerator(
        dataset_properties, output_folder, patch_size, info, patch_overlap, expand_by, mode
    )
    P.create_patches_from_all_bboxes()
    return 1, info["filename"]


# %%
if __name__ == "__main__":

    from fran.utils.common import *

    P = Project(project_title="lidc2")
    P.maybe_store_projectwide_properties()
# %%
    lmg = "lm_group1"
    P.global_properties[lmg]

    imported_folder = Path("/s/fran_storage/predictions/totalseg/LITS-827/")
    TSL = TotalSegmenterLabels()
    imported_labelsets = TSL.labels("lung", "right"), TSL.labels("lung", "left")
    P.imported_labels(lmg, imported_folder, imported_labelsets)
    remapping = TSL.create_remapping(imported_labelsets, [8, 9])
# %%

    from fran.preprocessing.labelbounded import ImporterDataset

    I = ImporterDataset(
        expand_by=20,
        imported_folder=imported_folder,
        imported_labelsets=imported_labelsets,
        keep_imported_labels=False,
        remapping=remapping,
    )
    I.setup()
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
    imported_fn = find_matching_fn(fn, imported_folder.glob("*"))
    lm_imp = sitk.ReadImage(imported_fn)
    print(lm_imp.GetSpacing())
    dici = {"lm": lm_pt, "lm_imported": lm_imp, "remapping": remapping}

# %%
# %%
    fixed_spacing_folder = Path("/home/ub/datasets/preprocessed/lidc2/fixed_spacing/spc_080_080_150")
    patch_size= [192,192,120]
    patch_overlap = 0.2

    PG = PatchDataGenerator(P,fixed_spacing_folder, patch_size,patch_overlap,expand_by=10)
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
    dici = Re(dici)
    dici = E(dici)
# %%
    bb = dici["bounding_box"]
    print(bb)
# %%
    P.shift_bboxes_by,
    P.output_patch_size,
    P.img.shape,
    P.add_to_bbox,

# %%
