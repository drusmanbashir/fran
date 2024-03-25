# %%
import cc3d
import torchio as tio
from label_analysis.helpers import relabel
from label_analysis.merge import merge, merge_pt
from label_analysis.totalseg import TotalSegmenterLabels
from monai.data import Dataset, GridPatchDataset, PatchIter
from monai.transforms import Compose
from monai.transforms.croppad.dictionary import BoundingRectd
from monai.transforms.intensity.array import RandShiftIntensity
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from torch.utils.data import DataLoader

from fran.transforms.imageio import LoadSITKd, LoadTorchd
from fran.transforms.inferencetransforms import BBoxFromPTd
from fran.transforms.misc_transforms import (MergeLabelmapsd, Recast, RemapSITK,
                                             )
from fran.transforms.spatialtransforms import PadDeficitImgMask, ResizeDynamicd
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
from fastcore.basics import GetAttr, listify, store_attr

from fran.preprocessing.datasetanalyzers import bboxes_function_version
from fran.utils.fileio import *
from fran.utils.helpers import *
from fran.utils.imageviewers import *


class PatchGenerator(GetAttr):
    _default = "project"

    def __init__(
        self, project, fixed_spacing_folder, patch_size, patch_overlap, expand_by
    ):
        store_attr()
        if patch_overlap is None:
            self.patch_overlap = 0.2
        if expand_by is None:
            self.expand_by = 10
        fixed_sp_bboxes_fn = fixed_spacing_folder / ("bboxes_info")
        self.fixed_sp_bboxes = load_dict(fixed_sp_bboxes_fn)

        dataset_properties_fn = self.fixed_spacing_folderolder / (
            "resampled_dataset_properties.json"
        )
        assert (
            dataset_properties_fn.exists()
        ), "Dataset properties file does not exist. Has the Resampling been run to create folder {}?".format(
            self.fixed_spacing_folderolder
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
        self.existing_files = list((self.output_folder / ("masks")).glob("*pt"))
        self.existing_case_ids= [info_from_filename(f.name,full_caseid=True)['case_id'] for f in self.existing_files]
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
            ]
            for bb in self.fixed_sp_bboxes
        ]

        res = multiprocess_multiarg(
            patch_generator_wrapper, args, debug=debug, progress_bar=True
        )
        # for bb in pbar(self.fixed_sp_bboxes):
        #     P = PatchGeneratorSingle(self.dataset_properties, self.output_folder, self.patch_size,bb,patch_overlap=patch_overlap,expand_by=self.expand_by)
        #     P.create_patches_from_all_bboxes()

    def generate_bboxes(self, num_processes=24, debug=False):
        masks_folder = self.output_folder / "masks"
        mask_filenames = list(masks_folder.glob("*pt"))
        bg_label = 0
        arguments = [[x, bg_label] for x in mask_filenames]
        res_cropped = multiprocess_multiarg(
            func=bboxes_function_version,
            arguments=arguments,
            num_processes=num_processes,
            debug=debug,
        )

        stats_outfilename = (masks_folder.parent) / ("bboxes_info.json")
        print("Saving bbox stats to {}".format(stats_outfilename))
        save_dict(res_cropped, stats_outfilename)

    def save_patches_config(self):
        patches_config = {
            "patch_overlap": self.patch_overlap,
            "expand_by": self.expand_by,
        }
        save_dict(patches_config, self.patches_config_fn)

    def create_output_folders(self):
        maybe_makedirs(
            [self.output_folder / ("masks"), self.output_folder / ("images")]
        )

    @property
    def output_folder(self):
        patches_fldr_name = "dim_{0}_{1}_{2}".format(*self.patch_size)
        output_folder_ = (
            self.patches_folder
            / self.fixed_spacing_folderolder.name
            / patches_fldr_name
        )
        return output_folder_


class PatchGeneratorFG(DictToAttr):
    def __init__(
        self,
        dataset_properties: dict,
        output_folder,
        output_patch_size,
        info,
        patch_overlap=(0, 0, 0),
        expand_by=None,
    ):
        """
        generates function from 'all_fg' bbox associated wit the given case
        expand_by is specified in mm, i.e., 30 = 30mm.
        spacings are essential to compute number of array elements to add in case expand_by is required. Default: None
        """
        store_attr("output_folder,output_patch_size,info,patch_overlap")
        self.output_masks_folder = output_folder / ("masks")
        self.output_imgs_folder = output_folder / ("images")
        self.mask_fn = info["filename"]
        self.img_fn = Path(str(self.mask_fn).replace("masks", "images"))
        self.assimilate_dict(dataset_properties)
        bbs = info["bbox_stats"]

        b = [b for b in bbs if b["label"] == "all_fg"][0]
        self.bboxes = b["bounding_boxes"][1:]
        if expand_by:
            self.add_to_bbox = [int(expand_by / sp) for sp in self.dataset_spacings]
        else:
            self.add_to_bbox = [
                0.0,
            ] * 3

    def load_img_mask_padding(self):
        mask = torch.load(self.mask_fn)
        img = torch.load(self.img_fn)
        self.img, self.mask, self.padding = PadDeficitImgMask(
            patch_size=self.output_patch_size,
            input_dims=3,
            pad_values=[self.dataset_min, 0],
            return_padding_array=True,
        ).encodes([img, mask])
        self.shift_bboxes_by = list(self.padding[::2])
        self.shift_bboxes_by.reverse()

    def maybe_expand_bbox(self, bbox):
        bbox_new = []
        # for s,ps in zip(bbox,min_sizes):
        #     sz = int(s.stop-s.start)
        #     diff  = np.maximum(0,ps-sz*stride)
        #     start_new= int(s.start-np.ceil(diff/2))
        #     stop_new = int(s.stop+np.floor(diff/2))
        #     s_new = slice(start_new,stop_new,stride)
        #     bbox_new.append(s_new)
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
        mask_f = self.mask[tuple(bbox_final)].unsqueeze(0)
        img_tio = tio.ScalarImage(tensor=img_f)
        mask_tio = tio.ScalarImage(tensor=mask_f)
        subject = tio.Subject(image=img_tio, mask=mask_tio)
        self.grid_sampler = tio.GridSampler(
            subject=subject,
            patch_size=self.output_patch_size,
            patch_overlap=self.patch_overlap,
        )

    def create_patches_from_grid_sampler(self):
        for i, a in enumerate(self.grid_sampler):
            nm = strip_extension(self.mask_fn.name)
            out_fname = nm + "_" + str(i) + ".pt"
            out_mask_fname = self.output_masks_folder / out_fname
            out_img_fname = self.output_imgs_folder / out_fname
            print("Saving to files {0} and {1}".format(out_img_fname, out_mask_fname))
            img = a["image"][tio.DATA].squeeze(0)
            mask = a["mask"][tio.DATA].squeeze(0)
            torch.save(img, out_img_fname)
            torch.save(mask, out_mask_fname)

    def create_patches_from_all_bboxes(self):
        self.load_img_mask_padding()
        for bbx in self.bboxes:
            bbox_new = self.maybe_expand_bbox(bbx)
            self.create_grid_sampler_from_patchsize(bbox_new)
            self.create_patches_from_grid_sampler()


def to_even(input_num, lower=True):
    np.fnc = np.subtract if lower == True else np.add
    output_num = np.fnc(input_num, input_num % 2)
    return int(output_num)


def patch_generator_wrapper(
    dataset_properties, output_folder, patch_size, info, patch_overlap, expand_by=None
):
    P = PatchGeneratorFG(
        dataset_properties, output_folder, patch_size, info, patch_overlap, expand_by
    )
    P.create_patches_from_all_bboxes()
    return 1, info["filename"]



# %%
if __name__ == "__main__":

    from fran.utils.common import *

    P = Project(project_title="tmp")
    P.maybe_store_projectwide_properties()
# %%
    lmg = "lm_group2"
    P.global_properties[lmg]

    imported_folder = Path("/s/fran_storage/predictions/totalseg/LITS-827/")
    TSL = TotalSegmenterLabels()
    imported_labelsets = TSL.labels("lung", "right"), TSL.labels("lung", "left")
    P.imported_labels(lmg, imported_folder, imported_labelsets)
    remapping = TSL.create_remapping(imported_labelsets, [8, 9])
# %%

    I = ImporterDataset(
        project=P,
        expand_by=20,
        spacing=[0.8, 0.8, 1.5],
        lm_group="lm_group2",
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
    fixed_folder = P.fixed_spacing_folder / ("spc_080_080_150/masks")
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
    dici = {"mask": lm_pt, "mask_imported": lm_imp, "remapping": remapping}

# %%
# %%
    PT = PatchImporterDataset(P, spacing, 20)
# %%
    dici = C(dici)
    mask_out = dici["mask"][dici["mask_imported"].meta["bounding_box"]]
    ImageMaskViewer(
        [mask_out[0], dici["mask_imported"][0]], data_types=["mask", "mask"]
    )
    view_sitk(dici["mask_imported"], lm_imp, data_types=["mask", "mask"])
    config = {
        "spacing": PI.spacing,
        "expand_by": PI.expand_by,
        "imported_folder": PI.imported_folder,
        "imported_labelsets": PI.imported_labelsets,
    }

# %%

    mask2 = merge_pt(dici["mask_imported"][0], dici["mask"])[0]
    ImageMaskViewer(
        [mask_out[0], dici["mask_imported"][0]], data_types=["mask", "mask"]
    )
    lm = sitk.ReadImage(fn)
    lm_out = merge(lm_imp, lm)

    view_sitk(lm, lm_out, data_types=["mask", "mask"])
# %%
    P = PatchGeneratorFG(
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

    dici =I.data[0]
    dici = C(dici)
    print(dici['bounding_box'])
# %%
    dici = R(dici)
    dici = L1(dici)
    dici = L2(dici)
    dici = Re(dici)
    dici = E(dici)
# %%
    bb = dici['bounding_box']
    print(bb)

