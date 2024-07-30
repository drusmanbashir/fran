# %%
# %%
import torch
from fastcore.all import Union, store_attr
from fastcore.foundation import GetAttr
from fran.preprocessing.datasetanalyzers import bboxes_function_version
from fran.transforms.misc_transforms import FgBgToIndicesd2
from fran.utils.string import info_from_filename
from pathlib import Path
import SimpleITK as sitk
from fastcore.basics import GetAttr, store_attr

from fran.utils.fileio import *
from fran.utils.helpers import *
from fran.utils.imageviewers import *
from label_analysis.totalseg import TotalSegmenterLabels
import torch
from fran.utils.dictopts import DictToAttr
import torchio as tio
from fran.transforms.spatialtransforms import PadDeficitImgMask
from fran.utils.fileio import load_dict, maybe_makedirs, save_dict
from fran.utils.helpers import multiprocess_multiarg
from fran.utils.string import info_from_filename, strip_extension

<<<<<<< HEAD
# %%
if "get_ipython" in globals():
    print("setting autoreload")
    from IPython import get_ipython

    ipython = get_ipython()
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
=======
>>>>>>> efc2e4fb (jj)
from pathlib import Path

import ipdb
import numpy as np
import SimpleITK as sitk
from fastcore.basics import GetAttr, store_attr

import ipdb

tr = ipdb.set_trace

from fran.preprocessing.datasetanalyzers import bboxes_function_version
<<<<<<< HEAD
# %%
=======
from fran.preprocessing.fixed_spacing import _Preprocessor
>>>>>>> efc2e4fb (jj)


class PatchGenerator(DictToAttr,_Preprocessor):
    def __init__(
        self,
        dataset_properties: dict,
        output_folder,
        output_patch_size,
        info,
        patch_overlap=(0, 0, 0),
        expand_by=None,
        mode: str = "fg",
    ):
        """
        generates function from 'fg' bbox associated with the given case
        expand_by is specified in mm, i.e., 30 = 30mm.
        spacings are essential to compute number of array elements to add in case expand_by is required. Default: None
        """
        assert mode in ["fg", "fgbg"], "Labels should be fg or fgbg"
        if mode == "fgbg":
            assert expand_by in [
                0,
                None,
            ], "Cannot expand bbox if entire image is being patched"
        store_attr("output_folder,output_patch_size,info,patch_overlap")
        self.lm_fn = info["filename"]
        self.img_fn = Path(str(self.lm_fn).replace("lms", "images"))
        self.assimilate_dict(dataset_properties)
        bbs = info["bbox_stats"]

        b = [b for b in bbs if b["label"] == "all_fg"][0]

        if mode == "fg":
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
        print("Processing filepair : ", self.lm_fn)
        print("Number of patches: ", len(self.grid_sampler))
        for i, a in enumerate(self.grid_sampler):
            img = a["image"][tio.DATA]#.squeeze(0)
            lm = a["lm"][tio.DATA]#.squeeze(0)
            Ind = FgBgToIndicesd2(keys=['lm'], image_key="image", image_threshold=-2600)
            dici = {'image':img,'lm':lm}
            dici = Ind(dici)

            fg_ind = dici['lm_fg_indices']
            bg_ind = dici['lm_bg_indices']
            image= dici['image']
            lm = dici['lm']
            inds = {
                "lm_fg_indices": fg_ind,
                "lm_bg_indices": bg_ind,
                "meta": image.meta,
            }

            self.save_indices(inds, self.indices_subfolder,suffix=str(i))
            # self.save_in(inds, self.indices_subfolder,contiguous=False,suffix=str(i))
            self.save_pt(image[0], "images",suffix=str(i))
            self.save_pt(lm[0], "lms",suffix=str(i))

    def process(self):
        self.create_output_folders()
        self.create_patches_from_all_bboxes()

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
        dataset_properties,
        output_folder,
        patch_size,
        info,
        patch_overlap,
        expand_by,
        mode,
    )
    P.create_patches_from_all_bboxes()
    return 1, info["filename"]


def contains_bg_only(bbox_stats):
    all_fg_bbox = [bb for bb in bbox_stats if bb["label"] == "all_fg"][0]
    bboxes = all_fg_bbox["bounding_boxes"]
    if len(bboxes) == 1:
        return True
    elif bboxes[0] != bboxes[1]:
        return False
    else:
        tr()


class PatchDataGenerator(_Preprocessor):
    _default = "project"

    def __init__(
        self,
        project,
        data_folder,
        patch_size,
        patch_overlap=0.25,
        expand_by=None,
        mode="fgbg",
        output_suffix=None,
    ):
        store_attr()
        fixed_sp_bboxes_fn = data_folder / ("bboxes_info")
        self.fixed_sp_bboxes = load_dict(fixed_sp_bboxes_fn)

        dataset_properties_fn = self.data_folder / ("resampled_dataset_properties.json")
        assert (
            dataset_properties_fn.exists()
        ), "Dataset properties file does not exist. Has the Resampling been run to create folder {}?".format(
            self.data_folder
        )
        self.dataset_properties = load_dict(dataset_properties_fn)
        self.dataset_properties['data_folder'] = str(data_folder)

        self.patches_config_fn = self.output_folder / ("patches_config.json")

        if self.patches_config_fn.exists():
            print(
                "Patches configs already exist. Overriding given values with those from file"
            )
            patches_config = load_dict(self.patches_config_fn)
            print(patches_config)
            self.patch_overlap = patches_config["patch_overlap"]
            self.expand_by = patches_config["expand_by"]

    def setup(self, overwrite=False):
        self.register_existing_files()
        if overwrite == False:
            self.remove_completed_cases()
    def process(self, debug=False):
        self.create_output_folders()
        self.create_patches(debug)

    def register_existing_files(self):
        self.existing_files = list((self.output_folder / ("lms")).glob("*pt"))
        self.existing_case_ids = [
            info_from_filename(f.name, full_caseid=True)["case_id"]
            for f in self.existing_files
        ]
        self.existing_case_ids = set(self.existing_case_ids)
        print("Case ids processed in a previous session: ", len(self.existing_case_ids))

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

    def create_patches(self,  debug=False):
        patch_overlap = [int(self.patch_overlap * ps) for ps in self.patch_size]
        patch_overlap = [to_even(ol) for ol in patch_overlap]
        maybe_makedirs(self.output_folder)
        self.save_patches_config()
        args = [
            [
                self.dataset_properties,
                self.output_folder,
                self.patch_size,
                bb,
                patch_overlap,
                self.expand_by,
                self.mode,
            ]
            for bb in self.fixed_sp_bboxes
        ]

        res = multiprocess_multiarg(
            patch_generator_wrapper, args, debug=debug, progress_bar=True
        )
        self.generate_bboxes(debug=debug)

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
            "data_folder": str(self.data_folder),
            "patch_overlap": self.patch_overlap,
            "expand_by": self.expand_by,
        }
        save_dict(patches_config, self.patches_config_fn)


    @property
    def output_folder(self):
        data_folder_name = self.data_folder.name
        pat = re.compile("_plan\d+")
        data_folder_name = pat.sub("",data_folder_name)

        patches_fldr_name = "dim_{0}_{1}_{2}".format(*self.patch_size)
        if self.output_suffix:
            patches_fldr_name+="_"+self.output_suffix

        self._output_folder = (
            self.patches_folder / data_folder_name/ patches_fldr_name
        )
        return self._output_folder


# %%
if __name__ == "__main__":
# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR>

    from fran.utils.common import *
    from fran.preprocessing.labelbounded import ImporterDataset

    P = Project(project_title="litsmc")
    P.maybe_store_projectwide_properties()
    
# %%
#SECTION:-------------------- PATCHGENERATOR--------------------------------------------------------------------------------------

    conf = ConfigMaker(
        P, raytune=False
    ).config
    plan = conf['plan']
    plan_name = "plan"+str(conf['dataset_params']['plan'])

    source_plan_name = plan['source_plan']
    source_plan = conf[source_plan_name]
    spacing = ast.literal_eval(source_plan['spacing'])
    src_data_mode= source_plan['mode']
    P.lbd_folder
    patch_size = ast.literal_eval(plan['patch_size'])
    patch_overlap = plan['patch_overlap']

    deb=False
# %%
    data_folder  = folder_name_from_list(
            prefix="spc",
            parent_folder=P.lbd_folder,
            values_list=spacing,
        suffix=source_plan_name
        )



# %%
    PG = PatchDataGenerator(
        P, data_folder, patch_size=patch_size, patch_overlap=patch_overlap, expand_by=0,output_suffix=plan_name
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
#SECTION:-------------------- PATCHGENERATOR Single module--------------------------------------------------------------------------------------
# %%

    patch_overlap = .25
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
    info =argi[3]

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
#SECTION:-------------------- ROUGH--------------------------------------------------------------------------------------
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
        print("FG", len(inds['lm_fg_indices']))
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
    imported_fn = find_matching_fn(fn, imported_folder.glob("*"))
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

# %%
