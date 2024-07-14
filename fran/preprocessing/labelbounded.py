# j %%
from label_analysis.totalseg import TotalSegmenterLabels
from monai.transforms import Compose
from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToDeviced
from torch.utils.data import DataLoader

from fran.data.dataloader import dict_list_collated
from fran.preprocessing.dataset import (
    CropToLabelDataset,
    FGBGIndicesDataset,
    ImporterDataset,
)
from fran.preprocessing.fixed_spacing import (
    generate_bboxes_from_lms_folder,
)
from fran.preprocessing.patch import PatchDataGenerator
from fran.transforms.imageio import LoadTorchd
from fran.utils.config_parsers import is_excel_nan
from fran.utils.string import info_from_filename

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
from fastcore.basics import GetAttr, store_attr

from fran.preprocessing.datasetanalyzers import bboxes_function_version
from fran.preprocessing.fixed_spacing import _Preprocessor
from fran.utils.fileio import *
from fran.utils.helpers import *
from fran.utils.imageviewers import *


class LabelBoundedDataGenerator(PatchDataGenerator, _Preprocessor, GetAttr):
    _default = "project"

    def __init__(
        self,
        project,
        expand_by,
        spacing,
        lm_group,
        mask_label=None,
        fg_indices_exclude: list = None,
    ) -> None:
        """
        mask_label: this label is used to apply mask, i.e., crop the image and lm
        """

        store_attr()  # WARN: leave this as top line otherwise getattr fails
        if is_excel_nan(self.lm_group):
            self.lm_group = "lm_group1"
        self.case_ids = self.get_case_ids_lm_group(self.lm_group)
        self.set_folders_from_spacing(self.spacing)
        print("Total case ids:", len(self.case_ids))
        if not self.mask_label:
            self.mask_label = 1

    def set_folders_from_spacing(self, spacing):
        self.fixed_spacing_subfolder = folder_name_from_list(
            prefix="spc",
            parent_folder=self.fixed_spacing_folder,
            values_list=spacing,
        )
        self.output_folder = folder_name_from_list(
            prefix="spc",
            parent_folder=self.lbd_folder,
            values_list=spacing,
        )

    def setup(self, device="cpu", batch_size=4, overwrite=True):
        device = resolve_device(device)
        print("Processing on ", device)
        self.register_existing_files()
        print("Overwrite:", overwrite)
        if overwrite == False:
            self.case_ids = self.remove_completed_cases()

        self.ds = CropToLabelDataset(
            case_ids=self.case_ids,
            expand_by=self.expand_by,
            spacing=self.spacing,
            data_folder=self.fixed_spacing_subfolder,
            mask_label=self.mask_label,
            fg_indices_exclude=self.fg_indices_exclude,
            device=device,
        )
        self.ds.setup()
        self.create_dl(batch_size=batch_size, num_workers=1)

    def create_dl(self, num_workers=1, batch_size=4):
        # same function as labelbounded
        self.dl = DataLoader(
            dataset=self.ds,
            num_workers=4,
            collate_fn=dict_list_collated(
                keys=[
                    "image",
                    "lm",
                    "lm_fg_indices",
                    "lm_bg_indices",
                    "foreground_start_coord",
                    "foreground_end_coord",
                ]
            ),
            # collate_fn=img_lm_metadata_lists_collated,
            batch_size=batch_size,
            pin_memory=False,
        )

    def remove_completed_cases(self):
        case_ids = set(self.case_ids).difference(self.existing_case_ids)
        print("Remaining case ids to process:", len(case_ids))
        return case_ids

    def process_batch(self, batch):
        images, lms, fg_inds, bg_inds, foreground_start_coord, foreground_end_coord = (
            batch["image"],
            batch["lm"],
            batch["lm_fg_indices"],
            batch["lm_bg_indices"],
            batch["foreground_start_coord"],
            batch["foreground_end_coord"],
        )
        for (
            image,
            lm,
            fg_ind,
            bg_ind,
            foreground_start_coord,
            foreground_end_coord,
        ) in zip(
            images, lms, fg_inds, bg_inds, foreground_start_coord, foreground_end_coord
        ):
            assert image.shape == lm.shape, "mismatch in shape".format(
                image.shape, lm.shape
            )
            assert image.dim() == 4, "images should be cxhxwxd"
            inds = {
                "lm_fg_indices": fg_ind,
                "lm_bg_indices": bg_ind,
                "foreground_start_coord": foreground_start_coord,
                "foreground_end_coord": foreground_end_coord,
                "meta": image.meta,
            }
            self.save_indices(inds, self.indices_subfolder)
            self.save_pt(image[0], "images")
            self.save_pt(lm[0], "lms")
            self.extract_image_props(image)

    def make_contiguous(self, batch):
        for key, listvals in batch.items():
            if isinstance(listvals[0], torch.Tensor):
                listvals = [val.contiguous() for val in listvals]
            batch[key] = listvals
        return batch

    def create_info_dict(self):
        resampled_dataset_properties = super().create_info_dict()
        resampled_dataset_properties["fg_indices_exclude"] = self.fg_indices_exclude
        resampled_dataset_properties["expand_by"] = self.expand_by
        resampled_dataset_properties["mask_label"] = self.mask_label
        resampled_dataset_properties["lm_group"] = self.lm_group
        return resampled_dataset_properties

    def get_case_ids_lm_group(self, lm_group):
        dsrcs = self.global_properties[lm_group]["ds"]
        cids = []
        for dsrc in dsrcs:
            cds = self.df["case_id"][self.df["ds"] == dsrc].to_list()
            cids.extend(cds)
        return cids

    @property
    def indices_subfolder(self):
        if self.fg_indices_exclude is not None:
            indices_subfolder = "indices_fg_exclude_{}".format(
                "".join([str(x) for x in self.fg_indices_exclude])
            )
        else:
            indices_subfolder = "indices"
        return indices_subfolder

    # def create_output_folders(self):
    #     super().create_output_folders()


class FGBGIndicesGenerator(LabelBoundedDataGenerator):
    """
    Outputs FGBGIndices only. No images of lms are created.
    Use this generator when LBD images and lms are already created, but a new set of FG indices is required, for example with exclusion of a new label

    """

    def __init__(self, project, data_folder, fg_indices_exclude: list = None) -> None:
        data_folder = Path(data_folder)
        self.output_folder = Path(data_folder)
        store_attr()

    def register_existing_files(self):
        self.existing_files = list(
            (self.output_folder / self.indices_subfolder).glob("*pt")
        )
        self.existing_case_ids = [
            info_from_filename(f.name, full_caseid=True)["case_id"]
            for f in self.existing_files
        ]
        self.existing_case_ids = set(self.existing_case_ids)
        print("Case ids processed in a previous session: ", len(self.existing_case_ids))

    def setup(self, device="cpu", batch_size=4, overwrite=False):
        device = resolve_device(device)
        print("Processing on ", device)
        self.register_existing_files()
        print("Overwrite:", overwrite)
        if overwrite == False:
            self.case_ids = self.remove_completed_cases()

        self.ds = FGBGIndicesDataset(
            case_ids=self.case_ids,
            data_folder=self.data_folder,
            fg_indices_exclude=self.fg_indices_exclude,
            device=device,
        )
        self.ds.setup()
        self.create_dl(batch_size=batch_size, num_workers=1)

    def create_dl(
        self, num_workers=1, batch_size=4
    ):  # optimised defaults. Do not change. GPU wont work (multiprocessing issues)
        # 'gpu' wont work on multiprocessing
        self.dl = DataLoader(
            dataset=self.ds,
            num_workers=num_workers,
            collate_fn=dict_list_collated(
                keys=[
                    "image",
                    "lm",
                    "lm_fg_indices",
                    "lm_bg_indices",
                    # "foreground_start_coord",
                    # "foreground_end_coord",
                ]
            ),
            batch_size=batch_size,
            pin_memory=False,
        )

    def process(
        self,
    ):
        if not hasattr(self, "dl"):
            print("No data loader created. No data to be processed. Create dl first")
            return 0
        print(
            "Retrieving FGBG indices from datafolder {0} and storing to subfolder {1}".format(
                self.data_folder, self.indices_subfolder
            )
        )
        self.create_output_folders()
        self.results = []
        self.shapes = []

        for batch in pbar(self.dl):
            self.process_batch(batch)
        # if self.results.shape[-1] == 3:  # only store if entire dset is processed
        #     self._store_dataset_properties()
        #     generate_bboxes_from_lms_folder(self.output_folder / ("lms"))
        # else:
        #     print(
        #         "since some files skipped, dataset stats are not being stored. run resampledatasetniftitotorch.get_tensor_folder_stats separately"
        #     )
        #

    def process_batch(self, batch):
        batch = self.make_contiguous(batch)
        (
            images,
            lms,
            fg_inds,
            bg_inds,
        ) = (
            batch["image"],
            batch["lm"],
            batch["lm_fg_indices"],
            batch["lm_bg_indices"],
            # batch["foreground_start_coord"],
            # batch["foreground_end_coord"],
        )
        for (
            image,
            lm,
            fg_ind,
            bg_ind,
            # foreground_start_coord,
            # foreground_end_coord,
        ) in zip(
            images,
            lms,
            fg_inds,
            bg_inds,  # , foreground_start_coord, foreground_end_coord
        ):
            assert image.shape == lm.shape, "mismatch in shape".format(
                image.shape, lm.shape
            )
            assert image.dim() == 4, "images should be cxhxwxd"
            inds = {
                "lm_fg_indices": fg_ind,
                "lm_bg_indices": bg_ind,
                # "foreground_start_coord": foreground_start_coord,
                # "foreground_end_coord": foreground_end_coord,
                "meta": image.meta,
            }
            self.save_indices(inds, self.indices_subfolder)
            # self.save_pt(image[0], "images")
            # self.save_pt(lm[0], "lms")
            self.extract_image_props(image)


#
#
#    def create_properties_dict(self):
#        resampled_dataset_properties = super().create_properties_dict()
#        labels ={k[0]:k[1] for k in self.remapping.items() if self.remapping[k[0]] != 0}
#        additional_props = {
#            "imported_folder": str(self.imported_folder),
#            "imported_labels":labels,
#            "keep_imported_labels": self.keep_imported_labels,
#        }
#        return resampled_dataset_properties|additional_props
#


class LabelBoundedDataGeneratorImported(PatchDataGenerator, _Preprocessor, GetAttr):

    _default = "project"

    def __init__(
        self,
        project,
        expand_by,
        spacing,
        lm_group,
        imported_folder,
        imported_labelsets,
        remapping,
        keep_imported_labels=False,
    ) -> None:
        store_attr()
        self.case_ids = self.get_case_ids_lm_group(lm_group)
        configs = self.get_resampling_config(spacing)
        self.fixed_spacing_subfolder = Path(configs["resampling_output_folder"])
        self.output_folder = Path(configs["lbd_output_folder"])

    def get_resampling_config(self, spacing):
        resamping_config_fn = self.project.fixed_spacing_folder / (
            "resampling_configs.json"
        )
        resampling_configs = load_dict(resamping_config_fn)
        for config in resampling_configs:
            if spacing == config["spacing"]:
                return config
        raise ValueError(
            "No resampling config found for this spacing: {0}. \nAll configs are:\n{1}".format(
                spacing, resampling_configs
            )
        )

    def create_dl(self, batch_size=4):
        self.register_existing_files()
        self.ds = ImporterDataset(
            case_ids=self.case_ids,
            expand_by=self.expand_by,
            spacing=self.spacing,
            data_folder=self.fixed_spacing_subfolder,
            imported_folder=self.imported_folder,
            remapping=self.remapping,
        )
        self.ds.setup()
        self.dl = DataLoader(
            dataset=self.ds,
            num_workers=1,
            collate_fn=dict_list_collated(
                keys=["image", "lm", "lm_imported", "bounding_box"]
            ),
            batch_size=batch_size,
        )

    def get_case_ids_lm_group(self, lm_group):
        dsrcs = self.global_properties[lm_group]["ds"]
        cids = []
        for dsrc in dsrcs:
            cds = self.df["case_id"][self.df["ds"] == dsrc].to_list()
            cids.extend(cds)
        return cids

    def create_properties_dict(self):
        resampled_dataset_properties = super().create_properties_dict()
        labels = {
            k[0]: k[1] for k in self.remapping.items() if self.remapping[k[0]] != 0
        }
        additional_props = {
            "imported_folder": str(self.imported_folder),
            "imported_labels": labels,
            "keep_imported_labels": self.keep_imported_labels,
        }
        return resampled_dataset_properties | additional_props


# %%
if __name__ == "__main__":

    from fran.utils.common import *

    P = Project(project_title="nodes3")
    P.maybe_store_projectwide_properties()
    # %%
    F = FGBGIndicesGenerator(
        project=P, data_folder="/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/"
    )
    F.setup()
    # %%
    F.process()
    # %%
    L = LabelBoundedDataGenerator(
        project=P,
        expand_by=40,
        spacing=[0.8, 0.8, 1.5],
        lm_group="lm_group1",
        mask_label=1,
        fg_indices_exclude=None,
    )
    L.indices_subfolder
    # %%
    L.create_dl(overwrite=False, device="cpu", batch_size=4)
    L.process()
    # %%
    fldr = Path(
        "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/indices_fg_exclude_1"
    )
    fns = list(fldr.glob("*pt"))
    # %%
    zeros = 0
    for fn in pbar(fns):
        tnser = torch.load(fn)
        lm_fg = tnser["lm_fg_indices"]
        if len(lm_fg) == 0:
            zeros += 1
        # %%

        if not "lm_fg_indices" in tnser.keys():
            tnser["lm_fg_indices"] = tnser["lm_fg_indicesmask_label"]

        keys = (
            "lm_fg_indices",
            "lm_bg_indices",
            "foreground_start_coord",
            "foreground_end_coord",
            "meta",
        )
        tnsr_neo = {k: tnser[k] for k in keys}
        tnsr_neo["lm_fg_indices"] = tnsr_neo["lm_fg_indices"].contiguous()
        tnsr_neo["lm_bg_indices"] = tnsr_neo["lm_fg_indices"].contiguous()
        torch.save(tnsr_neo, fn)
    # %%

    # %%
    # %%

    fn = "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_150/indices_fg_exclude_1/drli_002.pt"
    # %%
    lmg = "lm_group1"
    P.global_properties[lmg]

    imported_folder = Path("/s/fran_storage/predictions/totalseg/LITS-827/")
    TSL = TotalSegmenterLabels()
    imported_labelsets = TSL.labels("lung", "right"), TSL.labels("lung", "left")
    remapping = TSL.create_remapping(imported_labelsets, [8, 9])
    P.imported_labels(lmg, imported_folder, imported_labelsets)

    # %%
    L = LabelBoundedDataGenerator(
        project=P,
        expand_by=40,
        spacing=[0.8, 0.8, 1.5],
        lm_group="lm_group1",
        mask_label=1,
    )

    L.create_dl()
    L.process()

    # %%

    for batch in pbar(L.dl):
        images, lms = batch["image"], batch["lm"]
        print(images.shape)
    ImageMaskViewer([images[0][0], lms[0][0]])
    # %%
    L = LabelBoundedDataGenerator(
        project=P,
        expand_by=20,
        spacing=[0.8, 0.8, 1.5],
        lm_group="lm_group1",
        imported_folder=imported_folder,
        imported_labelsets=imported_labelsets,
        keep_imported_labels=False,
        remapping=remapping,
    )

    L.create_dl()
    L.process()

    # %%
    bbfn = "/home/ub/datasets/preprocessed/tmp/lbd/spc_080_080_150/bboxes_info.pkl"
    dic = load_dict(bbfn)
    generate_bboxes_from_lms_folder(
        Path("/r/datasets/preprocessed/lidc2/lbd/spc_080_080_150/lms/")
    )
    # %%
    spacing = [0.8, 0.8, 1.5]
    # %%
    L.expand_by = 50
    L.device = "cpu"

    L2 = LoadTorchd(keys=["lm", "image"])
    # En = EnsureTyped(keys = ["lm","image"])
    D = ToDeviced(device=L.device, keys=["lm", "image"])

    E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")

    margin = [int(L.expand_by / sp) for sp in L.spacing]
    margin2 = [int(L.expand_by / sp) * 2 for sp in L.spacing]
    # %%
    Cr1 = CropForegroundd(
        keys=["image", "lm"],
        source_key="lm",
        select_fn=lambda lm: lm == L.mask_label,
        margin=margin,
    )
    Cr2 = CropForegroundd(
        keys=["image", "lm"],
        source_key="lm",
        select_fn=lambda lm: lm == L.mask_label,
        margin=margin2,
    )
    tfms = [L2, D, E, Cr1]
    C1 = Compose(tfms)

    tfms2 = [L2, D, E, Cr2]
    C2 = Compose(tfms)
    # %%
    dici = L.ds.data[1].copy()
    dici1 = C1(dici)
    img1 = dici1["image"][0]
    lm1 = dici1["lm"][0]
    ImageMaskViewer([img1, lm1])
    # %%

    dici2 = L.ds.data[1].copy()
    dici2 = C2(dici2)
    img2 = dici2["image"][0]
    lm2 = dici2["lm"][0]
    ImageMaskViewer([img2, lm2])
# %%
