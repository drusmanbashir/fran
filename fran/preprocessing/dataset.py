# %%
import SimpleITK as sitk
import torch
from fastcore.all import Union,  store_attr
from fastcore.foundation import GetAttr
from monai.data.dataloader import DataLoader
from monai.transforms.compose import Compose
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import (EnsureChannelFirstd, EnsureTyped,
                                                 FgBgToIndicesd, ToDeviced)

from monai.data import Dataset
from fran.data.dataloader import img_lm_metadata_lists_collated
from fran.data.dataset import NormaliseClipd
from fran.managers.datasource import get_ds_remapping
from fran.preprocessing.datasetanalyzers import bboxes_function_version
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import ChangeDType
from fran.transforms.misc_transforms import (ChangeDtyped, DictToMeta, FgBgToIndicesd2, HalfPrecisiond, LabelRemapd,
                                             Recastd, LabelRemapSITKd)
from fran.transforms.spatialtransforms import ResizeToTensord


from monai.transforms.croppad.dictionary import CropForegroundd
from monai.data import Dataset
from monai.transforms import Compose
from monai.transforms.utility.dictionary import EnsureChannelFirstd

from fran.transforms.imageio import LoadSITKd, LoadTorchd
from fran.transforms.inferencetransforms import BBoxFromPTd
from fran.transforms.misc_transforms import (ApplyBBox, MergeLabelmapsd,
                                             Recastd, LabelRemapSITKd)
from fran.transforms.spatialtransforms import ResizeToTensord
from fran.utils.string import info_from_filename

if "get_ipython" in globals():
    print("setting autoreload")
    from IPython import get_ipython

    ipython = get_ipython()
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
from pathlib import Path
import SimpleITK as sitk
from fastcore.basics import GetAttr, store_attr

from fran.utils.fileio import *
from fran.utils.helpers import *
from fran.utils.imageviewers import *


class ResamplerDataset(GetAttr, Dataset):
    _default = "project"

    def __init__(
        self,
        project,
        df,
        spacing,
        half_precision=False,
        clip_center=False,
        store_label_inds=False,
        mean_std_mode: str = "dataset",
        device="cuda",
    ):

        assert mean_std_mode in [
            "dataset",
            "fg",
        ], "Select either dataset mean/std or fg mean/std for normalization"
        self.project = project
        self.df = df
        self.spacing = spacing
        self.half_precision = half_precision
        self.clip_center = clip_center
        self.device = device
        super(GetAttr).__init__()
        self.set_normalization_values(mean_std_mode)

    def setup(self):
        self.create_transforms()
        data = self.create_data_dicts()
        Dataset.__init__(self, data,self.transform)


    #
    # def __getitem__(self, index):
    #     dici = self.data[index]
    #     img_fname = dici["image"]
    #     mask_fname = dici["lm"]
    #     remapping = dici['remapping']
    #     img = sitk.ReadImage(img_fname)
    #     mask = sitk.ReadImage(mask_fname)
    #     dici = {
    #         "image": img,
    #         "lm": lm,
    #         "remapping": remapping,
    #     }
    #     dici = self.transform(dici)
    #     return dici
    #
    def create_data_dicts(self, overwrite=False):
        data = []
        for index in range(len(self.df)):
            cp = self.df.iloc[index]
            ds = cp["ds"]
            remapping = get_ds_remapping(ds, self.global_properties)
            img_fname = cp["image"]
            mask_fname = cp["lm"]
            dici = {
                "image": img_fname,
                "lm": mask_fname,
                "remapping": remapping,
            }
            data.append(dici)
        return data


    def create_transforms(self):
        L = LoadSITKd(keys=["image", "lm"], image_only=True)
        R = LabelRemapd(keys=["lm"], remapping_key="remapping")
        T = ToDeviced(keys=["image", "lm"], device=self.device)
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
        Si = Spacingd(keys=["image"], pixdim=self.spacing, mode="trilinear")
        Rz = ResizeToTensord(keys=["lm"], key_template_tensor="image", mode="nearest")

        # Sm = Spacingd(keys=["lm"], pixdim=self.spacing,mode="nearest")
        N = NormaliseClipd(
            keys=["image"],
            clip_range=self.global_properties["intensity_clip_range"],
            mean=self.mean,
            std=self.std,
        )
        Ch = ChangeDtyped(keys=['lm'],target_dtype = torch.uint8)

        # tfms = [R, L, T, Re, Ind, Ai, Am, E, Si, Rz,Ch]
        tfms = [L, R, T, Re, Ind, E, Si, Rz,Ch]

        if self.clip_center == True:
            tfms.extend([N])
        if self.half_precision == True:
            H = HalfPrecisiond(keys=["image"])
            tfms.extend([H])
        self.transform = Compose(tfms)

    def set_normalization_values(self, mean_std_mode):
        if mean_std_mode == "dataset":
            self.mean = self.global_properties["mean_dataset_clipped"]
            self.std = self.global_properties["std_dataset_clipped"]
        else:
            self.mean = self.global_properties["mean_fg"]
            self.std = self.global_properties["std_fg"]



class CropToLabelDataset(Dataset):
    def __init__(
        self,
        expand_by,
        case_ids,
        data_folder,
        spacing,
        mask_label,
        fg_indices_exclude=None,
        device='gpu'
    ):
        """
        mask_label: label used to crop. 
        fg_indices_exclude: list of labels which will not count as fg in random sampling during training.
        """
        store_attr('expand_by,spacing,case_ids,data_folder,mask_label, device,fg_indices_exclude') # wont work with Datasetparent otherwise

    def setup(self):
        self.create_transforms()
        data = self.create_data_dicts()
        Dataset.__init__(self, data=data, transform=self.transform)

    def create_data_dicts(self ):
        masks_folder = self.data_folder / "lms"
        images_folder = self.data_folder / "images"
        lm_fns = list(masks_folder.glob("*.pt"))
        img_fns = list(images_folder.glob("*.pt"))
        data = []
        for cid in self.case_ids:
            lm_fn = self.case_id_file_match(cid, lm_fns)
            img_fn = self.case_id_file_match(cid, img_fns)
            dici = {
                "lm": lm_fn,
                "image": img_fn,
            }
            data.append(dici)
        return data

    def case_id_file_match(self, case_id, fileslist):
        fns = [fn for fn in fileslist if case_id == info_from_filename(fn.name,full_caseid=True)['case_id']]
        if len(fns) != 1:
            tr()
        return fns[0]

    def create_transforms(self):
        L2 = LoadTorchd(keys=["lm", "image"])
        # En = EnsureTyped(keys = ["lm","image"])
        D = ToDeviced(device =self.device,keys=["lm","image"])

        E = EnsureChannelFirstd(
            keys=[ "image", "lm"], channel_dim="no_channel")
        
        margin= [int(self.expand_by / sp) for sp in self.spacing]
        C = CropForegroundd(keys = ["image","lm"], source_key = "lm", select_fn = lambda lm: lm==self.mask_label,margin = margin)
        Ind = FgBgToIndicesd2(keys=["lm"], image_key="image", ignore_labels = self.fg_indices_exclude,image_threshold=-2600)
        Am = DictToMeta(
            keys=["lm"],
            meta_keys=[ "lm_fg_indices", "lm_bg_indices"],
            renamed_keys=[ "lm_fg_indices", "lm_bg_indices"],
        )


        tfms = [L2,D,E,C,Ind]
        C = Compose(tfms)
        self.transform = C

class FGBGIndicesDataset(CropToLabelDataset):

    '''
    This dataset will only load labelmaps, retrieve indices, and output them.
    '''

    def __init__(self,  case_ids, data_folder, fg_indices_exclude=None, device='gpu'):
        store_attr('case_ids,data_folder,fg_indices_exclude, device')

    def create_transforms(self):
        L2 = LoadTorchd(keys=["lm", "image"])
        # En = EnsureTyped(keys = ["lm","image"])
        D = ToDeviced(device =self.device,keys=["lm","image"])

        E = EnsureChannelFirstd(
            keys=[ "image", "lm"], channel_dim="no_channel")
        
        Ind = FgBgToIndicesd2(keys=["lm"], image_key="image", ignore_labels = self.fg_indices_exclude,image_threshold=-2600)
       
        tfms = [L2,D,E,Ind]
        C = Compose(tfms)
        self.transform = C
    

class ImporterDataset(Dataset):

    def __init__(
        self,
        expand_by,
        case_ids,
        data_folder,
        spacing,
        imported_folder,
        remapping,
    ):
        """
        imported_folder: Folder containing sitk Labelmaps
        keep_imported_labels: bool If True, imported labels are incorporated into the generated images and may be used in training.
        """
        store_attr("expand_by,spacing,case_ids,data_folder,imported_folder,remapping")

    def setup(self):
        self.create_transforms()
        data = self.create_data_dicts(self.imported_folder, self.remapping)
        Dataset.__init__(self, data=data, transform=self.transform)

    def create_data_dicts(self, imported_folder, remapping):
        masks_folder = self.data_folder / "lms"
        images_folder = self.data_folder / "images"
        lm_fns = list(masks_folder.glob("*.pt"))
        img_fns = list(images_folder.glob("*.pt"))
        imported_files = list(imported_folder.glob("*"))
        data = []
        for cid in self.case_ids:
            lm_fn = self.case_id_file_match(cid, lm_fns)
            img_fn = self.case_id_file_match(cid, img_fns)
            imported_fn = self.case_id_file_match(cid, imported_files)
            dici = {
                "lm": lm_fn,
                "image": img_fn,
                "lm_imported": imported_fn,
                "remapping": remapping,
            }
            data.append(dici)
        return data

    def case_id_file_match(self, case_id, fileslist):
        fns = [fn for fn in fileslist if case_id in fn.name]
        if len(fns) != 1:
            tr()
        return fns[0]

    def create_transforms(self):

        R = LabelRemapSITKd(keys=["lm_imported"], remapping_key="remapping")
        L1 = LoadSITKd(keys=["lm_imported"], image_only=True)
        L2 = LoadTorchd(keys=["lm", "image"])
        Re = Recastd(keys=["lm_imported"])

        E = EnsureChannelFirstd(
            keys=["lm_imported", "image", "lm"], channel_dim="no_channel")
        
        Rz = ResizeToTensord(
            keys=["lm_imported"], key_template_tensor="lm", mode="nearest"
        )
        M = MergeLabelmapsd(keys=["lm_imported", "lm"], key_output="lm_out")
        B = BBoxFromPTd(
            keys=["lm_imported"], spacing=self.spacing, expand_by=self.expand_by
        )
        A = ApplyBBox(keys=["lm", "image", "lm_out"], bbox_key="bounding_box")
        tfms = [R, L1, L2, Re, E, Rz, M, B, A]
        C = Compose(tfms)
        self.transform = C

# %%
if __name__ == "__main__":

    pass
    
