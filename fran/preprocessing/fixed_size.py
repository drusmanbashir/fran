# %%
from pathlib import Path

from fran.preprocessing.preprocessor import Preprocessor
import ipdb
import torch
from fastcore.all import store_attr
from fastcore.basics import store_attr
from label_analysis.totalseg import TotalSegmenterLabels
from monai.transforms import Compose
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.spatial.dictionary import Resized
from monai.transforms.utility.dictionary import (DeleteItemsd,
                                                 EnsureChannelFirstd,
                                                 SqueezeDimd)
from utilz.fileio import *
from utilz.helpers import *
from utilz.imageviewers import *
from utilz.string import info_from_filename

from fran.configs.parser import ConfigMaker
from fran.preprocessing.patch import PatchDataGenerator
from fran.transforms.imageio import LoadTorchd, TorchWriter
from fran.transforms.misc_transforms import LabelRemapd
from fran.utils.folder_names import folder_names_from_plan

tr = ipdb.set_trace
import ray
import re


@ray.remote(num_cpus=1)
class FixedSizeMaker(object):
    """
    Used by 'whole' DataManager in training.
    """

    def __init__(self):
        pass

    def process(
        self,
        dicis,
        spatial_size,
        output_folder_im,
        output_folder_lm,
        remapping=None,
    ):

        L = LoadTorchd(keys=["lm", "image"])
        M = LabelRemapd(keys=["lm"], remapping_key="remapping")
        E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
        Rz = Resized(
            keys=["image", "lm"], spatial_size=spatial_size, mode=["linear", "nearest"]
        )
        S = SqueezeDimd(keys=["image", "lm"])

        Si = SaveImaged(
            keys=["image"],
            output_ext="pt",
            writer=TorchWriter,
            output_dir=output_folder_im,
            output_postfix="",
            output_dtype="float32",
            separate_folder=False,
        )
        Sl = SaveImaged(
            output_ext="pt",
            keys=["lm"],
            writer=TorchWriter,
            output_dtype="uint8",
            output_dir=output_folder_lm,
            output_postfix="",
            separate_folder=False,
        )
        Del = DeleteItemsd(keys=["image", "lm"])

        # S1 = SaveImage(output_ext='pt',  output_dir=self.output_fldr_imgs, output_postfix=str(1), output_dtype='float32', writer=TorchWriter,separate_folder=False)
        if remapping:
            tfms = Compose([L, M, E, Rz, S, Si, Sl, Del])
        else:
            tfms = Compose([L, E, Rz, S, Si, Sl, Del])
        for dici in dicis:
            try:
                dici = tfms(dici)
            except Exception as e:
                print("Exception")
                print(e)

        return 1


class FixedSizeDataGenerator(PatchDataGenerator):
    _default = "project"

    def __init__(self, project,plan, data_folder=None, output_folder =None ) -> None:
        #HACK: below is same as preprocessor init. Come back to this while u update PatchDataGenerator
        store_attr("project,plan,data_folder")
        self.data_folder = data_folder
        self.set_input_output_folders(data_folder, output_folder)

    def setup(self, overwrite=False):
        self.prepare_dicts()
        super().setup(overwrite=overwrite)

    def prepare_dicts(self):
        images_fldr = self.data_folder / ("images")
        lms_fldr = self.data_folder / ("lms")
        lms = list(lms_fldr.glob("*"))
        imgs = list(images_fldr.glob("*"))
        print("{0} images in data folder {1}".format(len(imgs), self.data_folder))

        self.dicis = []
        for img in imgs:
            case_id_info = info_from_filename(img.name, full_caseid=True)
            case_id = case_id_info["case_id"]
            lm_value = find_matching_fn(img, lms, ["case_id"])[0]
            # Create the dictionary for the current image
            dic = {"case_id": case_id, "image": img, "lm": lm_value, "remapping": self.plan["remapping_train"]}
            # Append the dictionary to the dicis list
            self.dicis.append(dic)

    def remove_completed_cases(self):
        dicis_out = []
        for dici in self.dicis:
            if dici["case_id"] not in self.existing_case_ids:
                dicis_out.append(dici)
        self.dicis = dicis_out
        print("Remaining cases: ", len(self.dicis))

    def process(self, num_processes=16):
        self.create_output_folders()
        self.create_tensors(num_processes=num_processes)

    def create_tensors(self, num_processes):
        dicis = list(chunks(self.dicis, num_processes))
        actors = [FixedSizeMaker.remote() for _ in range(num_processes)]
        if self.plan["remapping_train"] is not None:
            remapping =True
        else:
            remapping =False
        results = ray.get(
            [
                c.process.remote(
                    dicis,
                    self.plan["patch_size"],
                    self.output_folder / ("images"),
                    self.output_folder / ("lms"),
                    remapping=remapping
                )
                for c, dicis in zip(actors, dicis)
            ]
        )

    def set_input_output_folders(self, data_folder=None, output_folder=None):
        folders = folder_names_from_plan(self.project, self.plan)

        if data_folder is None:
            data_folder = folders["data_folder_source"]
        self.data_folder = Path(data_folder)
        if output_folder is None:
            whole_subfolder = folders[
                "data_folder_whole"
            ]
            self.output_folder =Path( whole_subfolder)
        else:
            self.output_folder = Path(output_folder)



    def create_output_folders(self):
        maybe_makedirs(
            [
                self.output_folder / ("lms"),
                self.output_folder / ("images"),
                # self.indices_subfolder,
            ]
        )
    #
    # def set_output_folder(self, parent_folder):
    #     output_folder = folder_name_from_list(
    #         prefix="sze",
    #         parent_folder=self.fixed_size_folder,
    #         values_list=self.spatial_size,
    #     )
    #
    #     self.output_folder = folder_name_from_list(
    #         prefix="spc",
    #         parent_folder=parent_folder,
    #         values_list=self.spacing,
    #     )
    #     if self.plan_name is not None:
    #         output_name = "_".join([self.output_folder.name, self.plan_name])
    #         self.output_folder = Path(self.output_folder.parent / output_name)
    #

# %%
if __name__ == "__main__":
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>
    from fran.managers import Project
    from fran.utils.common import *

    P = Project(project_title="totalseg")
    C = ConfigMaker(P)
    C.setup(2)
    conf = C.configs
    P.maybe_store_projectwide_properties()

# %%
    plan = conf["plan_train"]
# %%

    F = FixedSizeDataGenerator(
        project=P,
        plan =plan ,
        data_folder=None,
    )
# %%
    F.setup(overwrite=False)

# %%
    F.process()
# %%
    img_fn = "/r/datasets/preprocessed/totalseg/whole_images/096096096/images/totalseg_s0310.pt"
    lm_fn = "/r/datasets/preprocessed/totalseg/whole_images/096096096/lms/totalseg_s0310.pt"
    img = torch.load(img_fn, weights_only=False)
    lm = torch.load(lm_fn, weights_only=False)
    ImageMaskViewer([img,lm])
# %%

    output_folder = folder_name_from_list(
        prefix="sze", parent_folder=P.fixed_size_folder, values_list=[64, 64, 64]
    )
    output_folder_im = output_folder / "images"
    output_folder_lm = output_folder / "lms"
    maybe_makedirs([output_folder_im, output_folder_lm])

    images_fldr = self.data_folder / ("images")
    lms_fldr = self.data_folder / ("lms")

    lms = list(lms_fldr.glob("*"))
    imgs = list(images_fldr.glob("*"))
    pairs = [
        {"image": img, "lm": find_matching_fn(img, lms, "case_id")[0]} for img in imgs
    ]
# %%
    tots = len(pairs)
    n_proc = 32

# %%
    dicis = list(chunks(pairs, n_proc))
    actors = [FixedSizeMaker.remote() for _ in range(n_proc)]
# %%
    results = ray.get(
        [
            c.process.remote(
                dicis,
                spatial_size,
                output_folder_im,
                output_folder_lm,
                remapping_train=remapping_train,
            )
            for c, dicis in zip(actors, dicis)
        ]
    )
    # dici = L(dici)
# %%
    dici = tfms(dici)
# %%
    dici = L(dici)
    dici = M(dici)
# %%
    dici = E(dici)

    dici = Rz(dici)

    im, lm = dici["image"], dici["lm"]

# %%
    fn = "/s/fran_storage/datasets/preprocessed/fixed_size/litsmc/sze_64_64_64/images/totalseg_s0784.pt"
    fn2 = "/s/fran_storage/datasets/preprocessed/fixed_size/litsmc/sze_64_64_64/lms/totalseg_s0784.pt"
    trn = torch.load(fn, weights_only=False)
    t2 = torch.load(fn2, weights_only=False)
    ImageMaskViewer([trn, t2], dtypes="im")
# %%
# %%
# SECTION:-------------------- TROUBLE <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR> <CR>
# %%
    output_folder_lm = F.output_folder / "lms"
    output_folder_im = F.output_folder / "images"

# %%
    remapping_train= plan["remapping_train"]
    L = LoadTorchd(keys=["lm", "image"])
    M = LabelRemapd(keys=["lm"], remapping=remapping_train)
    E = EnsureChannelFirstd(keys=["image", "lm"], channel_dim="no_channel")
    Rz = Resized(
        keys=["image", "lm"], spatial_size=spatial_size, mode=["linear", "nearest"]
    )
    S = SqueezeDimd(keys=["image", "lm"])

    Si = SaveImaged(
        keys=["image"],
        output_ext="pt",
        writer=TorchWriter,
        output_dir=output_folder_im,
        output_postfix="",
        output_dtype="float32",
        separate_folder=False,
    )
    Sl = SaveImaged(
        output_ext="pt",
        keys=["lm"],
        writer=TorchWriter,
        output_dtype="uint8",
        output_dir=output_folder_lm,
        output_postfix="",
        separate_folder=False,
    )
    Del = DeleteItemsd(keys=["image", "lm"])

# %%
    dici = F.dicis[0]
    dici = L(dici)
    dici = M(dici)
    dici = E(dici)
    dici = Rz(dici)
    dici = S(dici)
    dici = Si(dici)
    dici = Sl(dici)
    dici = Del(dici)
# %%
    image = dici["image"]
    lm = dici["lm"]
# %%
    ImageMaskViewer([image, lm])
# %%
    # S1 = SaveImage(output_ext='pt',  output_dir=self.output_fldr_imgs, output_postfix=str(1), output_dtype='float32', writer=TorchWriter,separate_folder=False)
    if remapping_train:
        tfms = Compose([L, M, E, Rz, S, Si, Sl, Del])
    else:
        tfms = Compose([L, E, Rz, S, Si, Sl, Del])
    for dici in dicis:
        try:
            dici = tfms(dici)
            print("Saved {0}".format(dici["image"].meta["filename_or_obj"]))
        except Exception as e:
            print("Exception")
            print(e)

# %%
    fn = "/s/fran_storage/datasets/preprocessed/fixed_size/totalseg/sze_96_96_96/lms/totalseg_s0726.pt"
    fn = "/s/fran_storage/datasets/preprocessed/fixed_spacing/totalseg/spc_080_080_150/lms/totalseg_s0928.pt"
    fn = "/s/xnat_shadow/totalseg/lms/totalseg_s0928.nii.gz"
    fn = "/s/fran_storage/datasets/preprocessed/fixed_size/totalseg/sze_96_96_96/lms/totalseg_s0928.pt"
    import torch

    lm = torch.load(fn)
    lm.unique()
    lm[lm == 118] = 0
    torch.save(lm, fn)
    import SimpleITK as sitk

    lm = sitk.ReadImage(fn)
    from label_analysis.helpers import get_labels, relabel

    sitk.WriteImage(lm, fn)
    labs = get_labels(lm)
    118 in labs
    remapping = {118: 0}
    lm = relabel(lm, remapping)

# %%
