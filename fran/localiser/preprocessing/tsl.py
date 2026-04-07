# %%
from pathlib import Path

from monai.transforms import Transform
import ipdb
import torch
from fran.data.collate import as_is_collated
from fran.localiser.preprocessing.preprocess import Preprocessor2D
from fran.transforms.imageio import LoadSITKd, TorchWriter
from fran.transforms.misc_transforms import DictToMetad, MetaToDict
from fran.transforms.spatialtransforms import Project2D
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.transforms import Compose
from monai.transforms.croppad.dictionary import BoundingRectd
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.io.array import SaveImage
from monai.transforms.spatial.dictionary import Orientationd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToDeviceD
from tqdm.auto import tqdm
from utilz.fileio import is_sitk_file, maybe_makedirs
from utilz.helpers import find_matching_fn

tr = ipdb.set_trace


def tfms_from_dict(keys, transforms_dict):
    keys = keys.replace(" ", "").split(",")
    tfms = [transforms_dict[key] for key in keys]
    return Compose(tfms)


class Preprocessor2DTSL():
    def __init__(self, data_folder, output_fldr):
        self.data_folder = Path(data_folder)
        self.fldr_imgs = self.data_folder / "images"
        self.fldr_lms = self.data_folder / "lms"
        self.output_fldr = Path(output_fldr)
        self.output_fldr_imgs = self.output_fldr / ("images")
        self.output_fldr_lms = self.output_fldr / ("lms")
        self.tfms_keys = "L,Dev,E,O,Remap,N,P1,P2,P3"

    def setup(self, batch_size=8):
        imgs = self.fldr_imgs.glob("*")
        imgs = [img for img in imgs if is_sitk_file(img)]
        lms = list(self.fldr_lms.glob("*"))
        self.data = []
        for img in tqdm(imgs[:10]):
            lm = find_matching_fn(img, lms)[0]
            if lm:
                dici = {"image": img, "lm": lm}
                self.data.append(dici)

        self.create_transforms()
        tfms = tfms_from_dict(self.tfms_keys, self.transforms_dict)
        self.ds = Dataset(data=self.data, transform=tfms)
        self.create_dl(num_workers=batch_size * 2, batch_size=batch_size)

    def create_transforms(self):
        self.L = LoadSITKd(keys=["image", "lm"])
        self.O = Orientationd(keys=["image", "lm"], axcodes="RAS")
        self.Dev = ToDeviceD(keys=["image", "lm"], device="cuda")
        self.N = NormalizeIntensityd(["image"])
        self.E = EnsureChannelFirstd(["image", "lm"], channel_dim="no_channel")
        self.P1 = Project2D(
            keys=["lm", "image"],
            operations=["mean", "sum"],
            dim=1,
            output_keys=["lm1", "image1"],
        )
        self.P2 = Project2D(
            keys=["lm", "image"],
            operations=["sum", "mean"],
            dim=2,
            output_keys=["lm2", "image2"],
        )
        self.P3 = Project2D(
            keys=["lm", "image"],
            operations=["sum", "mean"],
            dim=3,
            output_keys=["lm3", "image3"],
        )
        self.BB1 = BoundingRectd(keys=["lm1"])
        self.BB2 = BoundingRectd(keys=["lm2"])
        self.BB3 = BoundingRectd(keys=["lm3"])
        self.M = MetaToDict(keys=["image"], meta_keys=["filename_or_obj"])
        self.D1 = DictToMetad(keys=["image1"], meta_keys=["lm1_bbox"])
        self.D2 = DictToMetad(keys=["image2"], meta_keys=["lm2_bbox"])
        self.D3 = DictToMetad(keys=["image3"], meta_keys=["lm3_bbox"])
        self.transforms_dict = {
            "Dev": self.Dev,
            "L": self.L,
            "O": self.O,
            "N": self.N,
            "E": self.E,
            "P1": self.P1,
            "P2": self.P2,
            "P3": self.P3,
            "BB1": self.BB1,
            "BB2": self.BB2,
            "BB3": self.BB3,
            "M": self.M,
            "D1": self.D1,
            "D2": self.D2,
            "D3": self.D3,
        }
        self.create_tsl_region_tfms()

    def create_tsl_region_tfms(self):
        self.Remap = MultiRemapsTSL(lm_key="lm")
        self.transforms_dict["Remap"] =self.Remap



    def create_dl(self, num_workers=4, batch_size=4):
        # same function as labelbounded
        self.dl = DataLoader(
            dataset=self.ds,
            num_workers=4,
            collate_fn=as_is_collated,
            batch_size=batch_size,
            pin_memory=False,
        )

    def process(self):
        maybe_makedirs([self.output_fldr_imgs, self.output_fldr_lms])

        S1 = SaveImage(
            output_ext="pt",
            output_dir=self.output_fldr_imgs,
            output_postfix=str(1),
            output_dtype="float32",
            writer=TorchWriter,
            separate_folder=False,
        )
        S2 = SaveImage(
            output_ext="pt",
            output_dir=self.output_fldr_imgs,
            output_postfix=str(2),
            output_dtype="float32",
            writer=TorchWriter,
            separate_folder=False,
        )
        S3 = SaveImage(
            output_ext="pt",
            output_dir=self.output_fldr_imgs,
            output_postfix=str(3),
            output_dtype="float32",
            writer=TorchWriter,
            separate_folder=False,
        )
        Sl1 = SaveImage(
            output_ext="pt",
            output_dir=self.output_fldr_lms,
            output_postfix=str(1),
            output_dtype="float32",
            writer=TorchWriter,
            separate_folder=False,
        )
        Sl2 = SaveImage(
            output_ext="pt",
            output_dir=self.output_fldr_lms,
            output_postfix=str(2),
            output_dtype="float32",
            writer=TorchWriter,
            separate_folder=False,
        )
        Sl3 = SaveImage(
            output_ext="pt",
            output_dir=self.output_fldr_lms,
            output_postfix=str(3),
            output_dtype="float32",
            writer=TorchWriter,
            separate_folder=False,
        )
        for batch in self.dl:
            images1 = batch["image1"]
            images2 = batch["image2"]
            images3 = batch["image3"]
            lms1 = batch["lm1"]
            lms2 = batch["lm2"]
            lms3 = batch["lm3"]
            for img in images1:
                S1(img)
            for img in images2:
                S2(img)
            for img in images3:
                S3(img)
            for img in lms1:
                Sl1(img)
            for img in lms2:
                Sl2(img)
            for img in lms3:
                Sl3(img)

class MultiRemapsTSL(Transform):
    def __init__(self, lm_key):
        self.lm_key = lm_key
        TSL = TotalSegmenterLabels()
        excluded_tags = ["misc","background"]
        df = TSL.df[~TSL.df.name_region.isin(excluded_tags)]
        self.regions = df.name_region.unique()
        self.Remaps ={}
        for region in self.regions:
            remap = TSL.create_remapping("label_full", region, as_list=True)
            R =MapLabelValued(keys=[self.lm_key], orig_labels=remap[0], target_labels=remap[1])
            self.Remaps[region] = R


    def __call__(self, data):
        remapped_lms = []
        for region, R in self.Remaps.items():
            remapped_lm = R(data)
            remapped_lms.append(remapped_lm[self.lm_key])
        data[self.lm_key] = torch.concat(remapped_lms, dim=0)
        return data
# %%
# SECTION:-----------------------------------------------------------SETUP-----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from fran.data.dataregistry import DS
    from label_analysis.totalseg import TotalSegmenterLabels
    from matplotlib import pyplot as plt
    from monai.transforms.utility.dictionary import MapLabelValued
    from utilz.imageviewers import ImageMaskViewer

    fldr_totalseg = DS["totalseg"].folder
    P = Preprocessor2DTSL(fldr_totalseg, "/s/xnat_shadow/totalseg2d")
# %%
    P.setup()
    P.create_dl()
# %%
    dl = P.dl
    iteri = iter(dl)
    batch = next(iteri)

    batch.keys()
    batch["image1"][0].shape

# %%
    img = batch["image1"][0]
    img2 = batch["image2"][0]
    lm2 = batch["lm2"][0]
    lm1 = batch["lm1"][0]
    lm2 = batch["lm2"][0]

# %%

# %%
    TSL = TotalSegmenterLabels()
    TSL.abdomen.label_full
    excluded_tags = ["misc","background"]
    df = TSL.df[~TSL.df.name_region.isin(excluded_tags)]
    regions = df.name_region.unique()
    n=0
# %%
    remap_ap = TSL.create_remapping("label_full", regions[n], as_list=True)
    remap_chest = TSL.create_remapping("label_full", regions[n+1], as_list=True)

    R = MapLabelValued(keys=["lm"], orig_labels=remap_ap[0], target_labels=remap_ap[1])
    R2 = MapLabelValued(keys=["lm"], orig_labels=remap_chest[0], target_labels=remap_chest[1])



# %%
    n = 0
    dici = P.data[n]
    dici = P.L(dici)
    dici = P.Dev(dici)
    dici = P.E(dici)
    dici = P.O(dici)
    dici = P.N(dici)
    dici = P.Remap(dici)
    # dici = P.P2(dici)
    # dici = P.P3(dici)
# %%
    MM = MultiRemapsTSL(lm_key="lm")
    dici2 = MM(dici)
    lm = dici2["lm"]

    lm.shape
    lm1 = lm[0]
    lm2 = lm[1]
    lm3 = lm[2]

    ImageMaskViewer([lm1, lm2 ], 'mm')
# %%
    dici1 = R(dici)
    dici2 = R2(dici)
    lm1 = dici1["lm"]
    lm2 = dici2["lm"]

    lm3 = torch.concat([lm1, lm2], dim=0)
# %%
    lm1 = dici1["lm"]
    lm2 = dici2["lm"]

# %%
# %%
    dici1 = R(dici)
    img = dici1["image"]
    lm = dici1["lm"]
    ImageMaskViewer([lm1, lm2], 'mm')
    dici2 = P.P1(dici1)

    img2 = dici2["image1"]
    lm2 = dici2["lm1"]
    lm2.shape
# %%
    # im = lm2.squeeze().cpu().numpy()  # shape -> (300, 410)
    im = lm.squeeze().cpu().numpy()  # shape -> (300, 410)

    plt.imshow(im)
    plt.colorbar()
    plt.title("lm2")
    plt.axis("off")
    plt.show()
# %%
    P.setup()
# %%

# %%
    P.process()
# %%
    for i, dat in enumerate(P.ds):
        print(dat.keys())
# %%
    a = next(iter(P.dl))
# %%
    lms = list(P.fldr_lms.glob("*"))
    imgs = list(P.fldr_imgs.glob("*"))[:20]


# %%

# %%


