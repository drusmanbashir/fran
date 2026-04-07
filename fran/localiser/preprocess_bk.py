# %%

from pathlib import Path

import ipdb
from fran.data.collate import as_is_collated
from fran.transforms.imageio import LoadSITKd, TorchWriter
from fran.transforms.misc_transforms import DictToMetad, MetaToDict
from fran.transforms.spatialtransforms import Project2D
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.transforms import Compose
from monai.transforms.croppad.dictionary import BoundingRectd
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.io.array import SaveImage
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from utilz.fileio import is_sitk_file, maybe_makedirs
from utilz.helpers import find_matching_fn

tr = ipdb.set_trace


class Preprocessor2D:
    def __init__(self, fldr_imgs, fldr_lms, output_fldr):
        # input_fldr has subfolders images and lms
        self.fldr_imgs = fldr_imgs
        self.fldr_lms = fldr_lms
        self.output_fldr = output_fldr

        self.output_fldr = Path(output_fldr)
        self.output_fldr_imgs = self.output_fldr / ("images")
        self.output_fldr_lms = self.output_fldr / ("lms")
        self.tfms_keys = ["L", "N", "E", "P1", "P2", "P3"]

    def setup(self, batch_size=8):
        imgs = self.fldr_imgs.glob("*")
        imgs = [img for img in imgs if is_sitk_file(img)]
        lms = list(self.fldr_lms.glob("*"))
        data_dicts = []
        for img in imgs:
            lm = find_matching_fn(img, lms, tag=["all"])[0]
            if lm:
                dici = {"image": img, "lm": find_matching_fn(img, lms, tag=["all"])[0]}
                data_dicts.append(dici)

        self.create_transforms()
        self.ds = Dataset(data=data_dicts, transform=self.tfms_from_dict(self.tfms_keys))
        self.create_dl(num_workers=batch_size * 2, batch_size=batch_size)

    def create_transforms(self):
        self.L = LoadSITKd(keys=["image", "lm"])
        self.N = NormalizeIntensityd(["image"])
        self.E = EnsureChannelFirstd(["image", "lm"])
        self.P1 = Project2D(
            keys=["lm", "image"],
            operations=["sum", "mean"],
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
            "L": self.L,
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

    def tfms_from_dict(self, keys):
        tfms = [self.transforms_dict[key] for key in keys]
        return Compose(tfms)

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


# %%


# %%
# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
if __name__ == "__main__":
    fldr_imgs = Path("/s/xnat_shadow/lidc2/images/")
    fldr_lms = Path("/s/fran_storage/predictions/totalseg/LITS-860/")

    P = Preprocessor2D(fldr_imgs, fldr_lms, "/s/xnat_shadow/lidc2d")

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
    lms = list(fldr_lms.glob("*"))
    imgs = list(fldr_imgs.glob("*"))[:20]

# %%

# %%
