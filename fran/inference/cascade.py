# %%
import gc
import sys
from collections.abc import Callable
from pathlib import Path
from time import time

import lightning.pytorch as pl
import monai
import numpy as np
import SimpleITK as sitk
from lightning.fabric import Fabric
from monai.apps.detection.transforms.array import *
from monai.config.type_definitions import DtypeLike, KeysCollection
from monai.data import MetaTensor, image_writer
from monai.data.box_utils import *
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset, PersistentDataset
from monai.data.image_writer import ITKWriter
from monai.data.itk_torch_bridge import itk_image_to_metatensor as itm
from monai.data.itk_torch_bridge import metatensor_to_itk_image
from monai.data.utils import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.inferers.merger import *
from monai.inferers.utils import sliding_window_inference
from monai.transforms import (AsDiscreted, Compose, EnsureChannelFirstd,
                              Invertd, LoadImage, Spacingd)
from monai.transforms.croppad.dictionary import (BoundingRectd,
                                                 ResizeWithPadOrCropd)
from monai.transforms.io.array import SaveImage
from monai.transforms.io.dictionary import LoadImaged, SaveImaged
from monai.transforms.post.dictionary import (Activationsd,
                                              KeepLargestConnectedComponentd,
                                              MeanEnsembled)
from monai.transforms.spatial.dictionary import Flipd, Orientationd, Resized
from monai.transforms.transform import MapTransform, Transform
from monai.transforms.utility.array import EnsureType
from monai.transforms.utils import generate_spatial_bounding_box
# from monai.transforms.utility.dictionary import AddChanneld, EnsureTyped
from monai.utils.enums import GridSamplePadMode
from torch.functional import Tensor
from torchvision.transforms.functional import resize

from fran.data.dataloader import img_metadata_collated
from fran.data.dataset import (FillBBoxPatchesd, NormaliseClip, NormaliseClipd,
                               SavePatchd)
from fran.managers.training import (DataManager, UNetTrainer,
                                    checkpoint_from_model_id)
from fran.transforms.inferencetransforms import KeepLargestConnectedComponentWithMetad, RenameDictKeys
from fran.utils.common import *
from fran.utils.dictopts import DictToAttr
from fran.utils.fileio import load_dict, load_yaml, maybe_makedirs
from fran.utils.helpers import get_available_device, timing
from fran.utils.itk_sitk import *
from fran.utils.string import drop_digit_suffix

sys.path += ["/home/ub/code"]
from mask_analysis.helpers import to_cc, to_int, to_label

# These are the usual ipython objects, including this one you are creating
ipython_vars = ["In", "Out", "exit", "quit", "get_ipython", "ipython_vars"]
import os
import sys

from fastcore.all import GetAttr, ItemTransform, Pipeline, Sequence
from fastcore.foundation import L, Union, listify, operator
from monai.transforms.post.array import (Activations, AsDiscrete, Invert,
                                         KeepLargestConnectedComponent,
                                         VoteEnsemble)

from fran.transforms.intensitytransforms import ClipCenterI

sys.path += ["/home/ub/Dropbox/code/fran/"]
import functools as fl

import torch.nn.functional as F
from fastcore.basics import store_attr
from fastcore.transform import Transform as TFC

from fran.utils.imageviewers import ImageMaskViewer


def slice_list(listi, start_end: list):
    return listi[start_end[0] : start_end[1]]


def list_to_chunks(input_list: list, chunksize: int):
    n_lists = int(np.ceil(len(input_list) / chunksize))

    fpl = int(len(input_list) / n_lists)
    inds = [[fpl * x, fpl * (x + 1)] for x in range(n_lists - 1)]
    inds.append([fpl * (n_lists - 1), None])

    chunks = list(il.starmap(slice_list, zip([input_list] * n_lists, inds)))
    return chunks


class Saved(Transform):
    def __init__(self, output_folder):
        self.output_folder = output_folder

    def __call__(self, patch_bundle):
        key = "pred"

        maybe_makedirs(self.output_folder)
        try:
            fl = Path(patch_bundle["image"].meta["filename_or_obj"]).name
        except:
            fl = "_tmp.nii.gz"
        outname = self.output_folder / fl

        meta = {
            "original_affine": patch_bundle["image"].meta["original_affine"],
            "affine": patch_bundle["image"].meta["affine"],
        }

        writer = ITKWriter()

        array_full = patch_bundle[key].detach().cpu()
        array_full.meta = patch_bundle["image"].meta
        channels = array.shape[0]
        # ch=0
        # array = array_full[ch:ch+1,:]
        writer.set_data_array(array)
        writer.set_data_array(patch_bundle["image"])
        writer.set_metadata(meta)
        assert (
            di := patch_bundle[key].dim()
        ) == 4, "Dimension should be 4. Got {}".format(di)
        writer.write(outname)


def transpose_bboxes(bbox):
    """
    Transposes the coordinates of a bounding box.

    Parameters:
    bbox (tuple): A tuple representing the coordinates of a bounding box in the format (x1, y1, x2, y2).

    Returns:
    tuple: A tuple representing the transposed coordinates of the bounding box in the format (y1, x1, y2, x2).

    Example:
    >>> transpose_bboxes((10, 20, 30, 40))
    (20, 10, 40, 30)
    """

    if not isinstance(bbox, tuple):
        raise TypeError("bbox must be a tuple")

    if len(bbox) != 4:
        raise ValueError("bbox must have exactly 4 elements")

    bbox = bbox[1], bbox[0], bbox[3], bbox[2]
    return bbox


#
class TransposeSITKd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def func(self, x):
        if isinstance(x, torch.Tensor):
            x = torch.permute(x, [2, 1, 0])
        elif isinstance(x, Union[tuple, list]):
            x = transpose_bboxes(x)
        else:
            raise NotImplemented
        return x

    def __call__(self, d: dict):
        for key in self.key_iterator(d):
            d[key] = self.func(d[key])
        return d


class ToCPUd(MapTransform):
    def __call__(self, d: dict):
        for key in self.key_iterator(d):
            d[key] = d[key].cpu()
        return d


class BBoxFromPred(MapTransform):
    def __init__(
        self,
        spacings,
        expand_by: int,  # in millimeters
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        store_attr("spacings,expand_by")

    def __call__(self, d: dict):
        for key in self.key_iterator(d):
            d[key] = self.func(d[key])
        return d

    def func(self, pred):
        add_to_bbox = [int(self.expand_by / sp) for sp in self.spacings]
        bb = generate_spatial_bounding_box(pred, channel_indices=0, margin=add_to_bbox)
        sls = [slice(0, 100, None)] + [slice(a, b, None) for a, b in zip(*bb)]
        pred.meta["bounding_box"] = sls
        return pred


class SlicesFromBBox(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def func(self, x):
        b = x[0]  # get rid of channel
        slices = [slice(0, 100)]  # 100 channels
        for ind in [0, 2, 4]:
            s = slice(b[ind], b[ind + 1])
            slices.append(s)
        return tuple(slices)

    def __call__(self, d: dict):
        for key in self.key_iterator(d):
            d[key] = self.func(d[key])
        return d


class SimpleTrainer(UNetTrainer):
    def test_step(self, batch, batch_idx):
        img = batch["image"]
        outputs = self.forward(img)
        outputs2 = outputs[0]
        batch["pred"] = outputs2
        # output=outputs[0]
        # outputs = {'pred':output,'org_size':batch['org_size']}
        # outputs_backsampled=self.post_process(outputs)
        return batch


class ImageBBoxDataset(Dataset):
    def __init__(self, data, transform: Union[Callable, None] = None) -> None:
        self.ds, self.bboxes = data
        self.transform = transform

    def __getitem__(self, idx):
        im = self.ds[idx]["image"]
        bbox = self.bboxes[idx]
        img_c = im[bbox]
        outputs = {"image": im, "image_cropped": img_c, "bbox": bbox}
        if self.transform:
            outputs = self.transform(outputs)
        return outputs

    def __len__(self):
        return len(self.ds)


def img_bbox_collated(batch):
    imgs = []
    imgs_c = []
    bboxes = []
    for i, item in enumerate(batch):
        imgs.append(item["image"])
        imgs_c.append(item["image_cropped"])
        bboxes.append(item["bbox"])
    output = {
        "image": torch.stack(imgs, 0),
        "image_cropped": torch.stack(imgs_c, 0),
        "bbox": bboxes,
    }
    return output


class PersistentDS(PersistentDataset):
    def __init__(self, imgs: Union[torch.Tensor, list], cache_dir) -> None:
        L = LoadImaged(
            keys=["image"], image_only=True, ensure_channel_first=True, simple_keys=True
        )
        # E = EnsureChannelFirstd(keys=["image"], channel_dim="no_channel")
        O = Orientationd(keys=["image"], axcodes="RAS")
        tfms = Compose([L, O])
        self.cache_dir = cache_dir
        super().__init__(imgs, tfms)

    def create_batch_transforms(self):
        # T = TransposeSITKd(keys=["image", "org_size"])
        # E = EnsureChannelFirstd(keys=["image"], channel_dim="no_channel")
        N = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )
        # S = Spacingd(keys=["image"], pixdim=self.dataset_params["spacings"])
        C = Compose([N], lazy=True)
        self.batch_transforms = C


class PatchDM(DataManager):
    def create_batch_transforms(self):
        T = TransposeSITKd(keys=["image", "org_size"])
        E = EnsureChannelFirstd(keys=["image"], channel_dim="no_channel")
        N = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )
        S = Spacingd(keys=["image"], pixdim=self.dataset_params["spacings"])
        C = Compose([T, E, N, S], lazy=True)
        self.transforms = C

    def patch_batchmaker(self, batch):  # note batch_size = 1
        imgs = []

        R = Resized(keys=["image"], spatial_size=patch_size)
        tfms = Compose([R])
        for i, item in enumerate(batch):
            img = tfms(item)["image"]
            imgs.append(img)
        output = {"image": torch.stack(imgs, 0)}
        return output

    def create_dataloader(self):
        pass


class WholeImagePredictor(GetAttr, DictToAttr):
    def __init__(self, project, run_name, data, devices=1, debug=True):
        """
        data is a dataset from Ensemble in this base class
        """

        store_attr("project,run_name,devices,debug")
        self.ckpt = checkpoint_from_model_id(run_name)
        dic1 = torch.load(self.ckpt)
        dic2 = {}
        relevant_keys = ["datamodule_hyper_parameters"]
        for key in relevant_keys:
            dic2[key] = dic1[key]
            self.assimilate_dict(dic2[key])

        self.prepare_model()
        self.prepare_data(data)
        self.create_postprocess_transforms()

    def create_postprocess_transforms(self):
        I = Invertd(keys=["pred"], transform=self.ds2.transform, orig_keys=["image"])
        D = AsDiscreted(keys=["pred"], argmax=True, threshold=0.5)
        K = KeepLargestConnectedComponentd(keys=["pred"])
        C = ToCPUd(keys=["image", "pred"])
        B = BBoxFromPred(
            keys=["pred"], expand_by=20, spacings=self.dataset_params["spacings"]
        )
        tfms = [I, D, K, C, B]
        if self.debug == True:
            Sa = SaveImaged(
                keys=["pred"],
                output_dir=self.output_folder,
                output_postfix="",
                separate_folder=False,
            )
            #
            # Sa = SavePatchd(keys=['pred'],output_folder=self.output_folder,postfix_channel=True)
            tfms.insert(1, Sa)
        C = Compose(tfms)
        self.postprocess_transforms = C

    def prepare_data(self, ds):
        R = Resized(keys=["image"], spatial_size=self.dataset_params["patch_size"])

        # R2 = ResizeWithPadOrCropd(
        #         keys=["image"],
        #         source_key="image",
        #         spatial_size=self.dataset_params["patch_size"],
        #     )
        N = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )
        tfm = Compose([R, N])
        self.ds2 = Dataset(data=ds, transform=tfm)
        self.pred_dl = DataLoader(self.ds2, num_workers=12, batch_size=12)

    def prepare_model(self):
        self.model = UNetTrainer.load_from_checkpoint(
            self.ckpt,
            project=self.project,
            dataset_params=self.dataset_params,
            strict=False,
        )

    def predict(self):
        outputs = []
        self.model.eval()
        with torch.inference_mode():
            for i, batch in enumerate(self.pred_dl):
                img = batch["image"].cuda()
                output = self.model(img)
                output = output[0]
                batch["pred"] = output
                batch["pred"].meta = batch["image"].meta
                outputs.append(batch)
        return outputs

    def postprocess(self, preds):
        out_final = []
        for batch in preds:
            out2 = decollate_batch(batch, detach=True)
            for ou in out2:
                tmp = self.postprocess_transforms(ou)
                out_final.append(tmp)
        return out_final

    @property
    def output_folder(self):
        run_name = listify(self.run_name)
        fldr = "_".join(run_name)
        fldr = self.project.predictions_folder / fldr
        return fldr


class PatchPredictor(WholeImagePredictor):
    def __init__(
        self,
        project,
        run_name,
        data,
        patch_overlap=0.25,
        bs=8,
        grid_mode="gaussian",
        devices=1,
        debug=True,
    ):
        super().__init__(project, run_name, data, devices, debug)
        """
        data is a list containing a dataset from ensemble and bboxes
        """

        self.grid_mode = grid_mode
        self.patch_size = self.dataset_params["patch_size"]
        self.inferer = SlidingWindowInferer(
            roi_size=self.dataset_params["patch_size"],
            sw_batch_size=bs,
            overlap=patch_overlap,
            mode=grid_mode,
            progress=True,
        )

    def prepare_data(self, data):
        S = Spacingd(keys=["image_cropped"], pixdim=self.dataset_params["spacings"])
        N = NormaliseClipd(
            keys=["image_cropped"],
            clip_range=self.dataset_params["intensity_clip_range"],
            mean=self.dataset_params["mean_fg"],
            std=self.dataset_params["std_fg"],
        )

        # E = EnsureTyped(keys=["image" ], device="cuda", track_meta=True),
        tfm = Compose([N, S])
        self.ds2 = ImageBBoxDataset(data, transform=tfm)
        self.pred_dl = DataLoader(
            self.ds2,
            num_workers=0,
            batch_size=1,
            collate_fn=img_bbox_collated,  # essential to avoid size mismatch
        )

    def predict(self):
        # outputs a list of prediction batches
        outputs = []
        for i, batch in enumerate(self.pred_dl):
            with torch.no_grad():
                img_input = batch["image_cropped"]
                img_input = img_input.cuda()
                output_tensor = self.inferer(inputs=img_input, network=self.model)
                output_tensor = output_tensor[0]
                batch["pred"] = output_tensor
                batch["pred"].meta = batch["image"].meta
                outputs.append(batch)
        return outputs

    def postprocess(self, outputs):
        I = Invertd(
            keys=["pred"], transform=self.ds2.transform, orig_keys=["image_cropped"]
        )
        C = ToCPUd(keys=["image", "pred"])
        tfms = [I, C]
        if self.debug == True:
            S = SavePatchd(["pred"], self.output_folder, postfix_channel=True)
            tfms += [S]
        Co = Compose(tfms)
        out_final = []
        for batch in outputs:  # batch_length is 1
            batch["pred"] = batch["pred"].squeeze(0).detach()
            batch["image"] = batch["image"].squeeze(0).detach()
            batch["bbox"] = batch["bbox"][0]
            batch = Co(batch)
            out_final.append(batch)
        return out_final

    def prepare_model(self):
        super().prepare_model()
        self.model.eval()
        fabric = Fabric(precision="16-mixed", devices=self.devices)
        self.model = fabric.setup(self.model)


class EnsemblePredictor:  # SPACING HAS TO BE SAME IN PATCHES
    def __init__(
        self,
        project,
        run_name_w,
        runs_p,
        device="cuda",
        debug=False,
        overwrite=False,
        save=True,
    ):
        self.predictions_folder = project.predictions_folder
        store_attr()

    def create_ds(self, im):
        L = LoadImaged(
            keys=["image"], image_only=True, ensure_channel_first=True, simple_keys=True
        )
        O = Orientationd(keys=["image"], axcodes="RSI")  # nOTE RPS
        FF = Flipd(keys=["image_cropped"], spatial_axis=0)
        tfms = Compose([L, O, FF])

        cache_dir = self.project.cold_datasets_folder / ("cache")
        maybe_makedirs(cache_dir)
        self.ds = PersistentDataset(data=im, transform=tfms, cache_dir=cache_dir)

    def get_patch_spacings(self, run_name):
        ckpt = checkpoint_from_model_id(run_name)
        dic1 = torch.load(ckpt)
        spacings = dic1["datamodule_hyper_parameters"]["dataset_params"]["spacings"]
        return spacings

    def predict(self, imgs, chunksize=12):
        """
        chunksize is necessary in large lists to manage system ram
        """

        imgs = self.parse_input(imgs)
        imgs = list_to_chunks(imgs, chunksize)
        for imgs_sublist in imgs:
            self.create_ds(imgs_sublist)
            self.bboxes = self.extract_fg_bboxes()
            pred_patches = self.patch_prediction(self.ds, self.bboxes)
            pred_patches = self.decollate_patches(pred_patches, self.bboxes)
            output = self.postprocess(pred_patches)
            if self.save == True:
                self.save_pred(output)
        return output

    def parse_input(self, imgs_inp):
        """
        input types:
            folder of img_fns
            nifti img_fns
            itk imgs (slicer)

        returns list of img_fns if folder. Otherwise just the imgs
        """

        if not isinstance(imgs_inp, list):
            imgs_inp = [imgs_inp]
        imgs_out = []
        for dat in imgs_inp:
            if any([isinstance(dat, str), isinstance(dat, Path)]):
                self.input_type = "files"
                dat = Path(dat)
                if dat.is_dir():
                    dat = list(dat.glob("*"))
                else:
                    dat = [dat]
            else:
                self.input_type = "itk"
                if isinstance(dat, sitk.Image):
                    dat = ConvertSimpleItkImageToItkImage(dat, itk.F)
                # if isinstance(dat,itk.Image):
                dat = itm(dat)
            imgs_out.extend(dat)
        imgs_out = [{"image": img} for img in imgs_out]
        return imgs_out

    def save_pred(self, preds):
        S = SaveImaged(
            keys=["pred"],
            output_dir=self.output_folder,
            output_postfix="",
            separate_folder=False,
        )
        for pp in preds:
            S(pp)

    def get_mini_bundle(self, patch_bundles, indx):
        patch_bundle = {}
        for key, val in patch_bundles.items():
            pred_patch = {key: val[indx]}
            patch_bundle.update(pred_patch)
        return patch_bundle

    def decollate_patches(self, pa, bboxes):
        num_cases = len(self.ds)
        keys = self.runs_p
        keys = En.runs_p
        output = []
        for case_idx in range(num_cases):
            img_bbox_preds = {}
            for i, run_name in enumerate(keys):
                pred = pa[run_name][case_idx]["pred"]
                img_bbox_preds[run_name] = pred
            img_bbox_preds.update(self.ds[case_idx])
            img_bbox_preds["bbox"] = bboxes[case_idx]
            output.append(img_bbox_preds)
        return output

    def extract_fg_bboxes(self):
        w = WholeImagePredictor(self.project, self.run_name_w, self.ds, debug=True)
        print("Preparing data")
        p = w.predict()
        preds = w.postprocess(p)
        bboxes = [pred["pred"].meta["bounding_box"] for pred in preds]
        return bboxes

    def patch_prediction(self, ds, bboxes):
        data = [ds, bboxes]
        preds_all_runs = {}
        for run in self.runs_p:
            p = PatchPredictor(self.project, run, data=data, debug=self.debug)
            preds = p.predict()
            preds = p.postprocess(preds)
            preds_all_runs[run] = preds
        return preds_all_runs

    @property
    def output_folder(self):
        fldr = "_".join(self.runs_p)
        fldr = self.predictions_folder / fldr
        return fldr

    def postprocess(self, patch_bundle):
        keys = self.runs_p
        A = Activationsd(keys="pred", softmax=True)
        D = AsDiscreted(keys=["pred"], argmax=True)
        K = KeepLargestConnectedComponentWithMetad(keys=["pred"], independent=False,applied_labels=1) # label=1 is the organ
        F = FillBBoxPatchesd()
        if len(keys) == 1:
            MR = RenameDictKeys(keys=keys, new_keys=["pred"])
        else:
            MR = MeanEnsembled(keys=keys, output_key="pred")
        tfms = [MR, A, D, K, F]

        # S = SaveListd(keys = ['pred'],output_dir=self.output_folder,output_postfix='',separate_folder=False)
        C = Compose(tfms)
        output = C(patch_bundle)
        return output


# %%

if __name__ == "__main__":
    # ... run your application ...
    proj = Project(project_title="litsmc")

    run_w = "LIT-145"
    run_ps = ["LIT-143", "LIT-150", "LIT-149", "LIT-153", "LIT-161"]
    run_ps = ["LITS-630", "LITS-633", "LITS-632", "LITS-647", "LITS-650"]
    run_ps = ["LITS-709"]

    # %%
    img_fna = "/s/xnat_shadow/litq/test/images_ub/"
    fns = "/s/datasets_bkp/drli_short/images/"
# %%
    img_fn = "/s/datasets_bkp/lits_segs_improved/images/lits_6ub.nii"
    img_fn2 = (
        "/s/xnat_shadow/crc/test/images/finalised/crc_CRC83b_20130726_Abdomen.nii.gz"
    )
    img_fn3 = (
        "/s/xnat_shadow/crc/test/images/finalised/crc_CRC014_20190923_CAP1p5.nii.gz"
    )
    img_fns = listify(img_fn3)

    img_fns = [img_fn, img_fn2, img_fn3]

# %%
    crc_fldr = "/s/xnat_shadow/crc/test/images/finalised/"
    crc_imgs = list(Path(crc_fldr).glob("*"))
    # %%
    chunk = 10
    import math

    n_imgs = len(crc_imgs)
    chunks = math.ceil(n_imgs / chunk)
    for n in range(0, chunks):
        imgs = crc_imgs[n * chunk : (n + 1) * chunk]
        # im = [{'image':im} for im in [img_fn,img_fn2]]
        En = EnsemblePredictor(proj, run_w, run_ps, debug=True, device=[1])
        preds = En.predict(img_fns)
# %%
    ds = En.ds
    bboxes = En.bboxes
    data = [ds, bboxes]
    preds_all_runs = {}

    run = En.runs_p[0]
    p = PatchPredictor(En.project, run, data=data, debug=En.debug)
    preds = p.predict()
    preds_patch = p.postprocess(preds)
    preds_all_runs[run] = preds_patch

    n = 0
    ds2 = p.ds2
    batch = ds2[0]
    im = batch["image_cropped"]
    im_c = preds_patch[n]["image_cropped"]

    FF = Flipd(keys=["image_cropped"], spatial_axis=0)
    batch2 = FF(batch)
    im_c2 = batch2["image_cropped"]
    ImageMaskViewer([im_c[0].permute(2, 1, 0), im_c2[0]], data_types=["image", "mask"])
# %%
# %%
    pred = preds_patch[1]
    pred["image"].shape
    pred["pred"].shape
    pred["bbox"]

    pred["pred"]
# %%

# %%
    outputs = preds
    I = Invertd(keys=["pred"], transform=p.ds2.transform, orig_keys=["image_cropped"])
    C = ToCPUd(keys=["image", "pred"])
    tfms = [I, C, F]
    if p.debug == True:
        S = SavePatchd(["pred"], p.output_folder, postfix_channel=True)
        tfms += [S]
    C = Compose(tfms)
# %%
    out_final = []
    for batch in outputs:  # batch_length is 1
        batch["pred"] = batch["pred"].squeeze(0).detach()
        batch["image"] = batch["image"].squeeze(0).detach()
        # batch['bbox']=batch['bbox']
        batch = I(batch)
        batch = S(batch)
        batch = C(batch)
        out_final.append(batch)

# %%
    ImageMaskViewer([image[0], full[2]])
# %%
    cropped_tnsr = batch["pred"]
    bbox = batch["bbox"]
    chs = cropped_tnsr.shape[0]
    for ch in range(1, chs):
        postfix = str(ch) if S.postfix_channel == True else None
        img_full = fill_bbox(bbox, cropped_tnsr)
        img_save = img_full[ch : ch + 1]

        S = SaveImage(
            output_dir=self.output_folder, output_postfix=postfix, separate_folder=False
        )
        S(img_save)

    I = Invertd(keys=["pred"], transform=p.ds2.transform, orig_keys=["image_cropped"])
    C = ToCPUd(keys=["image", "pred"])
    tfms = [I, C]
    if p.debug == True:
        S = SavePatchd(["pred"], p.output_folder, postfix_channel=True)
        tfms += [S]
    Co = Compose(tfms)
    outputs = preds
    out_final = []
    for batch in outputs:  # batch_length is 1
        batch["pred"] = batch["pred"].squeeze(0).detach()
        batch["image"] = batch["image"].squeeze(0).detach()
        batch["bbox"] = batch["bbox"][0]
        batch = Co(batch)
        out_final.append(batch)

# %%

    pred_patches = En.patch_prediction(En.ds, En.bboxes)
    pred_patches.keys()
    pred_patches2 = En.decollate_patches(pred_patches, En.bboxes)
    pred_patches2[0]["LITS-630"].shape
    output = En.postprocess(pred_patches2)
# %%
    patch_bundle = pred_patches2
    keys = En.runs_p
    E = EnsureChannelFirstd(keys=keys)
    M = MeanEnsembled(keys=keys, output_key="pred")
    A = Activationsd(keys="pred", softmax=True)
    D = AsDiscreted(keys=["pred"], argmax=True)
    F = FillBBoxPatchesd()
    K = KeepLargestConnectedComponentWithMetad(keys=["pred"], independent=False)
    # S = SaveListd(keys = ['pred'],output_dir=En.output_folder,output_postfix='',separate_folder=False)
    tfms = [E, M, A, D, F, K]
    C = Compose(tfms)
    patch_bundle2 = C(patch_bundle)
    pred = patch_bundle2[0]["pred"]
    pred = pred[0]
# %%
# %%
    pb = patch_bundle[0]
    pb2 = E(pb)
    pb3 = M(pb2)
    pb4 = A(pb3)
    pb5 = D(pb4)
    pb6 = F(pb5)
    pb7 = K(pb6)
    p2 = pb2[keys[0]][0, 1]
    p3 = pb3["pred"][1]
    p4 = pb4["pred"][1]
    p5 = pb5["pred"][0]
    p6 = pb6["pred"][0]
    p7 = pb7["pred"][0]
    pred_patches2[0][keys[0]].shape

    pred = pa2[0]["pred"]
    pa3 = Ec(pa2)
    pa3["pred"].shape
# %%
    ImageMaskViewer([p7, pred], data_types=["mask", "mask"])
# %%
    C = Compose(tfms)
    pb = patch_bundle[0]
    pb2 = M(pb)

    p = pb["LITS-630"]
    p2 = pb2["pred"]
    # AC = AddChanneld(keys = 'pred')
    A = Activationsd(keys="pred", softmax=True)
    D = AsDiscreted(keys=["pred"], argmax=True)
    K = KeepLargestConnectedComponentWithMetad(keys=["pred"], independent=False)
    F = FillBBoxPatchesd()
    tfms = [MorA, A, D, K, F]
    # S = SaveListd(keys = ['pred'],output_dir=self.output_folder,output_postfix='',separate_folder=False)
    C = Compose(tfms)
# %%

    patch_bundle2 = C(patch_bundle)

    patch_bundle2[0]["pred"].shape
# %%
    n = 0
    pa = patch_bundle[n]["LITS-630"]
    pa.shape
    img = patch_bundle2[n]["image"]
    pred = patch_bundle2[n]["pred"]
    pb = patch_bundle2[n]
# %%

    pb3 = D(pb)
    pred2 = pb3["pred"]
# %%
    ImageMaskViewer([pred, pred2])
# %%

    casei = output[0]
    img = casei["img"]
    pred = casei["pred"]
# %%
    pred_patches = En.patch_prediction(En.ds, En.bboxes)
    pred_patches = En.decollate_patches(pred_patches, En.bboxes)
    pb = patch_bundle = pred_patches
    keys = En.runs_p
# %%
    C = Compose(tfms)
    A = Activationsd(keys="pred", softmax=True)
    D = AsDiscreted(keys=["pred"], argmax=True)
    K = KeepLargestConnectedComponentWithMetad(keys=["pred"], independent=False,applied_labels=1) # label=1 is the organ
    F = FillBBoxPatchesd()
    if len(keys) == 1:
        MR = RenameDictKeys(keys=keys, new_keys=["pred"])
    else:
        MR = MeanEnsembled(keys=keys, output_key="pred")
    tfms = [MR, A, D, K, F]

    # S = SaveListd(keys = ['pred'],output_dir=self.output_folder,output_postfix='',separate_folder=False)
    C = Compose(tfms)

# %%

    if len(En.runs_p) == 0:
        E = EnsureChannelFirstd(keys="pred")
        tfms.insert(1, E)
    # S = SaveListd(keys = ['pred'],output_dir=En.output_folder,output_postfix='',separate_folder=False)
    output = C(patch_bundle)

# %%
    R = RenameDictKeys(keys = keys, new_keys = ['pred'])
# %%
    pb = patch_bundle[0]
    pb33 = C(pb)
    pb ['pred']= pb['LITS-709']
    pb2 = C(pb)
    pb3 = E(pb2)

    pb2 =  M(pb)
    pb4 =  A(pb3)
    pb3.keys()
    pred = pb33['pred']
    pred.shape
    img = pb33['image']
    img.shape
    pred.shape
    
    ImageMaskViewer([img[0], pred[0]])
# %%
