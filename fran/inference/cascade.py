# %%
import gc
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from lightning.fabric import Fabric
from monai.apps.detection.transforms.array import *
from monai.data.box_utils import *
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset, PersistentDataset
from monai.data.utils import decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.inferers.merger import *
from monai.transforms import (AsDiscreted, Compose, EnsureChannelFirstd,
                              Invertd, Spacingd)
from monai.transforms.io.dictionary import LoadImaged, SaveImaged
from monai.transforms.post.dictionary import (Activationsd,
                                              KeepLargestConnectedComponentd,
                                              MeanEnsembled)
from monai.transforms.spatial.dictionary import Orientationd, Resized
# from monai.transforms.utility.dictionary import AddChanneld, EnsureTyped
from torchvision.transforms.functional import resize

from fran.data.dataloader import img_metadata_collated
from fran.data.dataset import (FillBBoxPatchesd, NormaliseClip, NormaliseClipd,
                               SavePatchd)
from fran.inference.base import (BaseInferer, InferenceDatasetNii,
                                 list_to_chunks, load_dataset_params)
from fran.managers.training import (DataManager, UNetTrainer,
                                    checkpoint_from_model_id)
from fran.transforms.inferencetransforms import (
    BBoxFromPred, KeepLargestConnectedComponentWithMetad, RenameDictKeys,
    ToCPUd, TransposeSITKd)
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
import sys

from fastcore.all import GetAttr, ItemTransform, Pipeline, Sequence
from fastcore.foundation import L, Union, listify, operator

from fran.transforms.intensitytransforms import ClipCenterI

sys.path += ["/home/ub/Dropbox/code/fran/"]

import torch.nn.functional as F
from fastcore.basics import store_attr
from fastcore.transform import Transform as TFC

from fran.utils.imageviewers import ImageMaskViewer


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


class WholeImageInferer(GetAttr, DictToAttr):
    def __init__(
        self,
        project,
        run_name,
        devices=[1],
        debug=True,
        overwrite=False,
    ):
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


    def setup(self,data):
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
        if len(self.ds2)<4:
            nw_bs = [0,1] # Slicer bugs out
        else:
            nw_bs = [12,12]
        # bs=1
        self.pred_dl = DataLoader(self.ds2, num_workers=nw_bs[0], batch_size= nw_bs[1])

    def prepare_model(self):
        self.model = UNetTrainer.load_from_checkpoint(
            self.ckpt,
            project=self.project,
            dataset_params=self.dataset_params,
            strict=False,
        )
        self.model.eval()
        fabric = Fabric(precision="16-mixed", devices=self.devices)
        self.model = fabric.setup(self.model)

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


class PatchInferer(WholeImageInferer):
    def __init__(
        self,
        project,
        run_name,
        patch_overlap=0.25,
        bs=8,
        grid_mode="gaussian",
        devices=[1],
        debug=True,
        overwrite=False,
    ):
        super().__init__(project, run_name=run_name,  devices = devices, debug =debug)

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

        """
        data is a list containing a dataset from ensemble and bboxes
        """
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


class CascadeInferer:  # SPACING HAS TO BE SAME IN PATCHES
    def __init__(
        self,
        project,
        run_name_w,
        runs_p,
        devices=[0],
        debug=False,
        overwrite=False,
        save=True,
    ):
        """
        Creates a single dataset (cascade dataset) which normalises images once for both patches and whole images. Hence, the model should be developed from the same dataset std, mean values.
        """

        self.predictions_folder = project.predictions_folder
        self.dataset_params = load_dataset_params(runs_p[0])
        self.Ps = [PatchInferer(project=project,run_name=run, devices=devices, debug=debug, overwrite=overwrite) for run in runs_p]
        self.W = WholeImageInferer(
            project=project, run_name=run_name_w,  debug=True, devices=devices, overwrite=overwrite
        )
        store_attr()

    def create_ds(self, data):
        """
        data can be filenames or images. InferenceDatasetNii will resolve data type and add LoadImaged if it is a filename
        """

        cache_dir = self.project.cold_datasets_folder / ("cache")
        maybe_makedirs(cache_dir)
        # self.ds = PersistentDataset(data=im, transform=tfms, cache_dir=cache_dir)
        self.ds = InferenceDatasetNii(self.project, data, self.dataset_params)
        self.ds.set_transforms("E")

    def get_patch_spacings(self, run_name):
        ckpt = checkpoint_from_model_id(run_name)
        dic1 = torch.load(ckpt)
        spacings = dic1["datamodule_hyper_parameters"]["dataset_params"]["spacings"]
        return spacings

    def predict(self, imgs:list, chunksize=12):
        """
        imgs can be a list comprising any of filenames, folder, or images (sitk or itk)
        chunksize is necessary in large lists to manage system ram
        """
        imgs=listify(imgs)

        if self.overwrite==False and (isinstance(imgs[0],str) or isinstance(imgs[0], Path)):
            imgs = self.filter_existing_preds(imgs)
        else:
            pass
            # self.save = False  # don't save if input is pure images. Just output those.
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


    def filter_existing_preds(self, imgs):
        print("Filtering existing predictions\nNumber of images provided: {}".format(len(imgs)))
        new_Ps =[]
        for P in self.Ps:
            out_fns = [P.output_folder/img.name for img in imgs]
            new_P = np.array([not fn.exists() for fn in out_fns])
            new_Ps.append(new_P)
        if len(P)>1:
            new_Ps= np.logical_or(*new_Ps)
        else:
            new_Ps = new_Ps[0]
        imgs = list(il.compress(imgs, new_Ps))
        print("Number of images remaining to be predicted: {}".format(len(imgs)))
        return imgs



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
        print("Preparing data")
        self.W.setup(self.ds)
        p = self.W.predict()
        preds = self.W.postprocess(p)
        bboxes = [pred["pred"].meta["bounding_box"] for pred in preds]
        return bboxes

    def patch_prediction(self, ds, bboxes):
        data = [ds, bboxes]
        preds_all_runs = {}
        for P in self.Ps:
            P.setup(data)
            preds = P.predict()
            preds = P.postprocess(preds)
            preds_all_runs[P.run_name] = preds
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
        K = KeepLargestConnectedComponentWithMetad(
            keys=["pred"], independent=False, applied_labels=1
        )  # label=1 is the organ
        F = FillBBoxPatchesd()
        if len(keys) == 1:
            MR = RenameDictKeys(new_keys=["pred"], keys=keys)
        else:
            MR = MeanEnsembled(output_key="pred", keys=keys)
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
    run_ps = ["LITS-720"]

    run_ps = ["LITS-709"]
    run_ps = ["LITS-787"]
# %%
    img_fna = "/s/xnat_shadow/litq/test/images_ub/"
    fns = "/s/datasets_bkp/drli_short/images/"
# %%
    img_fn = "/s/xnat_shadow/crc/images/crc_CRC183_20170922_ABDOMEN.nii.gz"
    img_fn2 = "/s/xnat_shadow/crc/images/crc_CRC261_20170322_AbdoPelvis1p5.nii.gz"

    img_fns = [img_fn, img_fn2]

# %%
    crc_fldr = "/s/xnat_shadow/crc/completed/images"
    crc_imgs = list(Path(crc_fldr).glob("*"))
# %%
    En = CascadeInferer(proj, run_w, run_ps, debug=True, devices=[1])

    img_fn = Path("/s/xnat_shadow/litq/images/litq_31_20220826.nii.gz")
    preds = En.predict(crc_imgs)
# %%
    En = CascadeInferer(proj, run_w, run_ps, debug=True, devices=[1])
# %%
    preds = En.predict(img_fns)
# %%

    imgs = img_fns
    chunksize = 10
    imgs = list_to_chunks(imgs, chunksize)
    # for imgs_sublist in imgs:
    imgs_sublist = imgs[0]
    En.create_ds(imgs_sublist)
    En.bboxes = En.extract_fg_bboxes()
    pred_patches = En.patch_prediction(En.ds, En.bboxes)
    pred_patches = En.decollate_patches(pred_patches, En.bboxes)
    output = En.postprocess(pred_patches)
    if En.save == True:
        En.save_pred(output)

# %%
        w = WholeImageInferer(En.project, En.run_name_w, En.ds, debug=True)
        print("Preparing data")
        p = w.predict()
        preds = w.postprocess(p)
# %%
        out_final = []
        for batch in preds:
            batch = p[0]
            out2 = decollate_batch(batch, detach=True)
            ou = out2[0]
            for ou in out2:
                tmp = w.postprocess_transforms(ou)
                out_final.append(tmp)
# %%
        bboxes = [pred["pred"].meta["bounding_box"] for pred in preds]

# %%
    pred_patches = En.patch_prediction(En.ds, En.bboxes)
    pred_patches = En.decollate_patches(pred_patches, En.bboxes)
    pb = patch_bundle = pred_patches
    keys = En.runs_p
# %%
    C = Compose(tfms)
    A = Activationsd(keys="pred", softmax=True)
    D = AsDiscreted(keys=["pred"], argmax=True)
    K = KeepLargestConnectedComponentWithMetad(
        keys=["pred"], independent=False, applied_labels=1
    )  # label=1 is the organ
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
    R = RenameDictKeys(keys=keys, new_keys=["pred"])
# %%
    pb = patch_bundle[0]
    pb33 = C(pb)
    pb["pred"] = pb["LITS-709"]
    pb2 = C(pb)
    pb3 = E(pb2)

    pb2 = M(pb)
    pb4 = A(pb3)
    pb3.keys()
    pred = pb33["pred"]
    pred.shape
    img = pb33["image"]
    img.shape
    pred.shape

    ImageMaskViewer([img[0], pred[0]])
# %%
    w = WholeImageInferer(
        En.project, En.run_name_w, En.ds, debug=True, devices=En.devices
    )
# %%
    print("Preparing data")
    p = w.predict()

