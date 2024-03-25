# %%
import itertools as il
from collections.abc import Callable, Sequence
from pathlib import Path

import itk
import numpy as np
import SimpleITK as sitk
import torch
from fastcore.all import listify, store_attr
from fastcore.foundation import GetAttr
from lightning.fabric import Fabric
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset, PersistentDataset
from monai.data.itk_torch_bridge import itk_image_to_metatensor as itm
from monai.data.utils import decollate_batch
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged, SaveImaged
from monai.transforms.post.dictionary import Activationsd, AsDiscreted, Invertd
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import (EnsureChannelFirstd,
                                                 SqueezeDimd)
from prompt_toolkit.shortcuts import input_dialog

from fran.data.dataset import InferenceDatasetNii, InferenceDatasetPersistent
from fran.managers.training import UNetTrainer, checkpoint_from_model_id
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import SaveMultiChanneld, ToCPUd
from fran.utils.dictopts import DictToAttr
from fran.utils.fileio import maybe_makedirs
from fran.utils.helpers import slice_list
from fran.utils.imageviewers import ImageMaskViewer, view_sitk
from fran.utils.itk_sitk import ConvertSimpleItkImageToItkImage


def list_to_chunks(input_list: list, chunksize: int):
    n_lists = int(np.ceil(len(input_list) / chunksize))

    fpl = int(len(input_list) / n_lists)
    inds = [[fpl * x, fpl * (x + 1)] for x in range(n_lists - 1)]
    inds.append([fpl * (n_lists - 1), None])

    chunks = list(il.starmap(slice_list, zip([input_list] * n_lists, inds)))
    return chunks


def load_dataset_params(model_id):
    ckpt = checkpoint_from_model_id(model_id)
    dic_tmp = torch.load(ckpt)
    dataset_params = dic_tmp["datamodule_hyper_parameters"]["dataset_params"]
    return dataset_params


class BaseInferer(GetAttr, DictToAttr):
    def __init__(
        self,
        project,
        run_name,
        bs=8,
        patch_overlap=0.25,
        mode="gaussian",
        devices=[1],
        debug=True,
        save=True,
        overwrite=True,
    ):
        """
        data is a dataset from Ensemble in this base class
        """

        store_attr("project,run_name,devices,debug, overwrite,save")
        self.dataset_params = load_dataset_params(run_name)

        self.inferer = SlidingWindowInferer(
            roi_size=self.dataset_params["patch_size"],
            sw_batch_size=bs,
            overlap=patch_overlap,
            mode=mode,
            progress=True,
        )
        self.ckpt = checkpoint_from_model_id(run_name)
        self.prepare_model()
        # self.prepare_data(data)

    def run(self, imgs, chunksize=12):
        """
        chunksize is necessary in large lists to manage system ram
        """
        imgs = list_to_chunks(imgs, chunksize)
        for imgs_sublist in imgs:
            self.prepare_data(imgs_sublist)
            self.create_postprocess_transforms()
            preds = self.predict()
            # preds = self.decollate(preds)
            output = self.postprocess(preds)
            if self.save == True:
                self.save_pred(output)
        return output

    def prepare_data(self, imgs):
        """
        imgs: list
        """
        self.ds = InferenceDatasetNii(self.project, imgs, self.dataset_params)
        self.ds.set_transforms("ESN")
        self.pred_dl = DataLoader(self.ds, num_workers=0, batch_size=1, collate_fn=None)

    def save_pred(self, preds):
        S = SaveImaged(
            keys=["pred"],
            output_dir=self.output_folder,
            output_postfix="",
            separate_folder=False,
        )
        for pp in preds:
            S(pp)

    def create_postprocess_transforms(self):

        Sq = SqueezeDimd(keys=["pred"], dim=0)
        I = Invertd(
            keys=["pred"], transform=self.ds.transform, orig_keys=["image"]
        )  # watchout: use detach beforeharnd. make sure spacing are correct in preds
        A = Activationsd(keys="pred", softmax=True)
        D = AsDiscreted(keys=["pred"], argmax=True)  # ,threshold=0.5)
        C = ToCPUd(keys=["image", "pred"])
        Sa = SaveMultiChanneld(
            keys=["pred"],
            output_dir=self.output_folder,
            output_postfix="",
            separate_folder=False,
        )


        tfms = [Sq,  A, D,I, C]
        if self.debug == True:
            tfms = [Sq,I,Sa,A,D,C]
        C = Compose(tfms)
        self.postprocess_transforms = C

    def prepare_model(self):
        model = UNetTrainer.load_from_checkpoint(
            self.ckpt,
            project=self.project,
            dataset_params=self.dataset_params,
            strict=False,
        )

        fabric = Fabric(precision="16-mixed", devices=self.devices)
        self.model = fabric.setup(model)

    def predict(self):
        outputs = []
        self.model.eval()
        for i, batch in enumerate(self.pred_dl):
            with torch.no_grad():
                img_input = batch["image"]
                img_input = img_input.cuda()

                if "filename_or_obj" in img_input.meta.keys():
                    print("Processing: ", img_input.meta["filename_or_obj"])
                output_tensor = self.inferer(inputs=img_input, network=self.model)
                output_tensor = output_tensor[0]
                batch["pred"] = output_tensor
                batch["pred"].meta = batch["image"].meta.copy()
                outputs.append(batch)
        return outputs

    def postprocess(self, preds):
        out_final = []
        for batch in preds:
            tmp = self.postprocess_transforms(batch)
            out_final.append(tmp)
        return out_final

    @property
    def output_folder(self):
        run_name = listify(self.run_name)
        fldr = "_".join(run_name)
        fldr = self.project.predictions_folder / fldr
        return fldr


# %%

if __name__ == "__main__":
    # ... run your application ...
    from fran.utils.common import *

    proj = Project(project_title="nodes")
    run_ps = ["LITS-702"]
# %%
    proj = Project(project_title="totalseg")

    run_ps = ["LITS-827"]
# %% run_name = run_ps[0]

# %%
    img_fn = "/s/xnat_shadow/nodes/imgs_no_mask/nodes_4_20201024_CAP1p5mm_thick.nii.gz"

    fldr_lidc= Path("/s/xnat_shadow/lidc2/images/")
    imgs_lidc = list(fldr_lidc.glob("*"))
    img_fns = [img_fn]
    input_data = [{"image": im_fn} for im_fn in img_fns]
    debug = False

# %%
    P = BaseInferer(proj, run_ps[0], debug=debug)

# %%
    preds = P.run(imgs_lidc,chunksize=3)
# %%
    imgs = img_fns
    P.prepare_data(imgs)
    P.create_postprocess_transforms()
    preds = P.predict()

# %%
    Sq = SqueezeDimd(keys=["pred"], dim=0)

    A = Activationsd(keys="pred", softmax=True)
    D = AsDiscreted(keys=["pred"], argmax=True)  # ,threshold=0.5)
    C = ToCPUd(keys=["image", "pred"])



    I = Invertd(
        keys=["pred"], transform=P.ds.transform, orig_keys=["image"]
    )  # watchout: use detach beforeharnd. make sure spacing are correct in preds
    tfms = [Sq,  A,I ,D, C]
    A = Activationsd(keys="pred", softmax=True)
    pred = preds[0]
# %%
    pred['pred'].shape
# %%
    pred = Sq(pred)
    pred['pred'].shape
    pred = A(pred)
    pred['pred'].shape
    pred = D(pred)
    pred['pred'].shape
    pred = I(pred)
    pred['pred'].shape
    pred = C(pred)
# %%
    pred = P.ds.S.inverse(pred)
    tmp = preds[0]
    tmp["pred"] = tmp["pred"].detach()
# %%
    pred = preds[0]["pred"][0]
    pred = pred.detach().to('cpu')
    a = P.ds[0]
    im = a["image"]
    im = im[0]
    dici = {"image": im, "pred": im}
    ImageMaskViewer([im, pred['pred'][0]])

    output = P.postprocess(preds)
# %%
    lm_fn = "/s/fran_storage/predictions/nodes/LITS-702/nodes_4_20201024_CAP1p5mm_thick.nii.gz"
    lm = sitk.ReadImage(lm_fn)
    lm.GetSize()
    view_sitk(lm, lm)
    # preds = P.decollate(preds)
    # output= P.postprocess(preds)
# %%

    out_final = []
    batch = preds[0]
    for batch in preds:
        ou = out2[0]
        for ou in out2:
            tmp = P.postprocess_transforms(ou)
            out_final.append(tmp)
# %%

    C = ToCPUd(keys=["image", "pred"])
    Sq = SqueezeDimd(keys=["pred"], dim=0)
    batch = C(batch)
    batch = Sq(batch)
    batch["pred"].meta

    I = Invertd(keys=["pred"], transform=P.ds.transform, orig_keys=["image"])
    I(tmp)
    tmp = I(batch)
    tmp = P.postprocess_transforms(batch)
    out_final.append(tmp)
# %%
    data = P.ds[0]
    P.ds.transform.inverse(data)
    P.ds.transform.inverse(batch)
# %%
    I = Invertd(keys=["pred"], transform=P.ds.transform, orig_keys=["image"])
    I = Invertd(keys=["image"], transform=P.ds.transform, orig_keys=["image"])
    pp = preds[0].copy()
    print(pp.keys())
    pp["pred"] = pp["pred"][0:1, 0]

    pp["pred"].shape
    pp["pred"].meta
    a = I(pp)

    dici = {"image": img_fn}
    # L=LoadImaged(keys=['image'],image_only=True,ensure_channel_first=False,simple_keys=True)
    L = LoadSITKd(
        keys=["image"], image_only=True, ensure_channel_first=False, simple_keys=True
    )
    S = Spacingd(keys=["image"], pixdim=P.ds.dataset_params["spacing"])
    tfms = [L, S]
    Co = Compose(tfms)

    dd = L(dici)
    dda = S(dd)

# %%
    dd = Co(dici)
# %%
    Co.inverse(dd)
# %%
