# %%
import itertools as il
import ipdb

from fran.utils.string import ast_literal_eval
tr = ipdb.set_trace

from fran.managers.unet import UNetManager
from pathlib import Path
from fran.trainers import checkpoint_from_model_id

import itk
import numpy as np
import SimpleITK as sitk
import torch
from fastcore.all import listify, store_attr
from fastcore.foundation import GetAttr
from lightning.fabric import Fabric
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.itk_torch_bridge import itk_image_to_metatensor as itm
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.post.dictionary import Activationsd, AsDiscreted
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import (EnsureChannelFirstd,
                                                 SqueezeDimd)

from fran.data.dataset import NormaliseClipd
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import KeepLargestConnectedComponentWithMetad, SaveMultiChanneld, ToCPUd
from fran.transforms.spatialtransforms import ResizeToMetaSpatialShaped
from fran.utils.dictopts import DictToAttr, fix_ast
from fran.utils.helpers import slice_list
from fran.utils.imageviewers import ImageMaskViewer



def get_patch_spacing( run_name):
        ckpt = checkpoint_from_model_id(run_name)
        dic1 = torch.load(ckpt)
        config = dic1['datamodule_hyper_parameters']['config']
        spacing= config['plan'].get("spacing")
        if spacing is None:
            src_plan  = config['plan']['source_plan']
            src_plan = config[src_plan]
            spacing = src_plan['spacing']
        spacing= ast_literal_eval(spacing)
        return spacing


def list_to_chunks(input_list: list, chunksize: int):
    n_lists = int(np.ceil(len(input_list) / chunksize))

    fpl = int(len(input_list) / n_lists)
    inds = [[fpl * x, fpl * (x + 1)] for x in range(n_lists - 1)]
    inds.append([fpl * (n_lists - 1), None])

    chunks = list(il.starmap(slice_list, zip([input_list] * n_lists, inds)))
    return chunks


def load_params(model_id):
    ckpt = checkpoint_from_model_id(model_id)
    dic_tmp = torch.load(ckpt, map_location="cpu")
    dic_relevant = dic_tmp["datamodule_hyper_parameters"]
    # dic_tmp['datamodule_hyper_parameters']['plan']['spacing']= '.8,.8,1.5'# = fix_ast(dic_tmp, keys=['spacing'])
    # dic_relevant['plan']=fix_ast(dic_relevant['plan'], keys = ['spacing'])# = fix_ast(dic_tmp, keys=['spacing'])
    return dic_relevant


class BaseInferer(GetAttr, DictToAttr):
    def __init__(
        self,
        project,
        run_name,
        ckpt=None,
        state_dict=None,
        params=None,
        bs=8,
        patch_overlap=0.25,
        mode="gaussian",
        devices=[0],
        safe_mode=False,
        # reader=None,
        save_channels=True,
        save=True,
        k_largest=None,  # assign a number if there are organs involved
    ):
        """
        data is a dataset from Ensemble in this base class
        params: should be a dict with 2 keys: dataset_params and plan.
        """
        torch.cuda.empty_cache()

        store_attr("project,run_name,devices,save_channels, save,safe_mode, k_largest")
        if ckpt is None:
            self.ckpt = checkpoint_from_model_id(run_name)
        else:
            self.ckpt = ckpt
        if params is None:
            self.params = load_params(run_name)
        else:
            self.params = params
        self.plan = fix_ast(self.params['config']['plan'],["spacing"])

        if safe_mode == True:
            print(
                "================================================================\nSafe mode is on. Stitching will be on CPU. Slower speed expected\n================================================="
            )
            bs = 1
            stitch_device = "cpu"
        else:
            stitch_device = "cuda"
        self.inferer = SlidingWindowInferer(
            roi_size=self.params['config']["plan"]["patch_size"],
            sw_batch_size=bs,
            overlap=patch_overlap,
            mode=mode,
            progress=True,
            device=stitch_device,
        )
        self.tfms="ESN"

    def setup(self):
        if not hasattr(self, "model"):
            self.create_transforms()
            self.prepare_model()
        # self.create_postprocess_transforms()

    def run(self, imgs,chunksize=12, overwrite=True):
        """
        chunksize is necessary in large lists to manage system ram
        """
        self.setup()

        if overwrite == False and (
            isinstance(imgs[0], str) or isinstance(imgs[0], Path)
        ):
            imgs = self.filter_existing_preds(imgs)
        imgs = list_to_chunks(imgs, chunksize)
        for imgs_sublist in imgs:
            output = self.process_imgs_sublist(imgs_sublist)
        return output

    def filter_existing_preds(self, imgs):
        imgs = [Path(img) for img in imgs]
        print(
            "Filtering existing predictions\nNumber of images provided: {}".format(
                len(imgs)
            )
        )
        out_fns = [self.output_folder / img.name for img in imgs]
        to_do = [not fn.exists() for fn in out_fns]
        imgs = list(il.compress(imgs, to_do))
        print("Number of images remaining to be predicted: {}".format(len(imgs)))
        return imgs

    def process_imgs_sublist(self, imgs_sublist):
        data = self.load_images(imgs_sublist)
        self.prepare_data(data,self.tfms,  collate_fn=None)
        preds = self.predict()
        output = self.postprocess(preds)
        if self.save == True:
            self.save_pred(output)
        if self.safe_mode == True:
            self.reset()
        return output

    def reset(self):
        torch.cuda.empty_cache()
        # self.setup()

    def create_transforms(self):
        # single letter name is must for each tfm to use with set_transforms
        spacing = get_patch_spacing(self.run_name)
        self.L = LoadSITKd(
            keys=["image"],
            image_only=True,
            ensure_channel_first=False,
            simple_keys=True,
        )
        self.E = EnsureChannelFirstd(
            keys=["image"], channel_dim="no_channel"
        )  

        self.S = Spacingd(keys=["image"], pixdim= spacing)
        self.N = NormaliseClipd(
            keys=["image"],
            clip_range=self.params["dataset_params"]["intensity_clip_range"],
            mean=self.params["dataset_params"]["mean_fg"],
            std=self.params["dataset_params"]["std_fg"],
        )

        self.O = Orientationd(keys=["image"], axcodes="RPS")  # nOTE RPS

    def __repr__(self) -> str:
        return str(self.__class__)

    def set_transforms(self, tfms: str = ""):
        tfms_final = []
        for tfm in tfms:
            tfms_final.append(getattr(self, tfm))
        # if self.input_type == "files":
        #     tfms_final.insert(0, self.L)
        transform = Compose(tfms_final)
        return transform

    def load_images(self, data):
        """
        data can be filenames or images. InferenceDatasetNii will resolve data type and add LoadImaged if it is a filename
        """

        Loader = LoadSITKd(["image"])
        data = self.parse_input(data)
        data = [Loader(d) for d in data]
        return data

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
                    pass
                    # do nothing
                    # dat = ConvertSimpleItkImageToItkImage(dat, itk.F)
                elif isinstance(dat, itk.Image):
                    dat = itm(dat)
                else:
                    tr()
                dat = [dat]
            imgs_out.extend(dat)
        imgs_out = [{"image": img} for img in imgs_out]
        return imgs_out

    def prepare_data(self, data, tfms, collate_fn=None):
        """
        data: list
        """

        # if len(data)<4:
        nw, bs = 0, 1  # Slicer bugs out
        # else:
        # nw,bs = 12,12
        transform = self.set_transforms(tfms)
        # self.ds = InferenceDatasetNii(self.project, imgs, self.dataset_params)
        self.ds = Dataset(data=data, transform=transform)
        self.pred_dl = DataLoader(
            self.ds, num_workers=nw, batch_size=bs, collate_fn=collate_fn
        )

    def save_pred(self, preds):
        S = SaveImaged(
            keys=["pred"],
            output_dir=self.output_folder,
            output_postfix="",
            separate_folder=False,
        )
        for pp in preds:
            S(pp)

    def create_postprocess_transforms(self, preprocess_transform):
        Sq = SqueezeDimd(keys=["pred", "image"], dim=0)
        # below is expensive on large number of channels and on discrete data I am unsure if it uses nearest neighbours
        # I = Invertd(
        #     keys=["pred"], transform=self.ds.transform, orig_keys=["image"]
        # )  # watchout: use detach beforeharnd. make sure spacing are correct in preds
        A = Activationsd(keys="pred", softmax=True)
        D = AsDiscreted(keys=["pred"], argmax=True)  # ,threshold=0.5)
        U = ToCPUd(keys=["image", "pred"])
        Sa = SaveMultiChanneld(
            keys=["pred"],
            output_dir=self.output_folder,
            output_postfix="",
            separate_folder=False,
        )
        I = ResizeToMetaSpatialShaped(keys=["pred"], mode="nearest")
       
        out_final = []
        if self.save_channels == True:
            tfms = [Sq, Sa, A, D, I]
        else:
            tfms = [Sq, A, D, I]
        if self.k_largest:
            K = KeepLargestConnectedComponentWithMetad(
                    keys=["pred"], independent=False, num_components=self.k_largest
                )  # label=1 is the organ
            tfms.insert(-1, K)
        if self.safe_mode == True:
            tfms.insert(0, U)
        else:
            tfms.append(U)
        C = Compose(tfms)
        self.postprocess_transforms = C

    def prepare_model(self):
        device_id = self.devices[0]
        device = torch.device(f"cuda:{device_id}")
        model = UNetManager.load_from_checkpoint(
            self.ckpt,
            plan=self.plan,
            project=self.project,
            dataset_params=self.params['dataset_params'],
            strict=False,
            map_location=device,
        )
        fabric = Fabric(precision="16-mixed", devices=self.devices, accelerator="gpu")

        self.model = fabric.setup(model)

    def predict(self):
        outputs = []
        self.model.eval()
        with torch.inference_mode():
            for i, batch in enumerate(self.pred_dl):
                with torch.no_grad():
                    batch = self.predict_inner(batch)
                    outputs.append(batch)
        return outputs

    def predict_inner(self,batch):
                    img_input = batch["image"]
                    img_input = img_input.cuda()
                    if "filename_or_obj" in img_input.meta.keys():
                        print("Processing: ", img_input.meta["filename_or_obj"])
                    output_tensor = self.inferer(inputs=img_input, network=self.model)
                    output_tensor = output_tensor[0]
                    batch["pred"] = output_tensor
                    batch["pred"].meta = batch["image"].meta.copy()
                    return batch
    def postprocess(self, preds):
        self.create_postprocess_transforms(self.ds.transform)
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
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------

    from fran.utils.common import *


    from fran.managers.datasource import _DS
    from fran.managers.project import Project
    D = _DS()
    proj = Project(project_title="totalseg")
    run_tot= ["LITS-860"]
    safe_mode = False

    proj_nodes = Project(project_title="nodes")
    run_nodes= ["LITS-966"]

    proj_litsmc= Project(project_title="litsmc")
    fldr_crc = Path("/s/xnat_shadow/crc/images")
    imgs_crc = list(fldr_crc.glob("*"))

    fldr_lidc = Path("/s/xnat_shadow/lidc2/images/")
    imgs_lidc = list(fldr_lidc.glob("*"))
    fldr_nodes = Path("/s/xnat_shadow/nodes/images_pending/")
    img_nodes = list(fldr_nodes.glob("*"))
    fldr_litsmc = Path(D.litq['folder']),Path( D.drli['folder']), Path(D.lits['folder']), Path(D.litqsmall['folder'])
    imgs_litsmc = [list((fld/("images")).glob("*")) for fld in fldr_litsmc]
    imgs_litsmc = list(il.chain.from_iterable(imgs_litsmc))


    # img_nodes = ["/s/xnat_shadow/nodes/images_pending/nodes_24_20200813_ChestAbdoC1p5SoftTissue.nii.gz"]

# %%
#SECTION:-------------------- NODES--------------------------------------------------------------------------------------

    safe_mode = False
    bs = 5
    save_channels = False
    overwrite = False
    devices = [0]
    N = BaseInferer(
        proj_nodes,
        run_nodes[0],
        save_channels=save_channels,
        safe_mode=safe_mode,
        devices=devices,
    )

# %%
    preds = N.run(img_nodes, chunksize=1,overwrite=overwrite)
# %%
    data = P.ds.data[0]
# %%

# %%
#SECTION:-------------------- LITSMC--------------------------------------------------------------------------------------

    run_litsmc= ["LITS-1007"]
    run_litsmc= ["LITS-999"]
    safe_mode = False
    bs = 5
    save_channels = False
    overwrite = True
    devices = [1]
    L = BaseInferer(
        proj_litsmc,
        run_litsmc[0],
   
        save_channels=save_channels,
        safe_mode=safe_mode,
        devices=devices,
        k_largest=1,
    )

# %%
    preds = L.run(imgs_crc, chunksize=1,     overwrite=overwrite,)
# %%
    data = P.ds.data[0]
# %%
#SECTION:-------------------- TOTALSEG--------------------------------------------------------------------------------------
    
    save_channels = False
    safe_mode = True
    bs = 4
    overwrite = False
    devices = [0]
    

    T = BaseInferer(
        proj,
        run_tot[0],
        save_channels=save_channels,
        safe_mode=safe_mode,
        devices=devices
    )
# %%

    case_id = "crc_CRC089"
    imgs_crc = [fn for fn in imgs_crc if case_id in fn.name]
    preds = T.run(imgs_crc, chunksize=1)

# %%

    P.setup()
    imgs_sublist = img_fns
    data = P.load_images(imgs_sublist)
    P.prepare_data(data, tfms="ESN", collate_fn=None)
    preds = P.predict()
    # preds = P.decollate(preds)

    pred = preds[0]
    pp = preds[0]["pred"][0]
    ImageMaskViewer([pp, pp])
# %%
# %%

    Sq = SqueezeDimd(keys=["pred"], dim=0)
    pred = Sq(pred)
    A = Activationsd(keys="pred", softmax=True)
    pred = A(pred)
    D = AsDiscreted(keys=["pred"], argmax=True)  # ,threshold=0.5)
    C = ToCPUd(keys=["image", "pred"])
    Sa = SaveMultiChanneld(
        keys=["pred"],
        output_dir=P.output_folder,
        output_postfix="",
        separate_folder=False,
    )

    if P.save_channels == True:
        I = ResizeToSisterTensor(
            keys=["pred"], key_spatial_shape="image", mode="nearest"
        )
        tfms = [Sq, A, D, I, C]
    else:
        I = ResizeToSisterTensor(
            keys=["pred"], key_spatial_shape="image", mode="trilinear"
        )
    Sq = SqueezeDimd(keys=["pred"], dim=0)

    ca = Compose([Sq, A, D])
    pred = preds[0]

    pr = ca(pred)
    pr2 = I(pr)
    p = pr["pred"][0].cpu()
    img = pr["image"][0, 0]
# %%
# %%


    ImageMaskViewer([img, p])
# %%
