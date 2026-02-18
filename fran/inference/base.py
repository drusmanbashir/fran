# %%
import os
from utilz.cprint import cprint

from fran.inference.helpers import infer_project
from fran.managers.nep import download_neptune_checkpoint

os.environ["TORCHDYNAMO_DISABLE"] = "1"  # set as early as possible in the process

import itertools as il

import ipdb
import torch
import torch._dynamo as dynamo
from tqdm.auto import tqdm as pbar
from utilz.stringz import ast_literal_eval, headline

from fran.managers import Project

tr = ipdb.set_trace

from pathlib import Path
from typing import List, Optional

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
from monai.transforms.post.dictionary import (Activationsd, AsDiscreteD,
                                              AsDiscreted, Invertd)
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import (CastToTyped,
                                                 EnsureChannelFirstd,
                                                 SqueezeDimd)
from utilz.dictopts import DictToAttr, fix_ast
from utilz.helpers import slice_list

from fran.data.dataset import NormaliseClipd
from fran.managers.unet import UNetManager
from fran.trainers import checkpoint_from_model_id
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import (
    KeepLargestConnectedComponentWithMetad, SaveMultiChanneld,
    SqueezeListofListsd, ToCPUd)
from fran.transforms.spatialtransforms import ResizeToMetaSpatialShaped


# CODE: consider move util functions outside the class file  (see #2)
def get_device(devices: Optional[List[int]] = None) -> tuple:
    """
    Determine the appropriate device(s) based on CUDA availability.

    Args:
        devices: List of GPU device indices to use. If None, uses [0] if CUDA is available

    Returns:
        tuple: (device_ids, device, accelerator)
            - device_ids: List of device indices to use
            - device: torch.device object
            - accelerator: String indicating 'gpu' or 'cpu'
    """
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return [], torch.device("cpu"), "cpu"

    if devices is None:
        devices = [0]

    try:
        device_id = devices[0]
        device = torch.device(f"cuda:{device_id}")
        # Test if device is actually available
        torch.cuda.get_device_properties(device_id)
        return devices, device, "gpu"
    except (RuntimeError, AssertionError) as e:
        print(f"Error accessing CUDA device {devices}: {e}")
        print("Falling back to CPU")
        return [], torch.device("cpu"), "cpu"


def get_patch_spacing(run_name):
    ckpt = checkpoint_from_model_id(run_name)
    dic1 = torch.load(ckpt, weights_only=False)
    config = dic1["datamodule_hyper_parameters"]["configs"]
    spacing = config["plan_train"].get("spacing")
    if spacing is None:
        mode = config["plan_train"]["mode"] 
        if  mode == "whole":
            spacing = [.8,1.5,1.5]
            cprint("Mode is {0}. Using dummy spacing: {1}".format(mode, spacing),color= "red",bold=True,bg="yellow")
            print(spacing)
        else:
            raise NotImplementedError

        # src_plan = config["plan_train"]["source_plan"]
        # src_plan = config[src_plan]
        # spacing = src_plan["spacing"]
    spacing = ast_literal_eval(spacing)
    return spacing


def list_to_chunks(input_list: list, chunksize: int):
    if len(input_list) < chunksize:
        print("List too small, setting chunksize to len(list)")
        chunksize = np.minimum(len(input_list), chunksize)
    # assert len(input_list) >= chunksize, "Print list size too small: {}".format(
    #     len(input_list)
    # )
    n_lists = int(np.ceil(len(input_list) / chunksize))

    fpl = int(len(input_list) / n_lists)
    inds = [[fpl * x, fpl * (x + 1)] for x in range(n_lists - 1)]
    inds.append([fpl * (n_lists - 1), None])

    chunks = list(il.starmap(slice_list, zip([input_list] * n_lists, inds)))
    return chunks


def load_params(model_id):
    ckpt = checkpoint_from_model_id(model_id)
    dic_tmp = torch.load(ckpt, map_location="cpu", weights_only=False)
    dic_relevant = dic_tmp["datamodule_hyper_parameters"]
    return dic_relevant


def parse_input(imgs_inp):
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
            dat = Path(dat)
            if dat.is_dir():
                dat = list(dat.glob("*"))
            else:
                dat = [dat]
        else:
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


def load_images(data):
    """
    data can be filenames or images. InferenceDatasetNii will resolve data type and add LoadImaged if it is a filename
    """

    Loader = LoadSITKd(["image"])
    data = parse_input(data)
    data = [Loader(d) for d in data]
    return data


def filter_existing_files(files, target_folder):
    files = [Path(img) for img in files]
    print(
        "Filtering existing predictions\nNumber of images provided: {}".format(
            len(files)
        )
    )
    out_fns = [target_folder / img.name for img in files]
    to_do = [not fn.exists() for fn in out_fns]
    files = list(il.compress(files, to_do))
    print(
        "Number of images not found in folder {0}:  {1}".format(
            target_folder, len(files)
        )
    )
    return files


class BaseInferer(GetAttr, DictToAttr):
    def __init__(
        self,
        run_name,
        patch_overlap: float,
        project_title=None,
        ckpt=None,
        state_dict=None,
        params=None,
        bs=8,
        mode="constant",
        devices=[0],
        safe_mode=False,
        # reader=None,
        save_channels=False,
        save=True,
        k_largest=None,  # assign a number if there are organs involved
        debug=False,
    ):
        """
        BaseInferer applies the dataset spacing, normalization and then patch_size to use a sliding window inference over the resulting image
        data is a dataset from Ensemble in this base class
        params: should be a dict with 2 keys: dataset_params and plan.
        """
        torch.cuda.empty_cache()
        if not torch.cuda.is_available():
            print("CUDA not available. All processes will be on CPU.")
            safe_mode = True

        store_attr("debug,run_name,devices,save_channels, save,safe_mode, k_largest")
        if ckpt is None:
            self.ckpt = checkpoint_from_model_id(run_name)
        else:
            self.ckpt = ckpt
        if params is None:
            self.params = load_params(run_name)
        else:
            self.params = params
        assert not (
            safe_mode == True and save_channels == True
        ), "Safe mode cannot be used with save_channels"
        self.plan = fix_ast(self.params["configs"]["plan_train"], ["spacing"])
        self.check_plan_compatibility()
        self.dataset_params = self.params["configs"]["dataset_params"]
        
        if project_title is not None:
            from fran.managers import Project
            self.project = Project(project_title=project_title)
        else:
            self.project = infer_project(self.params)
        sw_device = "cuda"
        if safe_mode == True:
            cprint(
                "================================================================\nSafe mode is on. Stitching will be on CPU. Slower speed expected\n================================================="
            ,bg="red"
            )
            cprint("Patch Overlap: {}".format(patch_overlap),bg="red")
            bs = 1
            mode = "constant"
            device = "cpu"
            # Only set patch_overlap to 0.05 if not explicitly provided

        else:
            device = "cuda"
        self.inferer = SlidingWindowInferer(
            roi_size=self.params["configs"]["plan_train"]["patch_size"],
            sw_batch_size=bs,
            overlap=patch_overlap,
            mode=mode,
            progress=True,
            sw_device=sw_device,
            device=device,
        )
        self.safe_mode = safe_mode

    def create_and_set_postprocess_transforms(self):
        self.create_postprocess_transforms(self.ds.transform)
        self.set_postprocess_tfms_keys()
        self.set_postprocess_transforms()


    def create_and_set_preprocess_transforms(self):
            self.create_preprocess_transforms()
            self.set_preprocess_tfms_keys()
            self.set_preprocess_transforms()


    def set_preprocess_tfms_keys(self):
        self.preprocess_tfms_keys = "E,S,N"

    def set_postprocess_tfms_keys(self):
        if self.safe_mode == False:
            self.postprocess_tfms_keys = "Sq,A,Re,Int"
        else:
            self.postprocess_tfms_keys = "Sq,Re"
        # self.postprocess_tfms_keys = "Sq,A,Re,Int"
        if self.save_channels == True:
            self.postprocess_tfms_keys += ",SaM"
        if self.k_largest is not None:
            self.postprocess_tfms_keys += ",K"
        if self.save == True:
            self.postprocess_tfms_keys += ",Sav"

    def set_postprocess_transforms(self):
        self.postprocess_transforms = self.tfms_from_dict(
            self.postprocess_tfms_keys, self.postprocess_transforms_dict
        )
        self.postprocess_compose = Compose(self.postprocess_transforms)

    def set_preprocess_transforms(self):
        transform = self.tfms_from_dict(
            self.preprocess_tfms_keys, self.preprocess_transforms_dict
        )
        self.preprocess_compose= Compose(transform)

    #             output = self.postprocess_iterate(preds)
    # return output
    def postprocess_iterate(self, batch):
        if isinstance(batch, list):
            batch = batch[0]
        bbox = batch.get("bounding_box")
        if bbox and isinstance(bbox[0], list):
            bbox = bbox[0]
        batch["bounding_box"] = bbox
        for tfm in self.postprocess_transforms:
            headline(tfm)
            tr()
            batch = tfm(batch)
        return batch

    def check_plan_compatibility(self):
        assert (
            self.plan["mode"] == "source"
        ), "This inferer only works with source plans"

    def setup(self):
        if not hasattr(self, "model"):
            self.create_and_set_preprocess_transforms()
            self.prepare_model()

    def maybe_filter_images(self, imgs, overwrite=False):
        imgs = listify(imgs)
        if overwrite == False and (
            isinstance(imgs[0], str) or isinstance(imgs[0], Path)
        ):
            imgs = filter_existing_files(imgs, self.output_folder)
        else:
            pass

        if len(imgs) == 0:
            print("No images to process after filtering")
            raise SystemExit("Stopping execution - no images remain")
        return imgs

    def run(self, imgs: list, chunksize=12, overwrite=False):
        """
        imgs can be a list comprising any of filenames, folder, or images (sitk or itk)
        chunksize is necessary in large lists to manage system ram
        """
        self.setup()
        imgs = self.maybe_filter_images(imgs, overwrite)
        imgs = list_to_chunks(imgs, chunksize)
        for imgs_sublist in imgs:
            output = self.process_imgs_sublist(imgs_sublist)
        return output

    def postprocess(self, preds):
        if self.debug == False:
            output = self.postprocess_compose(preds)
        else:
            output = self.postprocess_iterate(preds)
        return output

    def process_imgs_sublist(self, imgs_sublist):
        data = load_images(imgs_sublist)
        self.prepare_data(data,  collate_fn=None)
        self.create_and_set_postprocess_transforms()

        outputs = []
        for batch in self.predict():
            batch = self.postprocess(batch)
            outputs.append(batch)

        if self.safe_mode:
            self.reset()
            outputs.append(None)

        return outputs

    def reset(self):
        torch.cuda.empty_cache()
        # self.setup()

    def __repr__(self) -> str:
        return str(self.__class__)

    def prepare_data(self, data, collate_fn=None):
        """
        data: list
        """

        nw, bs = 0, 1  # Slicer bugs out
        self.ds = Dataset(data=data, transform=self.preprocess_compose)
        self.pred_dl = DataLoader(
            self.ds, num_workers=nw, batch_size=bs, collate_fn=collate_fn
        )

    def create_preprocess_transforms(self):
        spacing = get_patch_spacing(self.run_name)

        self.preprocess_transforms_dict = {
            "L": LoadSITKd(
                keys=["image"],
                image_only=True,
                ensure_channel_first=False,
                simple_keys=True,
            ),
            "E": EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            "S": Spacingd(keys=["image"], pixdim=spacing),
            "N": NormaliseClipd(
                keys=["image"],
                clip_range=self.dataset_params["intensity_clip_range"],
                mean=self.dataset_params["mean_fg"],
                std=self.dataset_params["std_fg"],
            ),
            "O": Orientationd(keys=["image"], axcodes="RPS"),  # nOTE RPS
        }

        # Set individual attributes for backward compatibility
        for key, value in self.preprocess_transforms_dict.items():
            setattr(self, key, value)

    def create_postprocess_transforms(self, preprocess_transform):
        Sav = SaveImaged(
            keys=["pred"],
            output_dir=self.output_folder,
            output_postfix="",
            separate_folder=False,
            output_dtype=np.uint8,
        )
        Sq = SqueezeDimd(keys=["pred", "image"], dim=0)

        SqL = SqueezeListofListsd(keys=["bounding_box"])

        Re = ResizeToMetaSpatialShaped(keys=["pred"], mode="nearest")

        A = AsDiscreteD(argmax=True, keys=["pred"], dim=0)

        SaM = SaveMultiChanneld(
            keys=["pred"],
            output_dir=self.output_folder,
            output_postfix="",
            separate_folder=False,
        )
        Int = CastToTyped(keys=["pred"], dtype=np.uint8)

        K = KeepLargestConnectedComponentWithMetad(
            keys=["pred"], independent=False, num_components=self.k_largest
        )
        CPU = ToCPUd(keys=["image", "pred"])

        self.postprocess_transforms_dict = {
            "Sq": Sq,
            "SqL": SqL,
            "CPU": CPU,
            "Re": Re,
            "A": A,
            "SaM": SaM,
            "Int": Int,
            "K": K,
            "Sav": Sav,
        }

        for key, value in self.postprocess_transforms_dict.items():
            setattr(self, key, value)

    def tfms_from_dict(self, keys: str, tfms_dict):
        keys = keys.split(",")
        tfms = []
        for key in keys:
            tfm = tfms_dict[key]
            tfms.append(tfm)
        return tfms

    # def set_transforms(self, tfms: str = ""):
    #     tfms_final = []
    #     for tfm in tfms:
    #         if hasattr(self, 'transforms') and tfm in self.preprocess_transforms_dict:
    #             tfms_final.append(self.preprocess_transforms_dict[tfm])
    #         else:
    #             tfms_final.append(getattr(self, tfm))
    #     transform = Compose(tfms_final)
    #     return transform

    def prepare_model(self):
        if self.devices == "cpu":
            fabric_devices = "auto"
            accelerator = device = "cpu"
        else:
            fabric_devices = self.devices
            device_id = self.devices[0]
            device = torch.device(f"cuda:{device_id}")
            accelerator = "gpu"
        model = UNetManager.load_from_checkpoint(
            self.ckpt,
            plan=self.plan,
            project_title=self.project.project_title,
            dataset_params=self.dataset_params,
            strict=True,
            map_location=device,
        )
        model.eval()
        fabric = Fabric(
            precision="16-mixed", devices=fabric_devices, accelerator=accelerator
        )
        self.model = fabric.setup(model)

    def predict(self):
        # outputs = []
        self.model.eval()
        with torch.inference_mode():
            for i, batch in enumerate(
                pbar(self.pred_dl, desc="Processing predictions")
            ):
                # with torch.no_grad():
                batch = self.predict_inner(batch)
                yield batch

    @dynamo.disable()
    def _run_swi(self, img):
        # the only thing inside is the SlidingWindowInferer call
        return self.inferer(inputs=img, network=self.model)

    def predict_inner(self, batch):
        img = batch["image"]
        if self.devices != "cpu":
            img = img.cuda(non_blocking=True)
        if self.safe_mode:
            img = img.to("cpu")
        logits = self._run_swi(img)
        if isinstance(logits, tuple):
            logits = logits[0]  # model has deep supervision only 0 channel is needed
        # Collapse channels early; keep on same device
        if self.safe_mode == True:
            labels = torch.argmax(logits, dim=1, keepdim=True)
            labels = labels.to(torch.uint8)
            batch["pred"] = labels
            del logits
        else:
            batch["pred"] = logits
        batch["pred"].meta = batch["image"].meta.copy()
        return batch

    @property
    def output_folder(self):
        run_name = listify(self.run_name)
        fldr = "_".join(run_name)
        fldr = self.project.predictions_folder / fldr
        return fldr


class BaseInfererTorchScript(BaseInferer):
    def prepare_model(self):
        device_id = self.devices[0]
        device = torch.device(f"cuda:{device_id}")
        model = UNetManager.load_from_checkpoint(
            self.ckpt,
            plan=self.plan,
            project_title=self.project.project_title,
            dataset_params=self.dataset_params,
            strict=False,
            map_location=device,
        )
        model.eval()
        try:
            scripted_model = torch.jit.script(model)
            scripted_model.save("scripted_model.pt")
            print("Model successfully converted to TorchScript.")
        except Exception as e:
            print(f"Script conversion failed: {e}")
            # Attempt tracing as a fallback
            try:
                example_input = torch.rand(1, 1, 128, 128, 96)
                scripted_model = torch.jit.trace(model, example_input)
                scripted_model.save("traced_model.pt")
                print("Model successfully traced and converted to TorchScript.")
            except Exception as trace_e:
                print(f"Tracing failed: {trace_e}")
        scripted = model.to_torchscript()
        torch.jit.save(scripted, "tmp.pt")
        fabric = Fabric(precision="16-mixed", devices=self.devices, accelerator="gpu")
        self.model = fabric.setup(model)


if __name__ == "__main__":
# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>

    from fran.data.dataregistry import DS
    from fran.managers.project import Project
    from fran.utils.common import *

    D = DS
    proj = Project(project_title="totalseg")
    run_tot = ["LITS-860"]
    run_tot_big = "LITS-1437"
    run_whole_image = ["LITS-1088"]
    run_whole_image = ["LITS-1088"]
    run_nodes = ["LITS-1230"]
    run_nodes2 = ["LITS-1285"]
    run_nodes3 = ["LITS-1287"]
    safe_mode = False

    proj_litsmc = Project(project_title="litsmc")
    fldr_crc = Path("/s/xnat_shadow/crc/images")
    imgs_crc = list(fldr_crc.glob("*"))

    fldr_lidc = DS["lidc"].folder / ("images")
    imgs_lidc = list(fldr_lidc.glob("*"))
    fldr_nodes = Path("/s/xnat_shadow/nodes/images_pending/thin_slice/images")
    fldr_nodes2= Path("/s/xnat_shadow/nodes/images")
    img_nodes = list(fldr_nodes.glob("*"))
    img_nodes2 = list(fldr_nodes2.glob("*"))
    fldr_litsmc = (
        Path(D["litq"].folder),
        Path(D["drli"].folder),
        Path(D["lits"].folder),
        Path(D["litqsmall"].folder),
    )
    imgs_litsmc = [list((fld / ("images")).glob("*")) for fld in fldr_litsmc]
    imgs_litsmc = list(il.chain.from_iterable(imgs_litsmc))

    # img_nodes = ["/s/xnat_shadow/nodes/images_pending/nodes_24_20200813_ChestAbdoC1p5SoftTissue.nii.gz"]

# %%
# SECTION:-------------------- LITSMC-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>

    run_litsmc = ["LITS-1007"]
    run_litsmc = ["LITS-999"]
    safe_mode = False
    bs = 5
    save_channels = False
    overwrite = True
    devices = [1]
    L = BaseInferer(
        run_litsmc[0],
        save_channels=save_channels,
        safe_mode=safe_mode,
        devices=devices,
        k_largest=1,
    )

# %%
    preds = L.run(
        imgs_crc,
        chunksize=1,
        overwrite=overwrite,
    )
# %%
    data = P.ds.data[0]
# %%
# SECTION:-------------------- TOTALSEG-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>

    save_channels = False
    overwrite = False

    devices = [1]

# %%
    run = run_tot_big[0]
    debug_ = False
    safe_mode = True

# %%

    T = BaseInferer(
        run,
        patch_overlap=0.25,
        save_channels=save_channels,
        safe_mode=safe_mode,
        devices=devices,
        debug=debug_,
    )

# %%
    imf_fn = "/s/insync/datasets/bones/1/2 Source.nrrd"
    # preds = T.run(imgs_crc, chunksize=2, overwrite=overwrite)
    preds = T.run(imf_fn, chunksize=2, overwrite=overwrite)
# %%
    preds = T.run(img_nodes, chunksize=2, overwrite=overwrite)
    preds = T.run(img_nodes2, chunksize=2, overwrite=overwrite)
# %%
    print(getattr(T.N, "_torchdynamo_inline", None))
    case_id = "crc_CRC089"
    imgs_crc = [fn for fn in imgs_crc if case_id in fn.name]
    import torch._dynamo

    print(torch._dynamo.is_compiled_fn(T.N.forward))

    # datamodule_ %%
# %%
# %%
# SECTION:--------------------  NODES-------------------------------------------------------------------------------------- <CR> <CR>
    run = run_nodes2[0]
    run = run_nodes3[0]
    run = run_nodes[0]

    debug_ = False

    save_channels = False
    safe_mode = True
    bs = 1
    overwrite = True
    devices = [1]
    save_channels = False

    T = BaseInferer(
        run,
        save_channels=save_channels,
        safe_mode=safe_mode,
        devices=devices,
        debug=debug_,
    )

# %%
    preds = T.run(img_nodes[1], chunksize=2, overwrite=overwrite)
    preds[0]['pred'].meta
# %%
# SECTION:-------------------- TROUBLESHOOTING-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
    overwrite = True
    T.setup()
    imgs = img_nodes

# %%

    patch_overlap = 0.0
    mode = "constant"
    sw_device = "cuda"
    device = "cpu"
    T.inferer = SlidingWindowInferer(
        roi_size=T.params["configs"]["plan_train"]["patch_size"],
        sw_batch_size=bs,
        overlap=patch_overlap,
        mode=mode,
        progress=True,
        sw_device=sw_device,
        device=device,
    )
# %%
    if overwrite == False and (isinstance(imgs[0], str) or isinstance(imgs[0], Path)):
        imgs = T.filter_existing_preds(imgs)
    imgs = list_to_chunks(imgs, 4)
    # for imgs_sublist in imgs:
    #     output = T.process_imgs_sublist(imgs_sublist)
    imgs_sublist = imgs[0]
# %%

    # output = T.process_imgs_sublist(imgs_sublist)
    # return output

    data = T.load_images(imgs_sublist)
    T.prepare_data(data,  collate_fn=None)
    # preds = T.predict()
# %%
    data = T.ds[0]
# %%
    T.create_postprocess_transforms(T.ds.transform)
    T.model.eval()
    print(getattr(T.N, "_torchdynamo_inline", None))
    iteri = iter(T.pred_dl)
    batch = next(iteri)
    batch = T.predict_inner(batch)
# %%

    # def predict_inner(T, batch):
    img = batch["image"]
    if T.devices != "cpu":
        img = img.cuda(non_blocking=True)
    if T.safe_mode:
        img = img.to("cpu")

    logits = T.inferer(inputs=img, network=T.model)  # [B,117,D,H,W]
# %%
    logits = logits[0]  # model has deep supervision only 0 channel is needed
    # Collapse channels early; keep on same device
# %%
    if T.safe_mode == True or T.save_channels == False:
        labels = torch.argmax(logits, dim=1, keepdim=True)
        labels = labels.to(torch.uint8)
        batch["pred"] = labels
        del logits
    else:
        batch["pred"] = logits
    batch["pred"].meta = batch["image"].meta.copy()
# %%
    batch = T.postprocess_transforms(batch)
# %%
    #     if T.save == True:
    #         T.save_pred(batch)
    #     if T.safe_mode == True:
    #         T.reset()
    #         return None
    #     return batch
# %%

    """Process a subset of images using the data module"""
    T.pred_dl = T.data_module.setup(imgs_sublist)
    preds = T.predict()
    # output = T.postprocess(preds)
# %%
    T.create_postprocess_transforms()
    out_final = []
    # for batch in preds:
    batch = preds[0]
    tmp = T.postprocess_transforms(batch)
    out_final.append(tmp)
# %%
    if T.save:
        T.save_pred(output)
    if T.safe_mode:
        T.reset()
# %%

    T = En.Ps[0]
    Sq = SqueezeDimd(keys=["pred", "image"], dim=0)

    # below is expensive on large number of channels and on discrete data I am unsure if it uses nearest neighbours
    # I = Invertd(
    #     keys=["pred"], transform=T.ds.transform, orig_keys=["image"]
    # )  # watchout: use detach beforeharnd. make sure spacing are correct in preds
    A = Activationsd(keys="pred", softmax=True)
    D = AsDiscreted(keys=["pred"], argmax=True)  # ,threshold=0.5)
    U = ToCPUd(keys=["image", "pred"])
    Sa = SaveMultiChanneld(
        keys=["pred"],
        output_dir=T.output_folder,
        output_postfix="",
        separate_folder=False,
    )
    I = ResizeToMetaSpatialShaped(keys=["pred"], mode="nearest")

# %%
    out_final = []
    if T.save_channels == True:
        tfms = [Sq, Sa, A, D, I]
    else:
        tfms = [Sq, A, D, I]
    if T.k_largest:
        K = KeepLargestConnectedComponentWithMetad(
            keys=["pred"], independent=False, num_components=T.k_largest
        )  # label=1 is the organ
        tfms.insert(-1, K)
    if T.safe_mode == True:
        tfms.insert(0, U)
    else:
        tfms.append(U)
# %%
    pred = batch["pred"]
    pred.meta["spatial_shape"]
# %%
    batch = Sq(batch)
    batch = A(batch)
    batch = D(batch)
    batch = U(batch)
    batch = I(batch)

    x = torch.rand(1, 1, 128, 128, 96).to("cuda")
    output = T.model(x)
    [a.shape for a in output]

# %%
    if En.devices == "cpu":
        fabric_devices = "auto"
        accelerator = device = "cpu"
    else:
        fabric_devices = En.devices
        device_id = En.devices[0]
        device = torch.device(f"cuda:{device_id}")
        accelerator = "gpu"
    model = UNetManager.load_from_checkpoint(
        En.ckpt,
        plan=En.plan,
        project_title=En.project.project_title,
        dataset_params=En.dataset_params,
        strict=False,
        map_location=device,
    )
    model.eval()
    fabric = Fabric(
        precision="16-mixed", devices=fabric_devices, accelerator=accelerator
    )
    En.model = fabric.setup(model)

# %%
    tfms_keys = T.preprocess_tfms_keys
    transform = T.tfms_from_dict(tfms_keys, T.preprocess_transforms_dict)
# %%
    batch['pred'].shape
# %%
