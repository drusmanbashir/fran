# %%
import os
import pickle

from fran.inference.helpers import (
    filter_existing_files,
    get_patch_spacing,
    infer_project,
    list_to_chunks,
    load_images_nifti,
    load_params,
)
from utilz.cprint import cprint

os.environ["TORCHDYNAMO_DISABLE"] = "1"  # set as early as possible in the process

import ipdb
import torch
import torch._dynamo as dynamo
from fran.managers import Project
from tqdm.auto import tqdm as pbar
from utilz.stringz import headline

tr = ipdb.set_trace

from pathlib import Path

import numpy as np
import torch
from fastcore.all import listify
from fran.data.dataset import NormaliseClipd
from fran.managers.unet import UNetManager
from fran.trainers import checkpoint_from_model_id, write_normalized_ckpt
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import (
    KeepLargestConnectedComponentWithMetad,
    SaveMultiChanneld,
    SqueezeListofListsd,
    ToCPUd,
)
from fran.transforms.spatialtransforms import ResizeToMetaSpatialShaped
from lightning.fabric import Fabric
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.post.dictionary import AsDiscreteD
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import (
    CastToTyped,
    EnsureChannelFirstd,
    SqueezeDimd,
)
from utilz.dictopts import DictToAttr, fix_ast


def load_model_on_fabric(
    ModelClass,
    ckpt,
    devices=None,
    map_location=None,
    strict=True,
    precision="bf16-mixed",
    accelerator=None,
    fabric_kwargs=None,
    normalize_checkpoint_fn=None,
    normalize_state_dict_prefix="model._orig_mod",
    weights_only=False,
    **kwargs,
):
    resolved_map_location = "cuda" if map_location is None else map_location
    try:
        model = ModelClass.load_from_checkpoint(
            ckpt,
            map_location=resolved_map_location,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )
    except (RuntimeError, pickle.UnpicklingError) as exc:
        if normalize_checkpoint_fn is None:
            raise

        ckpt_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "state_dict" not in ckpt_dict:
            raise RuntimeError(
                f"Checkpoint load failed and checkpoint is missing 'state_dict': {ckpt}"
            ) from exc

        state_dict = ckpt_dict["state_dict"]
        if not any(
            key.startswith(normalize_state_dict_prefix) for key in state_dict.keys()
        ):
            raise

        ckpt = normalize_checkpoint_fn(ckpt)
        model = ModelClass.load_from_checkpoint(
            ckpt,
            map_location=resolved_map_location,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    model.eval()
    model._loaded_ckpt_path = ckpt

    if accelerator is None:
        accelerator = "cpu" if devices == "cpu" else "gpu"

    fabric_devices = "auto" if devices is None else devices
    fabric_options = {} if fabric_kwargs is None else dict(fabric_kwargs)
    fabric = Fabric(
        precision=precision,
        devices=fabric_devices,
        accelerator=accelerator,
        **fabric_options,
    )
    model = fabric.setup(model)
    return model, fabric


class BaseInferer(DictToAttr):
    def __init__(
        self,
        run_name,
        patch_overlap: float = 0.0,
        project_title=None,
        ckpt=None,
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
        keys_preproc="E,S,N",
        keys_postproc="Sq,A,Re,Int",
        model_manager=UNetManager,
    ):
        """
        BaseInferer applies the dataset spacing, normalization and then patch_size to use a sliding window inference over the resulting image
        data is a dataset from Ensemble in this base class
        params: should be a dict with 2 keys: dataset_params and plan.
        """
        torch.cuda.empty_cache()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Inference cannot proceed.")

        self.debug = debug
        self.run_name = run_name
        self.keys_preproc = keys_preproc
        self.keys_postproc = keys_postproc
        self.model_manager = model_manager
        self.keys_postproc_safe = "Sq,Re"
        self.devices = devices
        self.save_channels = save_channels
        self.save = save
        self.safe_mode = safe_mode
        self.k_largest = k_largest
        if ckpt is None:
            self.ckpt = checkpoint_from_model_id(run_name)
            cprint("Checkpoint: {}".format(self.ckpt), color="green")
        else:
            self.ckpt = ckpt
            cprint("No Checkpoint loaded. Using random weights", color="red")
        if params is None:
            self.params = load_params(run_name)
        else:
            self.params = params
        assert not (safe_mode == True and save_channels == True), (
            "Safe mode cannot be used with save_channels"
        )
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
                "================================================================\nSafe mode is on. Stitching will be on CPU. Slower speed expected\n=================================================",
                bg="red",
            )
            cprint("Patch Overlap: {}".format(patch_overlap), bg="red")
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

    def compute_loss(self, batch):
        pass

    def create_and_set_postprocess_transforms(self):
        self.create_postprocess_transforms(self.ds.transform)
        self.set_postprocess_tfms_keys()
        self.set_postprocess_transforms()

    def create_and_set_preprocess_transforms(self):
        self.create_preprocess_transforms()
        self.set_preprocess_tfms_keys()
        self.set_preprocess_transforms()

    def set_preprocess_tfms_keys(self):
        self.preprocess_tfms_keys = self.keys_preproc

    def set_postprocess_tfms_keys(self):
        if self.safe_mode == False:
            self.postprocess_tfms_keys = self.keys_postproc
        else:
            self.postprocess_tfms_keys = self.keys_postproc_safe
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
        self.preprocess_compose = Compose(transform)

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
        assert self.plan["mode"] == "source", (
            "This inferer only works with source plans"
        )

    def setup(self):
        if (
            getattr(self, "model", None) is None
            or getattr(self, "fabric", None) is None
        ):
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

    def run(self, data: list, chunksize=12, overwrite=False):
        """
        data: can be a list of images: comprising any of filenames, folder, or images (sitk or itk)
        chunksize is necessary in large lists to manage system ram
        """
        self.setup()
        data = self.maybe_filter_images(data, overwrite)
        data = list_to_chunks(data, chunksize)
        for imgs_sublist in data:
            output = self.process_data_sublist(imgs_sublist)
        return output

    def postprocess(self, preds):
        if self.debug == False:
            output = self.postprocess_compose(preds)
        else:
            output = self.postprocess_iterate(preds)
        return output

    def load_images(self, images):
        return load_images_nifti(images)

    def process_data_sublist(self, data_sublist):
        data = self.load_images(data_sublist)
        self.prepare_data(data, collate_fn=None)
        self.create_and_set_postprocess_transforms()

        outputs = []
        for batch in self.predict():
            batch = self.postprocess(batch)
            outputs.append(batch)
            self.compute_loss(batch)

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
        data: list of dicts
        """

        nw, bs = 0, 1  # Slicer bugs out
        self.ds = Dataset(data=data, transform=self.preprocess_compose)
        dl = DataLoader(self.ds, num_workers=nw, batch_size=bs, collate_fn=collate_fn)
        self.pred_dl = self.fabric.setup_dataloaders(dl)

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
            "O": Orientationd(keys=["image"], axcodes="RAS"),  # nOTE RAS
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
        keys = keys.replace(" ", "")
        keys_list = keys.split(",")
        tfms = []
        for key in keys_list:
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
            accelerator = device = "cpu"
        else:
            device_id = self.devices[0]
            device = torch.device(f"cuda:{device_id}")
            accelerator = "gpu"

        model, fabric = load_model_on_fabric(
            self.model_manager,
            self.ckpt,
            devices="cpu" if self.devices == "cpu" else self.devices,
            map_location=device,
            strict=True,
            accelerator=accelerator,
            normalize_checkpoint_fn=write_normalized_ckpt,
            plan=self.plan,
            project_title=self.project.project_title,
            dataset_params=self.dataset_params,
        )
        self.ckpt = getattr(model, "_loaded_ckpt_path", self.ckpt)
        self.fabric = fabric
        self.model = model

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
        img = batch["image"]  # already on correct device

        logits = self._run_swi(img)

        if isinstance(logits, tuple):
            logits = logits[0]

        if self.safe_mode:
            labels = torch.argmax(logits, dim=1, keepdim=True).to(torch.uint8)
            batch["pred"] = labels
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


if __name__ == "__main__":
# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>
    from fran.inference.common_vars import *

# %%
# SECTION:-------------------- LITSMC-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>

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
# SECTION:-------------------- TOTALSEG-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>

    save_channels = False
    overwrite = False
    overwrite = True

    devices = [0]

    patch_overlap = 0.5
# %%
    run = best_runs["totalseg"]["run_ids"][0]
    safe_mode = True

    debug_ = True
    debug_ = False
# %%

    T = BaseInferer(
        run,
        patch_overlap=patch_overlap,
        save_channels=save_channels,
        safe_mode=safe_mode,
        devices=devices,
        debug=debug_,
    )

# %%

    # preds = T.run(imgs_crc, chunksize=2, overwrite=overwrite)
    # imgs = kits_imgs
    # imgs = imgs_curvas

    imgs = nodes_imgs_training
    preds = T.run(imgs, chunksize=2, overwrite=overwrite)
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
# SECTION:--------------------  NODES-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>
    run = run_nodes2[0]
    run = run_nodes3[0]
    run_nodes = best_runs["nodes"]["run_ids"]
    run = run_nodes[0]


    debug_ = False

    save_channels = False
    safe_mode = True
    bs = 1
    overwrite = True
    devices = [0]
    save_channels = False

    T = BaseInferer(
        run,
        save_channels=save_channels,
        safe_mode=safe_mode,
        devices=devices,
        debug=debug_,
    )

# %%
    imgs = nodes_imgs_training
    preds = T.run(imgs, chunksize=2, overwrite=overwrite)
    preds[0]["pred"].meta
# %%
# SECTION:-------------------- TROUBLESHOOTING-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR> <CR> <CR>
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
    T.prepare_data(data, collate_fn=None)
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

    logits = T.inferer(inputs=img, network=T.model)  # [B,117,DS,H,W]
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
    DS = AsDiscreted(keys=["pred"], argmax=True)  # ,threshold=0.5)
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
        tfms = [Sq, Sa, A, DS, I]
    else:
        tfms = [Sq, A, DS, I]
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
    batch = DS(batch)
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
        precision="bf16-mixed", devices=fabric_devices, accelerator=accelerator
    )
    En.model = fabric.setup(model)

# %%
    tfms_keys = T.preprocess_tfms_keys
    transform = T.tfms_from_dict(tfms_keys, T.preprocess_transforms_dict)
# %%
    batch["pred"].shape
# %%
