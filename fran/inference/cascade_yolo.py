import json

from localiser.transforms.transforms import MapTransform
from monai.transforms.utility.dictionary import (
    CastToTyped,
    EnsureChannelFirstd,
    SqueezeDimd,
    )

from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from pathlib import Path
from fran.transforms.misc_transforms import DummyTransform
import ipdb
from localiser.inference.base import EnsureChannelFirstd, LocaliserInferer
from utilz.fileio import load_json
from utilz.imageviewers import ImageMaskViewer
tr = ipdb.set_trace
import torch

from utilz.cprint import cprint
from utilz.helpers import find_matching_fn, set_autoreload
from utilz.stringz import strip_extension

set_autoreload()
from fran.inference.cascade import CascadeInferer
from fran.inference.helpers import list_to_chunks, parse_input
from fran.transforms.imageio import LoadSITKd
from fran.transforms.spatialtransforms import CropByYolo
from localiser.transforms.tsl import TSLRegions

from localiser.utils.bbox_helpers import standardize_bboxes, yolo_bbox_to_slices

def add_channel_slices(bbox):
    return (slice(0,100),) + bbox

def orig_shape_from_meta(image):
    import nibabel as nib
    spatial_shape = image.meta['spatial_shape']
    axcodes = nib.aff2axcodes(image.meta['affine'].cpu().numpy())
    assert axcodes == ("R", "A", "S"), axcodes
    return spatial_shape

class StoreShape(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            data[key + "_orig_shape"] = data[key].shape[1:]
        return data


def collate_projections_with_shape(batch):
    from localiser.inference.base import collate_projections
    out = collate_projections(batch)
    out["image_orig_shape"] = [item["image_orig_shape"] for item in batch]
    return out

def _class_to_index(names):
    if isinstance(names, dict):
        return {str(v): int(k) for k, v in names.items()}
    return {str(name): idx for idx, name in enumerate(names)}


def _matching_classes(class_to_ind, selectors):
    return sorted(
        {
            class_idx
            for class_name, class_idx in class_to_ind.items()
            if any(selector in class_name for selector in selectors)
        }
    )


def localiser_regions_to_yolo_classes(yolo_specs, localiser_regions):
    class_to_ind = _class_to_index(yolo_specs["data"]["names"])
    if localiser_regions is None:
        return sorted(class_to_ind.values())

    if isinstance(localiser_regions, list | tuple | set):
        regions_list = [
            str(region).strip() for region in localiser_regions if str(region).strip()
        ]
        valid_regions = set(TSLRegions().regions)
        invalid = [region for region in regions_list if region not in valid_regions]
        if invalid:
            raise ValueError(f"Invalid TSL localiser_regions: {invalid}")
    else:
        regions = str(localiser_regions).replace(" ", "")
        regions_list = [r for r in regions.split(",") if r]

    return _matching_classes(class_to_ind, regions_list)


class LocaliserInfererPT(LocaliserInferer):
    # Oriented Loaded tensors expected
    keys_preproc = "E,O,St,Orig,W,S,P1,P2,PF,R,Rep,E2,N"
    def __init__(self, *args,  **kwargs):
        super().__init__(*args,  **kwargs)

    def filter_done_images(self, images, overwrite=False):
        if overwrite == True or isinstance(images[0], torch.Tensor):
            self.preprocess_transforms_dict["Lp"] = DummyTransform(
                keys=[self.image_key]
            )
            return images
        else:
            case_ids_done = self.done_case_ids()
            return [
                image
                for image in images
                if self.image_case_id(image) not in case_ids_done
            ]
    def create_preprocess_transforms(self):
        super().create_preprocess_transforms()
        self.preprocess_transforms_dict.update({
            "St": StoreShape(keys=[self.image_key]),
            })

    def prepare_data(self, data):
        nw = int(min(len(data) / 4, 6))
        transform = self.preprocess_iterate if self.debug else self.preprocess_compose
        self.ds = Dataset(data=data, transform=transform)
        self.pred_dl = DataLoader(
            self.ds,
            batch_size=self.bs,
            num_workers=nw,
            collate_fn=collate_projections_with_shape,
        )
        self.pred_dl = self.fabric.setup_dataloaders(
            self.pred_dl,
            move_to_device=False,
        )

    def package_preds(self, batch):
        outputs = super().package_preds(batch)
        for out, image_orig_shape in zip(outputs, batch["image_orig_shape"]):
            out["image_orig_shape"] = image_orig_shape
        return outputs
 
    # def load_images(self, images):
    #     return images

class CascadeInfererYOLO(CascadeInferer):
    YoloInferer = LocaliserInfererPT

    def __init__(
        self,
        localiser_regions: list[str],
        run_p="KITS23-SIRIG",
        devices=[0],
        safe_mode=False,
        patch_overlap=0.2,
        save_channels=False,
        save=False,
        k_largest=None,
        debug=False,
    ):
        self.localiser_regions = localiser_regions
        self.yolo_bs = 64
        self.yolo_specs = None
        self.classes = None
        self.cropper_yolo = CropByYolo(
            keys=["image"],
            lm_key=None,
            bbox_key="bbox",
            sanitize=False,
        )
        super().__init__(
            run_w=None,
            run_p=run_p,
            localiser_labels=localiser_regions,
            devices=devices,
            safe_mode=safe_mode,
            patch_overlap=patch_overlap,
            save_channels=save_channels,
            save=save,
            k_largest=k_largest,
            debug=debug,
        )

    def load_images(self, image_files: list[str | Path]):
        loader = LoadSITKd(["image"])
        E = EnsureChannelFirstd(keys=['image'])
        Or =  Orientationd(keys=['image'], axcodes="RAS", labels=None)
        data = parse_input(image_files)
        data = [loader(dat) for dat in data]
        data = [E(dat) for dat in data]
        data = [Or(dat) for dat in data]
        return data


    def setup_localiser_inferer(self):
        W = self.YoloInferer(
            localiser_regions=self.localiser_regions,
            bs=self.yolo_bs,
            devices=self.devices,
            debug=self.debug,
        )
        self.yolo_specs = W.yolo_state_dict
        self.classes = localiser_regions_to_yolo_classes(
            self.yolo_specs, self.localiser_regions
        )
        return W

    def run(self, data: list, chunksize=12, overwrite=False):
        """
        data: can be a list of images: comprising any of filenames, folder, or images (sitk or itk)
        chunksize is necessary in large lists to manage system ram
        """
        self.setup()
        data = self.maybe_filter_images(data, overwrite)
        data_bboxes = self.extract_fg_bboxes(data, overwrite=overwrite)
        data_chunks = list_to_chunks(data_bboxes, chunksize)
        for data_sublist in data_chunks:
            output = self.process_data_sublist(data_sublist)
        return output


    def process_data_sublist(self, data_sublist):
        self.create_and_set_postprocess_transforms()
        # data = self.load_images(imgs_sublist)
        data = self.apply_bboxes(data_sublist)
        Sq = SqueezeDimd(keys=["image"], dim=0)
        data = [Sq(dat) for dat in data]
        pred_patches = self.patch_prediction(data)
        pred_patches = self.decollate_patches(pred_patches, bboxes_sublist)
        output = self.postprocess(pred_patches)
        self.cuda_clear()
        return output


    def extract_fg_bboxes(self, data, overwrite=False):
        if overwrite is False:
            data_out = self.load_bboxes(data)
        else:
            outputs = self.W.run(data, overwrite=overwrite)
            data_out = []
            for dat, out in zip(data, outputs):
                pred = out["pred"]
                lat = pred[0]
                ap = pred[1]
                yolo_bbox = standardize_bboxes(
                    ap.boxes,
                    lat.boxes,
                    out["projection_meta"][0]["letterbox_padded"],
                    self.classes,
                    serialised=False,
                )

                img_shape = out['image_orig_shape']
                bounding_box = yolo_bbox_to_slices(img_shape, yolo_bbox)
                bounding_box = add_channel_slices(bounding_box)
                out["yolo_bbox"] = yolo_bbox
                out["bounding_box"] = bounding_box
                data_out.append(out)
        return data_out

    def apply_bboxes(self, data ):
        data2 = []
        for i, dat in enumerate(data):
            image = dat["image"]
            bbox = dat["bounding_box"]
            dat["image"] = image[bbox]
            data2.append(dat)
        return data2

    def load_bboxes(self, data):
        json_fns = list(self.W.output_folder.glob("*.json"))
        data_out = []
        for dat in data:
            out = {"image": dat}
            out = LoadSITKd(["image"])(out)
            out = EnsureChannelFirstd(keys=["image"])(out)
            out = Orientationd(keys=["image"], axcodes="RAS", labels=None)(out)
            img_shape = out["image"].shape[1:]

            fn  = find_matching_fn(dat.name, json_fns )
            if len (fn) == 0:
                continue
                # data_out.append(dici)
            else:
                bbox = load_json(fn[0])
                yolo_bbox = standardize_bboxes(
                    bbox["ap"],
                    bbox["lat"],
                    bbox["ap"]["meta"]["letterbox_padded"],
                    self.classes,
                    serialised=True,
                )

                bounding_box = yolo_bbox_to_slices(img_shape, yolo_bbox)
                bounding_box = add_channel_slices(bounding_box)
                out["yolo_bbox"] = yolo_bbox
                out["bounding_box"] = bounding_box
                data_out.append(out)
        return data_out


# %%
# SECTION:-------------------- setup--------------------------------------------------------------------------------------
# SETUP
if __name__ == "__main__":
    import os
    from fran.inference.common_vars import imgs_bosniak, runs_2d
    from localiser.preprocessing.data.nii2pt_tsl import tsl_folder_name_builder
    from utilz.fileio import load_yaml

    # YOLO PATH VARS (STANDARD LOCALISER TRAIN SETUP)
    common_vars_filename = Path(os.environ["FRAN_CONF"]) / "config.yaml"
    COMMON_PATHS = load_yaml(common_vars_filename)

    ckpt_parent_fldr = Path(COMMON_PATHS["checkpoints_parent_folder"])
    yolo_ckpt_parent = ckpt_parent_fldr / "yolo"
    data_parent_folder = Path(COMMON_PATHS["rapid_access_folder"])
    totalseg_data_folder = data_parent_folder / "totalseg2d"
    exclude_regions = ["gut"]
    merge_windows = False
    data_folder = tsl_folder_name_builder(
        totalseg_data_folder,
        exclude_regions=exclude_regions,
        merge_windows=merge_windows,
    )
    project_name = "totalseg_localiser/" + data_folder.name
    yolo_project_folder = yolo_ckpt_parent / project_name

    print("FRAN_CONF:", os.environ["FRAN_CONF"])
    print("common_vars_filename:", common_vars_filename)
    # print("COMMON_PATHS[yolo_output_folder]:", COMMON_PATHS["yolo_output_folder"])
    print(
        "COMMON_PATHS[checkpoints_parent_folder]:",
        COMMON_PATHS["checkpoints_parent_folder"],
    )
    print("COMMON_PATHS[rapid_access_folder]:", COMMON_PATHS["rapid_access_folder"])
    print("ckpt_parent_fldr:", ckpt_parent_fldr)
    print("yolo_ckpt_parent:", yolo_ckpt_parent)
    print("data_parent_folder:", data_parent_folder)
    print("totalseg_data_folder:", totalseg_data_folder)
    print("data_folder:", data_folder)
    print("project_name:", project_name)
    print("yolo_project_folder:", yolo_project_folder)

    exclude = ["gut", "neck"]
    if "neck" in exclude:
        run_yolo_wts = runs_2d["ab_ch_pe"][0]
    else:
        run_yolo_wts = runs_2d["ab_ch_ne_pe"][0]

# %%
    # SCRATCH INPUTS
    yolo_include_neck = False
    if yolo_include_neck == False:
        run_w = "ab_ch_pe"
    else:
        run_w = "ab_ch_ne_pe"

    imgs = imgs_bosniak[:4]
    run2d = Path(runs_2d[run_w][0])
    run2d_fldr = run2d.parent.parent
# %%

    D = CascadeInfererYOLO(
        localiser_regions="abdomen,pelvis",
        run_p="KITS23-SIRIG",
    )

# %%
    overwrite = False
    image_files = imgs
    data_bboxes = D.extract_fg_bboxes(image_files, overwrite=overwrite)
# %%
# %%
# %%
    chunksize = 2
    data_chunks = list_to_chunks(data_bboxes, chunksize)
    sublist = data_chunks[0]
# %%
    D.create_and_set_postprocess_transforms()
    data_sublist= sublist
    # data = D.load_images(imgs_sublist)
    data = D.apply_bboxes(data_sublist)
    Sq = SqueezeDimd(keys=["image"], dim=0)
    data = [Sq(dat) for dat in data]
    data[0]['image'].shape
    data[0]['bounding_box']
    bboxes_sublist = [dat["bounding_box"] for dat in sublist]
    pred_patches = D.patch_prediction(data)
    pred_patches = D.decollate_patches(pred_patches, bboxes_sublist)
    pred_patches[0].keys()
    pred_patches[0]['KITS23-SIRIG'].shape
    pred_patches[0]['bounding_box']
    output = D.postprocess(pred_patches)

# %%
    output = D.process_data_sublist(sublist)
    data = D.apply_bboxes(sublist)
    bboxes_sublist = [dat["bounding_box"] for dat in sublist]
    pred_patches = D.patch_prediction(data)
    pred_patches = D.decollate_patches(pred_patches, bboxes_sublist)
    output = D.postprocess(pred_patches)


# %%
    sublist
    D.create_and_set_postprocess_transforms()
    # data = D.load_images(imgs_sublist)
    data = D.apply_bboxes(data, bboxes_sublist)
    data[0]['image'].shape
    data[0]['bounding_box']
    pred_patches = D.patch_prediction(data)
    pred_patches = D.decollate_patches(pred_patches, bboxes_sublist)
    pred_patches[0].keys()
    pred_patches[0]['KITS23-SIRIG'].shape
    pred_patches[0]['bounding_box']
    output = D.postprocess(pred_patches)


# %%
    yb = data[0]["yolo_bbox"]

# %%
    fn = "/s/fran_storage/predictions/totalseg_localiser/train/kits21_b0330_0000.json"
    bbox = load_json(fn)
    # bbox
    
# %%
    data2 = D.apply_bboxes(outs)
    dat = data2[0]
    n=0
# %%
    img = dat["image"]
    img = img[0]
    img = img.permute(2, 0, 1)
    ImageMaskViewer([img, img],'ii')
# %%
    import matplotlib.pyplot as plt

    img = dat["image"][0]
    im = img.float().mean(dim=1)
    im3 = img.float().mean(dim=0)
# %%
    im= im.permute(1, 0)
    im3 = im3.permute(1, 0)
    im = im.flip(0)
    im3 = im3.flip(0)


# %%
    fix, axs = plt.subplots(1, 2)
    axs[0].imshow(im, cmap="gray")
    axs[1].imshow(im3, cmap="gray")
# %%
    plt.imshow(im, cmap="gray")

# %%

    if D.P is None:
        D.setup_patch_inferer()
    pred_patches = D.patch_prediction(data)

# %%
    len(data), data[0]["image"].shape

# %%
    overwrite = True
    overwrite = D.yolo_overwrite if overwrite is None else overwrite
    image_fns = [Path(dat["image"].meta["filename_or_obj"]) for dat in data]

    D.setup_yolo_inferer()
    bboxes = D.extract_fg_bboxes(data, overwrite=overwrite)
    bbox = bboxes[0]
    letterbox3tup = bbox["ap"]["meta"]["letterbox_padded"]
    bbo2 = standardize_bboxes(bbox["ap"], bbox["lat"], letterbox3tup)
# %%
    outputs[0].keys()
    outputs[0]["bboxes_final"]
    by_name = _build_bbox_lookup(outputs)
    bboxes = []
    for dat in data:
        fn = dat["image"].meta["filename_or_obj"]
        bbox = _resolve_bbox(by_name, fn)
        dat["yolo_bbox"] = bbox
        bboxes.append(bbox)
    D.yolo_bboxes = bboxes
    D.bboxes = bboxes

# %%
    D.setup_yolo_inferer()
    D.Y

# %%
    # EXTRACT YOLO BBOXES
    yolo_bboxes = D.extract_fg_bboxes(data, overwrite=True)
    len(yolo_bboxes), yolo_bboxes[0]

# %%
    # APPLY YOLO BBOXES
    data_cropped = D.apply_bboxes(data, D.bboxes)
    len(data_cropped), data_cropped[0]["image"].shape, data_cropped[0]["bounding_box"]
    img = data_cropped[0]["image"]
    img = img.permute(2, 0, 1)
    img1 = img.float().mean(dim=1)
    img2 = img.float().mean(dim=2)
    img1 = img1.flip(0)
    img2 = img2.flip(0)
    img2 = img2.flip(1)

    import matplotlib.pyplot as plt

    plt.imshow(img1, cmap="gray")
    plt.imshow(img2, cmap="gray")

# %%
    # SETUP PATCH INFERER
    D.setup_patch_inferer()
    D.P

# %%
    # PATCH PREDICTION
    pred_patches = D.patch_prediction(data_cropped)
    list(pred_patches), len(pred_patches[D.P.run_name])

# %%
    # RUN END-TO-END TO PATCH PREDICTION
    pred_patches = D.run(imgs, overwrite=False)
    list(pred_patches), len(pred_patches[D.P.run_name])
# %%
    dici = {"image":dat}
    L = LoadSITKd(["image"])
    O = Orientationd(keys=["image"], axcodes="RAS", labels=None)
    E = EnsureChannelFirstd(keys=["image"])
    dici2 = L(dici)
    dici2 = E(dici2)
    dici3 = O(dici2)
    dici3['image'].shape
# %%
    out.keys()
    out['image'].meta['spatial_shape']
    out['image'].meta['affine']
    import nibabel as nib
    img = out['image']
    axcodes = nib.aff2axcodes(out['image'].meta['affine'].cpu().numpy())
    assert axcodes == ("R", "A", "S"), axcodes


# %%
