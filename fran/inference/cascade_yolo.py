import json
import os
from pathlib import Path

from fran.inference.common_vars import imgs_bosniak, runs_2d
from fran.inference.cascade import CascadeInferer, PatchInferer, img_bbox_collated
from fran.inference.helpers import parse_input
from fran.transforms.imageio import LoadSITKd
from fran.transforms.spatialtransforms import CropByYolo
from localiser.inference.base import LocaliserInferer, LocaliserInfererPT, load_yolo_specs
from localiser.transforms.tsl import TSLRegions
from localiser.utils.bbox_helpers import standardize_bboxes
from ultralytics import YOLO
from utilz.fileio import load_yaml, maybe_makedirs


def _class_to_index(names):
    if isinstance(names, dict):
        return {str(v): int(k) for k, v in names.items()}
    return {str(name): idx for idx, name in enumerate(names)}


def _region_classes(yolo_specs, localiser_regions):
    class_to_ind = _class_to_index(yolo_specs["data"]["names"])
    if localiser_regions is None:
        return sorted(class_to_ind.values())

    if isinstance(localiser_regions, list | tuple | set):
        regions_list = [str(region).strip() for region in localiser_regions if str(region).strip()]
        valid_regions = set(TSLRegions().regions)
        invalid = [region for region in regions_list if region not in valid_regions]
        if invalid:
            raise ValueError(f"Invalid TSL localiser_regions: {invalid}")
    else:
        regions = str(localiser_regions).replace(" ", "")
        regions_list = [r for r in regions.split(",") if r]

    classes = []
    for class_name, class_idx in class_to_ind.items():
        if any(region in class_name for region in regions_list):
            classes.append(class_idx)
    return sorted(set(classes))


def yolo_classes_from_regions(yolo_folder, localiser_regions):
    yolo_specs = load_yolo_specs(Path(yolo_folder))
    return _region_classes(yolo_specs, localiser_regions)


def _selector_classes(yolo_specs, selectors):
    class_to_ind = _class_to_index(yolo_specs["data"]["names"])
    if selectors is None:
        return None
    if isinstance(selectors, str):
        selectors = [s for s in selectors.replace(" ", "").split(",") if s]

    classes = []
    for selector in selectors:
        if isinstance(selector, int):
            classes.append(int(selector))
            continue
        selector_str = str(selector).strip()
        if selector_str in class_to_ind:
            classes.append(class_to_ind[selector_str])
            continue
        if selector_str.isdigit():
            classes.append(int(selector_str))
            continue
        selector_key = selector_str.replace(" ", "")
        for class_name, class_idx in class_to_ind.items():
            if selector_key in class_name.replace(" ", ""):
                classes.append(class_idx)
    return sorted(set(classes))




def _filename_match_keys(filename):
    key = str(filename).replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    keys = []
    while key not in keys:
        keys.append(key)
        if key.endswith(".nii.gz"):
            key = key[:-7]
        elif key.endswith(".nii") or key.endswith(".pt") or key.endswith(".txt"):
            key = Path(key).stem
        else:
            key = Path(key).stem
        if not key:
            break
    return keys


def _build_bbox_lookup(outputs):
    by_key = {}
    for out in outputs:
        filename = out["projection_meta"][0]["filename_or_obj"]
        bbox = _standardize_serialized_bbox(out["bboxes_final"])
        for key in _filename_match_keys(filename):
            if key in by_key:
                prev = by_key[key]["filename"]
                raise KeyError(
                    f"Ambiguous YOLO bbox key '{key}' for '{prev}' and '{filename}'."
                )
            by_key[key] = {
                "bbox": bbox,
                "filename": str(filename),
            }
    return by_key


def _resolve_bbox(by_key, filename):
    tried = _filename_match_keys(filename)
    for key in tried:
        if key in by_key:
            return by_key[key]["bbox"]
    sample_keys = sorted(by_key)[:8]
    raise KeyError(
        f"Could not match YOLO bbox for '{filename}'. Tried keys: {tried}. "
        f"Sample available keys: {sample_keys}"
    )


def _standardize_serialized_bbox(bbox):
    return standardize_bboxes(
        bbox["ap"],
        bbox["lat"],
        bbox["ap"]["meta"]["letterbox_padded"],
        bbox["ap"]["classes"],
    )


class CascadeInfererYOLO(CascadeInferer):
    YoloInferer = LocaliserInferer
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
        self.run_p = run_p
        self.devices = devices
        self.safe_mode = safe_mode
        self.patch_overlap = patch_overlap
        self.save_channels = save_channels
        self.save = save
        self.debug = debug
        self.k_largest = k_largest
        self.yolo_specs = None
        self.W = None
        self.P = None
        self.cropper_yolo = CropByYolo(
            keys=["image"],
            lm_key="image",
            bbox_key="bbox",
            margin=20,
            sanitize=False,
        )
        self.setup_yolo_inferer(localiser_regions)



    def setup_yolo_inferer(self,localiser_regions):
        from localiser.inference.base import resolve_yolo_wts
        self.yolo_ckpt = resolve_yolo_wts( localiser_regions)
        yolo_folder = self.yolo_ckpt.parent.parent
        self.yolo_specs = load_yolo_specs(Path(yolo_folder))
        self.W = LocaliserInferer(
            yolo_folder,
            batch_size=64,
        )


    def load_images(self, image_files):
        loader =  LoadSITKd(["image"])
        data = parse_input(image_files)
        data = [loader(dat) for dat in data]
        return data


    def extract_fg_bboxes(self, data, overwrite=None):
        outputs = self.W.run(data, overwrite=overwrite)
        by_name = _build_bbox_lookup(outputs)
        bboxes = []
        for dat in data:
            fn = dat["image"].meta["filename_or_obj"]
            bbox = _resolve_bbox(by_name, fn)
            dat["yolo_bbox"] = bbox
            bboxes.append(bbox)
        self.yolo_bboxes = bboxes
        self.bboxes = bboxes
        return bboxes

    def setup_patch_inferer(self):
        self.P = PatchInferer(
            run_name=self.run_p,
            devices=self.devices,
            patch_overlap=self.patch_overlap,
            save_channels=self.save_channels,
            safe_mode=self.safe_mode,
            save=self.save,
            debug=self.debug,
        )
        return self.P

    def apply_bboxes(self, data=None, bboxes=None):
        data = self.data if data is None else data
        bboxes = self.bboxes if bboxes is None else bboxes
        data2 = []
        for dat, bbox in zip(data, bboxes):
            cropped = self.cropper_yolo({"image": dat["image"], "bbox": bbox})
            dat["image"] = cropped["image"]
            dat["bounding_box"] = bbox
            data2.append(dat)
        self.data_cropped = data2
        return data2

    def patch_prediction(self, data=None):
        data = self.data_cropped if data is None else data
        preds_all_runs = super().patch_prediction(data)
        self.pred_patches = preds_all_runs
        return preds_all_runs

    def run(self, imagefiles, overwrite=None):
        data = self.load_images(imagefiles)
        bboxes = self.extract_fg_bboxes(data, overwrite=overwrite)
        data = self.apply_bboxes(data, bboxes)
        if self.P is None:
            self.setup_patch_inferer()
        pred_patches = self.patch_prediction(data)
        return pred_patches

# %%
# SECTION:-------------------- setup--------------------------------------------------------------------------------------
# SETUP
if __name__ == "__main__":
    from localiser.preprocessing.data.nii2pt_tsl import tsl_folder_name_builder

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

    exclude=["gut", "neck"]
    if "neck" in exclude:
        run_yolo_wts = runs_2d['ab_ch_pe'][0]
    else:
        run_yolo_wts = runs_2d['ab_ch_ne_pe'][0]

# %%
    # SCRATCH INPUTS
    yolo_include_neck = False
    if yolo_include_neck==False:
        run_w = "ab_ch_pe"
    else:
        run_w = "ab_ch_ne_pe"


    imgs = imgs_bosniak[:4]
    run2d = Path(runs_2d[run_w][0])
    run2d_fldr = run2d.parent.parent
# %%

    # INSTANTIATE DUMMY
    D = CascadeInfererYOLO(
        localiser_regions="abdomen,pelvis",
        run_p="KITS23-SIRIG",
    )

# %%
    image_files = imgs
    data = D.load_images(image_files)
    bboxes = D.extract_fg_bboxes(data, overwrite=overwrite)
    data = D.apply_bboxes(data, bboxes)
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
    letterbox3tup = bbox['ap']['meta']['letterbox_padded']
    bbo2 = standardize_bboxes(bbox['ap'], bbox['lat'], letterbox3tup)
# %%
    outputs[0].keys()
    outputs[0]['bboxes_final']
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
    img = data_cropped[0]['image']
    img = img.permute(2,0,1)
    img1 = img.float().mean(dim=1)
    img2= img.float().mean(dim=2)
    img1 = img1.flip(0)
    img2 = img2.flip(0)
    img2 = img2.flip(1)


    import matplotlib.pyplot as plt
    plt.imshow(img1,cmap='gray')
    plt.imshow(img2,cmap='gray')


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
