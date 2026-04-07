# %%
import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
from fastcore.basics import listify
from fran.configs.mnemonics import Mnemonics
from fran.transforms.imageio import ToTensorT
from label_analysis.utils import SITKImageMaskFixer
from utilz.fileio import Union, maybe_makedirs, os, pd, save_dict, save_list, sitk, str_to_path, torch, tr
from utilz.helpers import find_matching_fn, info_from_filename, multiprocess_multiarg, re
from utilz.stringz import info_from_filename


def sanitize_meta_for_monai(obj):
    """
    Recursively convert NumPy scalar values in metadata to native Python types.
    This avoids MONAI switch_endianness failures on np.int64/np.float* scalars.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: sanitize_meta_for_monai(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_meta_for_monai(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_meta_for_monai(v) for v in obj)
    return obj


def to_even(input_num, lower=True):
    np.fnc = np.subtract if lower == True else np.add
    output_num = np.fnc(input_num, input_num % 2)
    return int(output_num)


def bbox_bg_only(bbox_stats):
    all_fg_bbox = [bb for bb in bbox_stats if bb["label"] == "all_fg"][0]
    bboxes = all_fg_bbox["bounding_boxes"]
    if len(bboxes) == 1:
        return True
    elif bboxes[0] != bboxes[1]:
        return False
    else:
        tr()


def import_h5py():
    import h5py

    return h5py


def get_intensity_range(global_properties: dict) -> list:
    key_idx = [
        key for key in global_properties.keys() if "intensity_percentile" in key
    ][0]
    intensity_range = global_properties[key_idx]
    return key_idx, intensity_range


@str_to_path()
def get_img_mask_filepairs(
    parent_folder: Union[str, Path],
    cases_db: Union[str, Path] = None,
    return_metadata: bool = False,
):
    """
    param: parent_folder. Must contain subfolders labelled masks and images
    Files in either folder belonging to a given case should be identically named.
    """
    if cases_db:
        db_path = Path(cases_db)
        if db_path.exists():
            con = sqlite3.connect(str(db_path))
            try:
                cur = con.cursor()
                cols = [
                    row[1]
                    for row in cur.execute("PRAGMA table_info(datasources)").fetchall()
                ]
                has_ds_type = "ds_type" in cols
                if has_ds_type:
                    query = (
                        "SELECT ds, alias, ds_type, case_id, image, lm FROM datasources"
                    )
                else:
                    query = "SELECT ds, alias, case_id, image, lm FROM datasources"
                rows = cur.execute(query).fetchall()
            finally:
                con.close()

            pairs = []
            for row in rows:
                if has_ds_type:
                    ds, alias, ds_type, case_id, image, lm = row
                else:
                    ds, alias, case_id, image, lm = row
                    ds_type = None
                if return_metadata:
                    pairs.append(
                        {
                            "ds": ds,
                            "alias": alias,
                            "ds_type": ds_type,
                            "case_id": case_id,
                            "image": Path(image),
                            "lm": Path(lm),
                        }
                    )
                else:
                    pairs.append([Path(image), Path(lm)])
            return pairs

    imgs_folder = Path(parent_folder) / "images"
    masks_folder = Path(parent_folder) / "lms"
    imgs_all = list(imgs_folder.glob("*"))
    masks_all = list(masks_folder.glob("*"))
    assert len(imgs_all) == len(masks_all), (
        "{0} and {1} folders have unequal number of files!".format(
            imgs_folder, masks_folder
        )
    )
    img_label_filepairs = []
    for img_fn in imgs_all:
        label_fn = find_matching_fn(img_fn, masks_all)[0]
        assert label_fn.exists(), (
            f"{label_fn} doest not exist, \ncorresponding to {img_fn}"
        )
        img_label_filepairs.append([img_fn, label_fn])
    return img_label_filepairs


@str_to_path(0)
def verify_dataset_integrity(folder: Path, debug=False, fix=False):
    """
    folder has subfolders images and masks
    """
    print("Verifying dataset integrity")
    subfolder = list(folder.glob("mask*"))[0]
    args = [[fn, fix] for fn in subfolder.glob("*")]
    res = multiprocess_multiarg(verify_img_label_match, args, debug=debug, io=True)
    errors = [item for item in res if re.search("mismatch", item[0], re.IGNORECASE)]
    if len(errors) > 0:
        outname = folder / ("errors.txt")
        print(f"Errors found saved in {outname}")
        save_list(errors, outname)
        res.insert(0, errors)
    else:
        print("All images and masks are verified for matching sizes and spacings.")
    return res


def verify_datasets_integrity(folders: list, debug=False, fix=False) -> list:
    folders = listify(folders)
    res = []
    for folder in folders:
        res.extend(verify_dataset_integrity(folder, debug, fix))
    return res


def verify_img_label_match(label_fn: Path, fix=False):
    imgs_foldr = label_fn.parent.str_replace("lms", "images")
    img_fnames = list(imgs_foldr.glob("*"))
    assert imgs_foldr.exists(), (
        "{0} corresponding to {1} parent folder does not exis".format(
            imgs_foldr, label_fn
        )
    )
    img_fn = find_matching_fn(label_fn, img_fnames)
    if ".pt" in label_fn.name:
        return verify_img_label_torch(label_fn)
    else:
        S = SITKImageMaskFixer(img_fn, label_fn)
        S.process(fix=fix)
        return S.log


@str_to_path()
def verify_img_label_torch(label_fn: Path):
    if isinstance(label_fn, str):
        label_fn = Path(label_fn)
    img_fn = label_fn.str_replace("lms", "images")
    img, mask = list(map(torch.load, [img_fn, label_fn]))
    if img.shape != mask.shape:
        print(f"Image mask mismatch {label_fn}")
        return "\nMismatch", img_fn, label_fn, str(img.shape), str(mask.shape)


def get_label_stats(mask, label, separate_islands=True, dusting_threshold: int = 0):
    import cc3d

    if torch.is_tensor(mask):
        mask = mask.numpy()
    label_tmp = np.copy(mask.astype(np.uint8))
    label_tmp[mask != label] = 0
    if dusting_threshold > 0:
        label_tmp = cc3d.dust(
            label_tmp, threshold=dusting_threshold, connectivity=26, in_place=True
        )

    if separate_islands:
        label_tmp, N = cc3d.largest_k(label_tmp, k=1000, return_N=True)
    stats = cc3d.statistics(label_tmp)
    return stats


def percentile_range_to_str(percentile_range):
    def _num_to_str(num: float):
        if num in [0, 100]:
            substr = str(num)
        else:
            str_form = str(num).split(".")
            prefix_zeros = 2 - len(str_form[0])
            suffix_zeros = 1 - len(str_form[1])
            substr = "0" * prefix_zeros + str_form[0] + str_form[1] + "0" * suffix_zeros
        return substr

    substrs = [_num_to_str(num) for num in percentile_range]
    return "_".join(["intensity_percentile"] + substrs)


def get_std_numerator(img_fname, dataset_mean, clip_range=None):
    img = ToTensorT(torch.float32)(img_fname)
    if clip_range is not None:
        img = torch.clip(img, min=clip_range[0], max=clip_range[1])
    var = (img - dataset_mean) ** 2
    var_sum = var.sum()
    return var_sum


def get_means_voxelcounts(img_fname, clip_range=None):
    img = ToTensorT(torch.float32)(img_fname)
    if clip_range is not None:
        img = torch.clip(img, min=clip_range[0], max=clip_range[1])
    return img.mean().item(), img.numel()


def infer_dataset_stats_window(project):
    candidates = []
    global_props = getattr(project, "global_properties", {}) or {}
    mnemonic = global_props.get("mnemonic")
    if isinstance(mnemonic, (list, tuple)):
        candidates.extend(mnemonic)
    elif mnemonic is not None:
        candidates.append(mnemonic)
    candidates.append(getattr(project, "project_title", None))

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            canonical = Mnemonics.match(str(candidate))
            if canonical == Mnemonics.lungs.name:
                return "lung"
        except ValueError:
            continue
    return "abdomen"


def show_gif_in_chrome_if_available(gif_path: Path) -> None:
    gif_path = Path(gif_path).resolve()
    print(f"Dataset stats GIF: {gif_path}")
    chrome_path = next(
        (
            candidate
            for candidate in (
                "google-chrome",
                "google-chrome-stable",
                "chromium",
                "chromium-browser",
            )
            if shutil.which(candidate)
        ),
        None,
    )
    if chrome_path is None:
        print("Google Chrome not available, skipping GIF preview.")
        return
    try:
        subprocess.Popen(
            [chrome_path, "--new-window", gif_path.as_uri()],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Failed to open dataset stats GIF in Chrome: {e}")


def create_dataset_stats_artifacts(
    output_folder,  gif: bool = True, label_stats=False, preview=True, gif_window="abdomen" 
):
    dataset_root = Path(output_folder)
    lms_folder = dataset_root / "lms"
    if not lms_folder.exists():
        print(f"Skipping dataset stats: missing labels folder {lms_folder}")
        return
    stats_folder = dataset_root / "dataset_stats"
    from label_analysis.dataset_stats import end2end_lms_stats_and_plots
    from utilz.overlay_grid_gif import create_nifti_overlay_grid_gif

    if label_stats or gif:
        maybe_makedirs([stats_folder])
    if label_stats == True:
        df, _ = end2end_lms_stats_and_plots(
            lis_folder=lms_folder,
            output_folder=stats_folder,
        )
    if gif == True:
        output_gif = stats_folder / "snapshot.gif"
        create_nifti_overlay_grid_gif(
            dataset_root=dataset_root,
            output_gif=output_gif,
            grid_shape=(3, 3),
            num_frames=20,
            stride=4,
            window=gif_window,
            fps=5,
        )
        if preview == True:
            show_gif_in_chrome_if_available(output_gif)


def postprocess_artifacts_missing(data_folder:Path) ->dict:
    data_folder = Path(data_folder)
    missings={ "label_stats":True, "gif":True}
    stats_folder = data_folder / "dataset_stats"
    labels_stats_fn = data_folder / "lesion_stats.csv"
    gif_fn = stats_folder / "snapshot.gif"
    if labels_stats_fn.exists():
        missings["label_stats"] = False
    if gif_fn.exists():
        missings["gif"] = False
    return missings



class BBoxesFromMask(object):
    """ """

    def __init__(
        self,
        filename,
        bg_label=0,  # so far unused in this code
    ):
        if not isinstance(filename, Path):
            filename = Path(filename)
        if filename.suffix == ".pt":
            self.mask = torch.load(filename, weights_only=False)
        else:
            self.mask = sitk.ReadImage(str(filename))
        if isinstance(self.mask, torch.Tensor):
            self.mask = np.array(self.mask)
        if isinstance(self.mask, sitk.Image):
            self.mask = sitk.GetArrayFromImage(self.mask)
        case_id = info_from_filename(filename.name, full_caseid=True)["case_id"]
        self.bboxes_info = {
            "case_id": case_id,
            "filename": filename,
        }
        self.bg_label = bg_label

    def __call__(self):
        bboxes_all = []
        label_all_fg = self.mask.copy()
        label_all_fg[label_all_fg > 1] = 1
        labels = np.unique(self.mask)
        labels = np.delete(labels, self.bg_label)
        for label in labels:
            stats = {"label": label}
            stats.update(get_label_stats(self.mask, label, True))
            bboxes_all.append(stats)

        stats = {"label": "all_fg"}
        stats.update(get_label_stats(label_all_fg, 1, False))
        bboxes_all.append(stats)
        self.bboxes_info["bbox_stats"] = bboxes_all
        return self.bboxes_info


def bboxes_function_version(filename, bg_label):

    A = BBoxesFromMask(filename, bg_label=bg_label)
    return A()


@str_to_path(0)
def summarize_indices_folder(
    indices_folder,
):
    """
    Summarize per-file FG/BG indices saved in <base_folder>/<indices_subfolder>.
    Stores and returns a dict with per-patch rows and aggregate fg/bg stats.
    """
    indices_folder = Path(indices_folder)
    if not indices_folder.exists():
        raise FileNotFoundError(f"indices folder not found: {indices_folder}")

    base_folder = indices_folder.parent
    rows = []
    for fn in sorted(indices_folder.glob("*.pt")):
        inds = torch.load(fn, weights_only=False)
        fg = inds["lm_fg_indices"]
        bg = inds["lm_bg_indices"]
        n_fg = int(len(fg))
        n_bg = int(len(bg))
        case_id = info_from_filename(fn.name, full_caseid=True)["case_id"]
        rows.append(
            {
                "case_id": case_id,
                "fn_name": fn.name,
                "n_fg": n_fg,
                "n_bg": n_bg,
                "has_fg": bool(n_fg > 0),
            }
        )

    results_df = pd.DataFrame(rows, index=None)
    output_csv_name = "resampled_dataset_properties.csv"
    save_dict(results_df, base_folder / output_csv_name)
    return results_df


def compute_fgbg_ratio(resampled_dataset_properties_df, nnz_allowed):
    n_fg_total = resampled_dataset_properties_df["n_fg"].sum()
    if nnz_allowed == True:
        inds = resampled_dataset_properties_df.index
    else:
        inds = resampled_dataset_properties_df.index[
            resampled_dataset_properties_df["has_fg"] == True
        ]
    n_bg_total = resampled_dataset_properties_df.loc[inds, "n_bg"].sum()
    fgbg_ratio = n_fg_total / n_bg_total
    return fgbg_ratio


if __name__ == "__main__":
    # %%
    fn = "/s/fran_storage/datasets/raw_data/lidc/lms/lidc_0030.nii.gz"
    fn = "/s/xnat_shadow/crc/lms/crc_CRC004_20190425_CAP1p5.nii.gz"
    A = BBoxesFromMask(fn, bg_label=0)
    A()
    print(A.bboxes_info)
    # %%
    import os
    import zipfile

    import torch
    import torch.serialization

    t = torch.arange(6, dtype=torch.float32).reshape(2, 3).contiguous()

    # Force the zip-format that C++ expects:
    pth = (
        "/r/datasets/preprocessed/litsmc/lbd/spc_080_080_080_ex000/images/drli_001ub.pt"
    )
    im = torch.load(pth, weights_only=False)
    im = torch.Tensor(im)
    torch.save(im, "/tmp/pt_tensor.pt", _use_new_zipfile_serialization=True)
    # %%

    path = "/tmp/pt_tensor.pt"

    # 1) Save a single tensor in the new zip format
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3).contiguous()
    # torch.save(t, path, _use_new_zipfile_serialization=True)
    torch.jit.save(torch.jit.script(t), path)
    # %%
    import torch
    from torch import nn

    class TensorContainer(nn.Module):
        def __init__(self, tensor_dict):
            super().__init__()
            for key, value in tensor_dict.items():
                setattr(self, key, value)

    x = torch.ones(4, 4)
    tensor_dict = {"x": x}
    tensors = TensorContainer(tensor_dict)
    tensors = torch.jit.script(tensors)
    tensors.save(path)
    # 2) Verify the file format and the object type
    print("FILE_EXISTS:", os.path.exists(path))  # True
    print("IS_ZIPFILE (zipfile):", zipfile.is_zipfile(path))  # True

    with open(path, "rb") as f:
        print("IS_ZIPFILE (torch):", torch.serialization._is_zipfile(f))  # True

    obj = torch.load(path, map_location="cpu")
    print("PY_OBJ_TYPE:", type(obj))  # <class 'torch.Tensor'>
    print("PY_TENSOR_SHAPE:", obj.shape)

    # %%
    import io

    x = torch.arange(10)
    f = io.BytesIO()
    torch.save(x, f, _use_new_zipfile_serialization=True)
    # send f wherever

    # %%
    path = "/tmp/pt_tensor.pt"
    with open(path, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(f.getbuffer())
# %%
