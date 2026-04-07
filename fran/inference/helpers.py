import itertools as il
import math
from pathlib import Path
from typing import List, Optional

import itk
import numpy as np
import SimpleITK as sitk
import torch
from fran.managers import Project
from fran.trainers import checkpoint_from_model_id
from fran.transforms.imageio import LoadImage, LoadSITKd, SITKReader, TorchReader
from monai.data.itk_torch_bridge import itk_image_to_metatensor as itm
from monai.transforms.io.dictionary import LoadImaged
from utilz.cprint import cprint
from utilz.helpers import slice_list
from utilz.stringz import ast_literal_eval


def get_sitk_target_size_from_spacings(sitk_array, spacing_dest):
    sz_source, spacing_source = sitk_array.GetSize(), sitk_array.GetSpacing()
    sz_dest, _ = get_scale_factor_from_spacings(sz_source, spacing_source, spacing_dest)
    return sz_dest


def rescale_bbox(scale_factor, bbox):
    bbox_out = []
    for a, b in zip(scale_factor, bbox):
        bbox_neo = slice(int(b.start * a), int(np.ceil(b.stop * a)), b.step)
        bbox_out.append(bbox_neo)
    return tuple(bbox_out)


def apply_threshold(input_img, threshold):
    input_img[input_img < threshold] = 0
    input_img[input_img >= threshold] = 1
    return input_img


def get_amount_to_pad(img_shape, patch_size):

    pad_deficits = np.maximum(0, np.array(patch_size) - img_shape)
    padding = (
        (math.floor(pad_deficits[0] / 2), math.ceil(pad_deficits[0] / 2)),
        (math.floor(pad_deficits[1] / 2), math.ceil(pad_deficits[1] / 2)),
        (math.floor(pad_deficits[2] / 2), math.ceil(pad_deficits[2] / 2)),
    )
    return padding


def get_scale_factor_from_spacings(sz_source, spacing_source, spacing_dest):
    scale_factor = [a / b for a, b in zip(spacing_source, spacing_dest)]
    sz_dest = [round(a * b) for a, b in zip(sz_source, scale_factor)]
    return sz_dest, scale_factor


def infer_project(configs):
    """Recursively search through params dictionary to find 'project' key and set it as attribute"""

    def find_project(dici):
        if isinstance(dici, dict):
            for k, v in dici.items():
                if k == "project_title":
                    return v
                result = find_project(v)
                if result is not None:
                    return result
        elif isinstance(dici, list):
            for item in dici:
                result = find_project(item)
                if result is not None:
                    return result
        return None

    project_title = find_project(configs)
    if project_title is not None:
        project = Project(project_title)
        return project
    else:
        raise ValueError("No 'project_title' key found in params dictionary")


def get_device(devices: Optional[List[int]] = None) -> tuple:
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return [], torch.device("cpu"), "cpu"

    if devices is None:
        devices = [0]

    try:
        device_id = devices[0]
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.get_device_properties(device_id)
        return devices, device, "gpu"
    except (RuntimeError, AssertionError) as e:
        print(f"Error accessing CUDA device {devices}: {e}")
        print("Falling back to CPU")
        return [], torch.device("cpu"), "cpu"


def get_patch_spacing(run_name):
    ckpt = checkpoint_from_model_id(run_name)
    dic1 = torch.load(ckpt, map_location="cpu", weights_only=False)
    config = dic1["datamodule_hyper_parameters"]["configs"]
    spacing = config["plan_train"].get("spacing")
    if spacing is None:
        mode = config["plan_train"]["mode"]
        if mode == "whole":
            spacing = [0.8, 1.5, 1.5]
            cprint(
                "Mode is {0}. Using dummy spacing: {1}".format(mode, spacing),
                color="red",
                bold=True,
                bg="yellow",
            )
            print(spacing)
        else:
            raise NotImplementedError
    spacing = ast_literal_eval(spacing)
    return spacing


def list_to_chunks(input_list: list, chunksize: int):
    if len(input_list) < chunksize:
        print("List too small, setting chunksize to len(list)")
        chunksize = np.minimum(len(input_list), chunksize)
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
            elif isinstance(dat, itk.Image):
                dat = itm(dat)
            else:
                raise TypeError(f"Unsupported input type: {type(dat)}")
            dat = [dat]
        imgs_out.extend(dat)
    imgs_out = [{"image": img} for img in imgs_out]
    return imgs_out


def load_images_nifti(data):
    loader = LoadSITKd(["image"])
    data = parse_input(data)
    data = [loader(d) for d in data]
    return data


def load_images_pt(data):
    loader = LoadImaged(["image"], reader=TorchReader)
    data = parse_input(data)
    data = [loader(d) for d in data]
    return data


load_images = load_images_nifti


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
