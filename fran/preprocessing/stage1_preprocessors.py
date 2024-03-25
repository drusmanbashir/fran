# %%
import cc3d
import torchio as tio

from fran.transforms.spatialtransforms import PadDeficitImgMask
from fran.utils.image_utils import resize_tensor_3d
from fran.utils.string import info_from_filename, strip_extension

if "get_ipython" in globals():
    print("setting autoreload")
    from IPython import get_ipython

    ipython = get_ipython()
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
from pathlib import Path

# from fastai.vision.all import *
# export
import ipdb
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from fastcore.basics import GetAttr, listify, store_attr

from fran.preprocessing.datasetanalyzers import bboxes_function_version
from fran.utils.fileio import *
from fran.utils.helpers import *
from fran.utils.imageviewers import *

tr = ipdb.set_trace


def tensors_from_dict_file(filename):
    img_mask = torch.load(filename)
    img = img_mask["img"]
    mask = img_mask["mask"]
    return img, mask


def view_sitk(img_mask_pair):
    img, mask = img_mask_pair
    img, mask = map(sitk.GetArrayFromImage, [img, mask])
    img, mask = img.transpose(2, 1, 0), mask.transpose(2, 1, 0)
    ImageMaskViewer([img, mask])


def resample_tensor_dict(in_filename, out_filename, output_size, overwrite=True):
    if write_file_or_not(out_filename, overwrite) == True:
        img_mask = torch.load(in_filename)
        resized_tensor = {}
        resized_tensor = {
            {img_type: resize_tensor_3d(tensr)} for img_type, tensr in img_mask.items()
        }
        torch.save(resized_tensor, out_filename)


def resample_img_mask_tensors(in_filename, out_filename, output_size, overwrite=True):
    if write_file_or_not(out_filename, overwrite) == True:
        img, mask = tensors_from_dict_file(in_filename)
        resized_tensor = {}
        for x, key in zip([img, mask], ["img", "mask"]):
            mode = "nearest" if "int" in str(x.dtype) else "trilinear"
            x = resize_tensor_3d(x, output_size, mode)
            resized_tensor.update({key: x})
        torch.save(resized_tensor, out_filename)


def calculate_patient_bbox(img, threshold):
    ii = img.clone()
    ii[img < threshold] = 0
    ii[img >= threshold] = 1
    stats_patient = cc3d.statistics(ii.numpy().astype(np.uint8))
    patient_bb = stats_patient["bounding_boxes"][1]
    return patient_bb


def files_exist(filename, any_or_all="any"):
    if not isinstance(filename, list):
        filename = listify(filename)
    return any([fn.exists() for fn in filename])


class CropToPatientTorchToTorch(object):
    def __init__(
        self, output_parent_folder, spacing, pad_each_side: str = "4cm", overwrite=True
    ):
        """
        params:
        target_length : 20 cm by default
        """
        self.output_folders = output_parent_folder
        assert "cm" in pad_each_side, "Must give length in cm for clarity"
        maybe_makedirs(self.output_folders)
        pad_each_side = float(pad_each_side[:-2])
        self.overwrite = overwrite
        self.pad_voxels_each_side = int(pad_each_side * 10 / spacing[0])

    @property
    def output_folders(self):
        return self._output_folders

    @output_folders.setter
    def output_folders(self, output_parent_folder):
        self._output_folders = [
            output_parent_folder / subfld for subfld in ["images", "masks"]
        ]

    @property
    def output_filenames(self):
        return self._output_filenames

    @output_filenames.setter
    def output_filenames(self, filename):
        self._output_filenames = [fldr / filename.name for fldr in self._output_folders]

    def _save_to_file(self):
        argss = zip([self.img_cropped, self.mask_cropped], self.output_filenames)
        [torch.save(a, b) for a, b in argss]

    def process_case(self, img_fn, mask_fn: Path, threshold=-0.4):
        self.output_filenames = img_fn
        if files_exist(self.output_filenames) and self.overwrite == False:
            print("File(s) {} exists. Skipping.".format(self.output_filenames))
            return 0
        else:
            self.img_cropped, self.mask_cropped = self._load_and_crop(
                img_fn, mask_fn, threshold
            )
            self._save_to_file()
            return 1

    def _load_and_crop(self, img_fn, mask_fn, threshold):
        img, mask = map(torch.load, [img_fn, mask_fn])
        # organ_z_center, organ_length_voxels= self._get_organ_stats(mask)
        # pad_total_each_side=int(organ_length_voxels/2+self.pad_voxels_each_side)
        # slices_craniocaudal= slice(int(np.maximum(0,organ_z_center-pad_total_each_side)),int(np.minimum(mask.shape[0],organ_z_center+pad_total_each_side)))
        patient_bb = calculate_patient_bbox(img.clone(), threshold=threshold)
        cropped_bb = tuple([patient_bb[0], patient_bb[1], patient_bb[2]])
        img_cropped, mask_cropped = img[cropped_bb].clone(), mask[cropped_bb].clone()
        return img_cropped, mask_cropped

    def _get_organ_stats(self, mask):
        mask_binary = mask.clone().numpy()
        mask_binary[mask_binary > 1] = 1
        stats_mask = cc3d.statistics(mask_binary)
        mask_centroid = stats_mask["centroids"][-1]
        organ_z_center = int(mask_centroid[0])
        organ_length_voxels = (
            stats_mask["bounding_boxes"][1][0].stop
            - stats_mask["bounding_boxes"][1][0].start
        )
        return organ_z_center, organ_length_voxels


class CropToPatientTorchTonifti(CropToPatientTorchToTorch):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def output_folders(self):
        return self._output_folder

    @output_folders.setter
    def output_folders(self, output_folder):
        self._output_folder = [
            output_folder / "images_nii" / "images",
            output_folder / "images_nii" / "masks",
        ]

    @property
    def output_filenames(self):
        return self._output_filename

    @output_filenames.setter
    def output_filenames(self, filename):
        self._output_filename = [
            folder / (str(filename.name).replace(".pt", ".nii.gz"))
            for folder in self._output_folder
        ]

    def _save_to_file(self, img_cropped, mask_cropped):
        for im, fn in zip([img_cropped, mask_cropped], self.output_filenames):
            im = im.numpy()
            save_sitk(im, fn)

    def _get_organ_stats(self, mask):
        mask_binary = mask.clone().numpy()
        mask_binary[mask_binary > 1] = 1
        stats_mask = cc3d.statistics(mask_binary)
        mask_centroid = stats_mask["centroids"][-1]
        organ_z_center = int(mask_centroid[0])
        organ_length_voxels = (
            stats_mask["bounding_boxes"][1][0].stop
            - stats_mask["bounding_boxes"][1][0].start
        )
        return organ_z_center, organ_length_voxels


class WholeImageTensorMaker:
    def __init__(self, proj_defaults, source_spacing, output_size, num_processes):
        store_attr("proj_defaults, source_spacing,output_size,num_processes")
        resampling_configs_fn = proj_defaults.fixed_spacing_folder / (
            "resampling_configs"
        )
        resampling_configs = load_dict(resampling_configs_fn)
        self.set_files_folders(resampling_configs)
        if any([not fn.exists() for fn in self.mask_files]):
            raise "Some file(s) do not exist. Dataset corrupt"
        print("Run process_tensors()")

    def set_files_folders(self, resampling_configs):
        self.input_folder = [
            conf["resampling_output_folder"]
            for conf in resampling_configs
            if conf["spacing"] == self.source_spacing
        ][0]
        self.output_parent_folder = self.proj_defaults.whole_images_folder / (
            "dim_{0}_{1}_{2}".format(*self.output_size)
        )
        self.output_folder_imgs = self.output_parent_folder / ("images")
        self.output_folder_masks = self.output_parent_folder / ("masks")
        self.img_files = list((self.input_folder / ("images")).glob("*pt"))
        self.mask_files = [
            self.input_folder / ("masks/{}".format(fn.name)) for fn in self.img_files
        ]

    def get_args_for_resizing(self):
        maybe_makedirs([self.output_folder_imgs, self.output_folder_masks])
        arglist_imgs = [
            [
                img_filename,
                self.output_size,
                "trilinear",
                self.output_folder_imgs / img_filename.name,
            ]
            for img_filename in self.img_files
        ]
        arglist_masks = [
            [
                mask_filename,
                self.output_size,
                "nearest",
                self.output_folder_masks / mask_filename.name,
            ]
            for mask_filename in self.mask_files
        ]
        return arglist_imgs, arglist_masks

    # def generate_bboxes_from_masks_folder(self,debug=False,num_processes=8):
    #     generate_bboxes_from_masks_folder(self.output_folder_masks,self.proj_defaults,0.2,debug,num_processes)


def resize_and_save_tensors(input_filename, output_size, mode, output_filename):
    input_tensor = torch.load(input_filename)
    resized_tensor = resize_tensor_3d(input_tensor, output_size, mode)
    torch.save(
        resized_tensor.to(torch.float32 if mode == "trilinear" else torch.uint8),
        output_filename,
    )


def cropper_wrapper_nifti(filename, args):
    C = CropToPatientTorchTonifti(*args)

    return C.process_case(filename)


def cropper_wrapper_torch(filename, args):
    C = CropToPatientTorchToTorch(*args)
    return C.process_case(filename)


def get_cropped_label_from_bbox_info(
    outfolder, bbox_info, label="tumour", label_index=2
):
    filename = bbox_info["filename"]
    out_filename = outfolder / filename.name
    img_mask = torch.load(filename)

    img = img_mask["img"]
    mask = img_mask["mask"]

    bbox_stats = bbox_info["bbox_stats"]
    ref = [b for b in bbox_stats if b["tissue_type"] == label]
    try:
        refo = ref[0]
        slcs = refo["bounding_boxes"]
        slc = slcs[1]
        img_tmr = img[slc]
        mask_tmr = mask[slc]
        mask_tmr[mask_tmr != label_index] = 0
        mask_tmr[mask_tmr == label_index] = 1
        img_tmr = img_tmr * mask_tmr
        mask_tmr[mask_tmr == 1] = label_index
        out_tensr = {"img": img_tmr, "mask": mask_tmr}
        torch.save(out_tensr, out_filename)
        return out_filename
    except:
        print("Label {0} not in this case {1}".format(label, bbox_info["case_id"]))
        return 0, filename


def pad_bbox(bbox, padding_torch_style):  # padding is reverse order Torch style
    padding = padding_torch_style[::-1]
    assert len(padding) == 6, "Padding must be 6-tuple"
    out_slcs = []
    for indx in range(len(bbox)):
        slcs = bbox[indx]
        slc_new = slice(
            int(slcs.start - padding[indx * 2]), int(slcs.stop + padding[indx * 2 + 1])
        )
        out_slcs.append(slc_new)
    return tuple(out_slcs)


#
# def patch_generator_wrapper(output_folder,output_patch_size, info,oversampling_factor=0,expand_by=None):
#     # make sure output_folder already has been created
#     if not oversampling_factor: oversampling_factor=0.
#     assert oversampling_factor<0.9 , "That will create a way too large data folder. Choose an oversampling_factor between [0, 0.9)"
#     patch_overlap = [int(oversampling_factor*ps) for ps in output_patch_size]
#     patch_overlap=[to_even(ol) for ol in patch_overlap]
#
#     dataset_properties_fn = info['filename'].parent.parent/("resampled_dataset_properties")
#     dataset_properties=load_dict(dataset_properties_fn)
#     P= PatchGeneratorSingle(dataset_properties,output_folder,output_patch_size, info,patch_overlap,expand_by)
#     P.create_patches_from_all_bboxes()
#     return 1,info['filename']
# %%

if __name__ == "__main__":
    ######################################################################################
# %% [markdown]
    ## Creates low res images
# %%

# %%

    from fran.utils.common import *

    P = Project(project_title="litsmc")
    proj_defaults = P
    PG = PatchGeneratorDataset(
        P,
        Path(
            "/s/fran_storage/datasets/preprocessed/fixed_spacing/litsmc/spc_080_080_150"
        ),
        [192, 192, 128],
        0.25,
        20,
    )
    PG.create_patches(overwrite=False, debug=True)
    PG.generate_bboxes(debug=False)
# %%

# %%
    all_cases = set([bb["case_id"] for bb in PG.fixed_sp_bboxes])
    new_case_ids = all_cases.difference(PG.existing_case_ids)
    print(
        "Total cases {0}.Found {1} new cases".format(len(all_cases), len(new_case_ids))
    )
    PG.fixed_sp_bboxes = [
        bb for bb in PG.fixed_sp_bboxes if bb["case_id"] in new_case_ids
    ]

# %%
    fixed_spacing_folderolder = proj_defaults.fixed_spacing_folder / ("spc_080_080_150")
    stage0_bboxes_fn = fixed_spacing_folderolder / ("bboxes_info")
    output_folder = Path(
        "/home/ub/datasets/preprocessed/litsmc/patches/spc_080_080_150/dim_192_192_128/"
    )

    stage0_bboxes = load_dict(stage0_bboxes_fn)

    info = stage0_bboxes[0]
    patch_config_fn = "/home/ub/datasets/preprocessed/litsmc/patches/spc_080_080_150/dim_192_192_128/patches_config.json"
    patches_config = load_dict(patch_config_fn)
    oversampling_factor, expand_by = patches_config.values()
    output_patch_size = [192, 192, 128]
    patch_overlap = [int(oversampling_factor * ps) for ps in output_patch_size]
    patch_overlap = list(map(to_even, patch_overlap))
    tr()

    dataset_properties_fn = info["filename"].parent.parent / (
        "resampled_dataset_properties"
    )
    dataset_properties = load_dict(dataset_properties_fn)

    P = PatchGeneratorFG(
        dataset_properties,
        output_folder,
        output_patch_size,
        info,
        patch_overlap,
        expand_by,
    )
    P.create_patches_from_all_bboxes()
    output_shape = [128, 128, 96]
    overs = 0.25
    fixed_folder = proj_defaults.fixed_spacing_folder / ("spc_080_080_150/images")
    fixed_files = list(fixed_folder.glob("*.pt"))
    dataset_properties = load_dict(
        Path(
            "/s/fran_storage/datasets/preprocessed/fixed_spacing/lits/spc_080_080_150/resampled_dataset_properties.json"
        )
    )
    output_patch_size = [192, 192, 196]
    output_folder = Path("/home/ub/tmp")
    dici_fn = Path(
        "/s/fran_storage/datasets/preprocessed/fixed_spacing/lits/spc_080_080_150/bboxes_info.pkl"
    )
    inf = load_dict(dici_fn)

    info = inf[0]
    P = PatchGeneratorFG(dataset_properties, output_folder, output_patch_size, info)
# %%
    n = 0
    img_fn = fixed_files[n]
    mask_fn = img_fn.str_replace("images", "masks")
# %%

    spacing = load_dict(proj_defaults.resampled_dataset_properties_filename)[
        "preprocessed_dataset_spacing"
    ]
    global_props = load_dict(proj_defaults.raw_dataset_properties_filename)[-1]
# %%

    C = CropToPatientTorchToTorch(
        output_parent_folder=Path("tmp"), spacing=[0.77, 0.77, 1], pad_each_side="0cm"
    )
    C.process_case(img_fn=img_fn, mask_fn=mask_fn)

# %%
    img, mask = map(torch.load, [img_fn, mask_fn])
    ImageMaskViewer([img, mask])
# %%

    ImageMaskViewer([C.img_cropped, C.mask_cropped])

# %%
    stage1_fldr = Path(
        "/home/ub/datasets/preprocessed/lits/patches/spc_077_077_100/dim_320_320_256/"
    )
    img_fldr = stage1_fldr / ("images")
    mask_fldr = stage1_fldr / ("masks")
    stage1_img_fn = list(img_fldr.glob("*"))
    stage1_mask_fn = list(mask_fldr.glob("*"))
# %%
    n = 100
    filenames = [stage1_img_fn[n], stage1_mask_fn[n]]
    img, mask = list(map(torch.load, filenames))
    x = [x.permute(2, 1, 0) for x in [img, mask]]
    ImageMaskViewer([x[0], x[1]], intensity_slider_range_percentile=[0, 100])
    # ImageMaskViewer([img[lims],mask[lims]])
# %%
    fn = "/home/ub/datasets/preprocessed/litsmc/patches/spc_080_080_150/dim_192_192_128/bboxes_info.pkl"
    bboxes = load_dict(fn)
    ######################################################################################
    tnsr_fn = bb["filename"]
    tnsr = torch.load(tnsr_fn)
    print(tnsr.shape)
# %% [markdown]
## Trialling torch to nibabel format for rapid loading
#
