# %%
from fastcore.all import test_eq
from fastcore.basics import store_attr
from fastcore.transform import Pipeline
from fran.transforms.intensitytransforms import ClipCenter

from fran.preprocessing.datasetanalyzers import (
    bboxes_function_version,
)
from fran.preprocessing.datasetanalyzers import *
from fran.transforms.spatialtransforms import Unsqueeze
from fran.transforms.batchtransforms import ResizeBatch
from fran.transforms.misc_transforms import Squeeze
from fran.transforms.totensor import ToTensorImgMask
from fran.utils.image_utils import *
from fran.utils.imageviewers import *
from fran.utils.helpers import *
from mask_analysis.utils import *
import numpy as np

# from fastai.vision.all import *
# export
import ipdb
import pandas as pd
from fran.utils.image_utils import *

tr = ipdb.set_trace


def generate_bboxes_from_masks_folder(
    masks_folder, bg_label=0,  debug=False, num_processes=16
):
    mask_files = masks_folder.glob("*pt")
    arguments = [
        [x,  bg_label] for x in mask_files
    ]  # 0.2 factor for thresholding as kidneys are small on low-res imaging and will be wiped out by default threshold 3000
    bboxes = multiprocess_multiarg(
        func=bboxes_function_version,
        arguments=arguments,
        num_processes=num_processes,
        debug=debug,
    )
    bbox_fn = masks_folder.parent / ("bboxes_info")
    print("Storing bbox info in {}".format(bbox_fn))
    save_dict(bboxes, bbox_fn)


def save_properties(properties, output_folder, verbose=True):
    case_id = properties["case_id"]
    outfilename = output_folder / ("{}.pkl".format(case_id))
    if verbose == True:
        print("Saving properties to file: {}".format(outfilename))
    save_pickle(properties, outfilename)


def crop_clip(case, proj_defaults):
    img, mask, properties = get_img_mask_from_nii(case)
    mask_gantry = np.ones(img.shape)
    mask_gantry[img < 50] = 0
    mask_gantry[img > 250] = 0
    bbox_gantry = get_bbox_from_mask(mask_gantry)

    img_gantry_cropped, mask_gantry_cropped = [
        crop_to_bbox(
            x,
            bbox_gantry,
            crop_axes="xyz",
            stride=proj_defaults.crop_stride,
            crop_padding=proj_defaults.crop_padding,
        )
        for x in [img, mask]
    ]
    img_gantry_cropped[
        img_gantry_cropped < proj_defaults.HU_clip_range[0]
    ] = proj_defaults.HU_clip_range[0]
    img_gantry_cropped[
        img_gantry_cropped > proj_defaults.HU_clip_range[1]
    ] = proj_defaults.HU_clip_range[1]
    properties["crop_bbox"] = bbox_gantry
    properties["classes"] = np.max(
        mask_gantry_cropped
    )  # UB changed this code might break nnUNet preprocessing - so be it!
    properties["size_after_cropping"] = img_gantry_cropped.shape

    return img_gantry_cropped, mask_gantry_cropped, properties


def crop_normalize_store(
    case,
    output_folder,
    overwrite,
    mean_all_data,
    std_all_data,
    HU_clip_range,
    crop_axes,
    crop_padding,
    crop_stride,
):
    img, mask, properties = get_img_mask_from_nii(case)
    bbox = get_bbox_from_mask(mask)
    properties["bbox"] = bbox
    img, mask = crop_and_clip(
        HU_clip_range, fg, crop_axes, crop_padding, crop_stride, img, mask
    )
    img = (img - mean_all_data) / std_all_data
    try:
        outfilename = output_folder / ("{}.npz".format(properties["case_id"]))
        if overwrite == True or not outfilename.exists():
            img, mask = np.expand_dims(img, 0), np.expand_dims(mask, 0)
            all_data = np.vstack((img, mask))
            np.savez_compressed(outfilename, data=all_data)
            print("{} is saved".format(properties["case_id"]))
            save_properties(properties, output_folder)
        else:
            pass
    except Exception as e:  # if 'get_ipython' in globals():
        #         print("setting autoreload")
        #         from IPython import get_ipython
        #         ipython = get_ipython()
        #         ipython.run_line_magic('load_ext', 'autoreload')
        #         ipython.run_line_magic('autoreload', '2')

        print("Exception in {}:".format(properties["case_id"]))
        print(e)


def nii_sitk_to_np(nii_fname):
    sitk_img = sitk.ReadImage(nii_fname)
    np_img = sitk.GetArrayFromImage(sitk_img)
    return np_img


def create_filename(output_folder, case_id, ext="pt"):
    return Path(output_folder) / (case_id + "." + ext)


def crop_to_bbox(array, bbox, crop_axes="z", crop_padding=0.1, stride=[1, 1, 1]):
    """
    param array: NP Array to be cropped
    param bbox: Bounding box (3D only supported)
    param crop_axes: by default only crops along z plane, any combination of 'xyz' may be used (e.g., 'xz' will crop in x and z axes)
    param crop_padding: add crop_padding [0,1] fraction to all the planes of cropping.
    param stride: stride in each plane
    """
    assert len(array.shape) == 3, "only supports 3d images"
    bbox_extra_pct = [
        int((bbox[i][1] - bbox[i][0]) * crop_padding / 2) for i in range(len(bbox))
    ]
    bbox_mod = [
        [
            max(0, bbox[j][0] - bbox_extra_pct[j]),
            min(bbox[j][1] + bbox_extra_pct[j], array.shape[j]),
        ]
        for j in range(array.ndim)
    ]
    slices = []
    for dim, axis in zip([0, 1, 2], ["x", "y", "z"]):
        if axis in crop_axes:
            slices.append(slice(bbox_mod[dim][0], bbox_mod[dim][1], stride[dim]))
        else:
            slices.append(slice(0, array.shape[dim], stride[dim]))
    return array[tuple(slices)]


def get_sitk_target_size_from_spacings(sitk_array, spacing_dest):
    sz_source, spacing_source = sitk_array.GetSize(), sitk_array.GetSpacing()
    scale_factor = [a / b for a, b in zip(spacing_source, spacing_dest)]
    sz_dest = [round(a * b) for a, b in zip(sz_source, scale_factor)]
    return sz_dest


class ResampleDatasetniftiToTorch:
    def __init__(
        self,
        proj_defaults,
        minimum_final_spacing,
        enforce_isotropy=True,
        half_precision=False,
        clip_centre=True
    ) -> None:
        """
        proj_defaults is a dict generated by running utils.config.proj_defaults_from_file
        minimum_final_spacing is only used when enforce_isotropy is True
        """
        store_attr('proj_defaults,half_precision,clip_centre')
        self.raw_dataset_properties = load_dict(
            self.proj_defaults.raw_dataset_properties_filename
        )
        self._dataset_size = len(self.raw_dataset_properties)
        self.global_properties = load_dict(
            self.proj_defaults.global_properties_filename
        )

        if enforce_isotropy == True:
            self.spacings = [
                np.maximum(
                    minimum_final_spacing,
                    np.mean(self.global_properties["spacings_median"][1:]),
                ),
            ] * 3  # ignores first index (z) and averages over x and y
            print(
                "Enfore isotropy is true. Setting same spacings based on dataset medians"
            )
        else:
            print("Enfore isotropy is False. Setting spacings based on dataset medians")
            self.spacings = self.global_properties["spacings_median"]

    def resample_cases(
        self,
        multiprocess=True,
        num_processes=8,
        overwrite=False,
        debug=False,
    ):
        print("Resampling dataset to spacing: {0}".format(self.spacings))
        output_subfolders = [
            self.resampling_output_folder / ("images"),
            self.resampling_output_folder / ("masks"),
        ]
        maybe_makedirs(output_subfolders)
        argslist = [
            [
                props ,
                *output_subfolders,
                self.clip_centre,
                self.global_properties,
                self._spacings,
                overwrite,
                self.half_precision
            ]
            for props in self.raw_dataset_properties
        ]
        self.results = multiprocess_multiarg(
            func=niipair_to_torch_wrapper,
            arguments=argslist,
            num_processes=num_processes,
            multiprocess=multiprocess,
            debug=debug,
        )
        self.results = pd.DataFrame(self.results).values
        if self.results.shape[-1] == 3:  # only store if entire dset is processed
            self._store_resampled_dataset_properties()
        else:
            print(
                "Since some files skipped, dataset stats are not being stored. Run ResampleDatasetniftiToTorch.get_tensor_folder_stats separately"
            )
        update_resampling_configs(self.spacings, self.resampling_output_folder)

    def _store_resampled_dataset_properties(self):
        resampled_dataset_properties = dict()
        resampled_dataset_properties["dataset_spacings"] = self.spacings
        resampled_dataset_properties["dataset_max"] = self.results[:, 0].max().item()
        resampled_dataset_properties["dataset_min"] = self.results[:, 1].min().item()
        resampled_dataset_properties["dataset_median"] = np.median(self.results[:, 2])
        resampled_dataset_properties_fname = (
            self.resampling_output_folder / "resampled_dataset_properties.json"
        )
        maybe_makedirs(self.resampling_output_folder)
        print(
            "Writing preprocessing output properties to {}".format(
                resampled_dataset_properties_fname
            )
        )
        save_dict(resampled_dataset_properties, resampled_dataset_properties_fname)

    def generate_bboxes_from_masks_folder(self, bg_label=0,debug=False, num_processes=8):
        masks_folder = self.resampling_output_folder / ("masks")
        print("Generating bbox info from {}".format(masks_folder))
        generate_bboxes_from_masks_folder(
            masks_folder,
            bg_label,
            debug,
            num_processes,
        )

    @property
    def dataset_size(self):
        """The dataset_size property."""
        return self._dataset_size

    @property
    def spacings(self):
        return self._spacings

    @spacings.setter
    def spacings(self, spacings: Union[list, np.ndarray]):
        self._spacings = spacings
        self.resampling_output_folder = spacings

    @property
    def resampling_output_folder(self):
        """The resampling_output_folder property."""
        return self._resampling_output_folder

    @resampling_output_folder.setter
    def resampling_output_folder(self, value):
        if isinstance(value, (int, float)):
            value = [
                value,
            ] * 3
        assert all(
            [isinstance(value, (list, tuple)), len(value) == 3]
        ), "Provide a list with x,y,z spacings"
        self._resampling_output_folder = folder_name_from_list(
            prefix="spc",
            parent_folder=self.proj_defaults.fixed_spacings_folder,
            values_list=value,
        )
        print(
            "Based on output spacings {0},\n setting resampling output folder to : {1}".format(
                self.spacings, self._resampling_output_folder
            )
        )

    def update_specsfile(self):
        specs = {
            "spacings": self.spacings,
            "resampling_output_folder": self.resampling_output_folder,
        }
        specs_file = self.resampling_output_folder.parent / ("resampling_configs")

        try:
            saved_specs = load_dict(specs_file)
            matches = [specs == dic for dic in saved_specs]
            if not any(matches):
                saved_specs.append(specs)
                save_dict(saved_specs, specs_file)
        except:
            saved_specs = [specs]
            save_dict(saved_specs, specs_file)

    def get_tensor_folder_stats(self, debug=True):
        img_filenames = (self.resampling_output_folder / ("images")).glob("*")
        args = [[img_fn] for img_fn in img_filenames]
        results = multiprocess_multiarg(get_tensorfile_stats, args, debug=debug)
        self.results = pd.DataFrame(results).values
        self._store_resampled_dataset_properties()


def get_tensorfile_stats(filename):
    tnsr = torch.load(filename)
    return get_tensor_stats(tnsr)


def get_tensor_stats(tnsr):
    dic = {
        "max": tnsr.max().item(),
        "min": tnsr.min().item(),
        "median": np.median(tnsr),
    }
    return dic

def verify_resampling_configs(resampling_configs_fn):
    output_specs = []
    try:
        saved_specs = load_dict(resampling_configs_fn)
        print(
            "Verifying existing spacings configurations and deleting defunct entries if needed."
        )
        print(
            "Number of fixed_spacings configurations on file: {}".format(
                len(saved_specs)
            )
        )
        for dic in saved_specs:
            if dic["resampling_output_folder"].exists():
                print(str(dic["resampling_output_folder"]) + " verified..")
                output_specs.append(dic)
            else:
                print(
                    str(dic["resampling_output_folder"])
                    + " does not exist. Removing from specs"
                )
        save_dict(output_specs, resampling_configs_fn)
    except:
        print("Resampling configs file either does not exist or is invalid")


def update_resampling_configs(spacings, resampling_output_folder):
    specs = {
        "spacings": spacings,
        "resampling_output_folder": resampling_output_folder,
    }
    specs_file = resampling_output_folder.parent / ("resampling_configs")
    verify_resampling_configs(specs_file)
    try:
        output_specs = load_dict(specs_file)
    except:
        print("Creating new reesampling configs file.")
        output_specs = []
    matches = [specs == dic for dic in output_specs]
    if not any(matches):
        output_specs.append(specs)
        save_dict(output_specs, specs_file)
    else:
        print("Set of specs already exist in a folder. Nothing is changed.")


class GetSizeDest(ItemTransform):
    def __init__(self, spacings):
        store_attr()

    def encodes(self, x):
        sz_dests = [get_sitk_target_size_from_spacings(x_, self.spacings) for x_ in list(x)]
        if test_eq(*sz_dests): 
            print ("Differing sized arrays: {0} and {1}".format(
            x[0].GetSize(), x[1].GetSize()
        )
                   )
        return x[0], x[1], sz_dests[0]


class SITKToNumpy(ItemTransform):

    def __init__(self):
        store_attr()
    def encodes(self, x):
        x = [sitk.GetArrayFromImage(xx) for xx in x]
        return x


class NumpyToTorch(ItemTransform):
    def __init__(self,img_dtype = torch.float32, mask_dtype= torch.uint8): store_attr()
    def encodes(self, x):
        img, mask = x
        img = torch.tensor(img,dtype = self.img_dtype)
        if mask.dtype!=np.uint8 :
            mask =mask.astype(np.uint8)
        mask= torch.tensor(mask,dtype=self.mask_dtype)
        return img,mask


class GetFilenames(ItemTransform):
    def encodes(self, single_case_properties):
        img_fname = single_case_properties["properties"]["img_file"]
        mask_fname = single_case_properties["properties"]["mask_file"]
        return img_fname, mask_fname


class TransposeSITKImageMask(ItemTransform):
    """
    given default sequence assumes data is in dicom orientation
    """

    def __init__(self, sequence=[2, 1, 0]):
        store_attr()

    def encodes(self, x):
        permute = lambda x: torch.permute(x, dims=self.sequence)
        return list(map(permute, x))

def niipair_to_torch_wrapper(
    single_case_properties,
    output_img_folder,
    output_mask_folder,
    clip_centre,
    global_properties,
    spacings,
    overwrite,
    half_precision
):
        N = NiipairToTorch(
         
            output_img_folder, output_mask_folder, global_properties, spacings, half_precision,'dataset',clip_centre
        )

        return N.process_sitk_to_tensors(single_case_properties, overwrite=overwrite)

class HalfPrecision(ItemTransform):
    def encodes(self,x):
        img,mask =x
        img = img.to(torch.float16)
        return img,mask

class NiipairToTorch(DictToAttr):
    """
    Resizes niipair to target dataset spacing and resamples accordingly.
    Clips to mean /std ranges
    Creates numpy for preprocessing. Then saves as a img and mask tensor in images and masks folders

    """

    def __init__(
        self,
        output_img_folder,
        output_mask_folder,
        global_properties,
        spacings,
        half_precision,
        mean_std_mode:str='dataset',
        clip_centre=True
    ):
        assert mean_std_mode in ['dataset','fg'], "Select either dataset mean/std or fg mean/std for normalization"
        self.assimilate_dict(global_properties)
        self.set_normalization_values(mean_std_mode)
        store_attr(but="global_properties")

    def proceed(self, overwrite):
        if any(write_files_or_not(self.output_filenames, overwrite)) == False:
            return False
        else:
            return True

    def process_sitk_to_tensors(self, single_case_properties, overwrite=True):
        self._create_output_filenames(single_case_properties)
        if self.proceed(overwrite) == True:

            pipeline1 = Pipeline([GetFilenames, ReadSITKImgMask, GetSizeDest(self.spacings)])
            img, mask, sz_dest = pipeline1(single_case_properties)
            pipeline2 = [
                    SITKToNumpy,
                    NumpyToTorch(),
                    TransposeSITKImageMask,  
                    Unsqueeze,
                    Unsqueeze,
                    ResizeBatch(target_size=sz_dest),
                    Squeeze(0),
                    Squeeze(0),
                ]

            if self.half_precision==True: pipeline2.append(HalfPrecision)

            if self.clip_centre==True:
                pipeline2.append(
                    ClipCenter(
                        clip_range=self.intensity_clip_range,
                        mean=self.mean,
                        std=self.std,
                    ),
                )
            pipeline2 = Pipeline(pipeline2)
            self.img, self.mask = pipeline2([img, mask])
            self.write_tensors_to_disc()
            return get_tensor_stats(self.img)
        else:
            return 1

    def write_tensors_to_disc(self):
        for arr , fn in zip([self.img, self.mask], self.output_filenames):
            torch.save(arr,fn)

    def _create_output_filenames(self, single_case_properties):
        case_id = single_case_properties["case_id"]
        self.output_filenames = [
            create_filename(output_folder, case_id, ext="pt")
            for output_folder in [self.output_img_folder, self.output_mask_folder]
        ]
    def set_normalization_values(self,mean_std_mode):
        if mean_std_mode == 'dataset':
            self.mean = self.mean_dataset_clipped 
            self.std = self.std_dataset_clipped
        else:
            self.mean = self.mean_fg
            self.std = self.std_fg

# %%
if __name__ == "__main__":
    globalp = load_dict(Path("/s/fran_storage/projects/lax/global_properties.json"))
    case_props = load_dict(
        Path("/s/fran_storage/projects/lax/raw_dataset_properties.pkl")
    )
# %%
    single_case_properties = case_props[0]
    output_folder = Path("/home/ub/tmp")
    output_masks = output_folder / ("masks")
    output_imgs = output_folder / ("imgs")
    maybe_makedirs([output_imgs, output_masks])
# %%
    N = NiipairToTorch(
        output_mask_folder=output_masks,
        output_img_folder=output_imgs,
        global_properties=globalp,
        spacings=[1.0, 2.0, 3.0],
        half_precision=True
    )
# %%
    N.process_sitk_to_tensors(single_case_properties, overwrite=True)
# %%
    fns = N.output_filenames
    x = [torch.load(fn) for fn in fns]
    x = [xx.permute(2,1,0) for xx in x]
    [s.dtype for s in x]
# %%
    ImageMaskViewer(x)
    
# %%
    ###################kl###################################################################
# %% [markdown]
    ## Creating stage0 dataset from nifti
# %%
    folder = "/s/datasets/preprocessed/fixed_spacings/lits/spc_100_100_200"
    folder_ni = "/s/fran_storage/datasets/raw_data/lits/"
    res = verify_dataset_integrity(folder_ni, debug_mode=False, fix=True)
# %%
    resampling_output_folder = Path(
        "/s/datasets/preprocessed/fixed_spacings/lits/spc_100_100_200/"
    )
    spacings = [0.77, 0.77, 1.0]
    resampling_output_folder = Path(
        "/s/datasets/preprocessed/fixed_spacings/lits/spc_077_077_100/"
    )
    update_resampling_configs(spacings, resampling_output_folder)
    dd = load_dict(resampling_output_folder.parent / ("resampling_configs"))
    pp(dd)
    

    from nbs.common_imports import *
    from fran.preprocessing.stage0_preprocessors import *
    from fran.preprocessing.stage1_preprocessors import *

# %%
    spacings = [.8,.8,1.5]
    P = Project(project_title="litsmc"); proj_defaults= P
# %%

    Resampler = ResampleDatasetniftiToTorch(
                    proj_defaults,
                    minimum_final_spacing=0.5,
                    enforce_isotropy=False,
                    half_precision=True,
                    clip_centre=False
                )

    Resampler.spacings = spacings
# %%
    R = ResampleDatasetniftiToTorch(
        proj_defaults,
        minimum_final_spacing=0.5,
        enforce_isotropy=False,
    )
# %%
    R.get_tensor_folder_stats(debug=False)
# %%

# %%
    R.spacings = [3.0, 1, 1]
    project_title = proj_defaults.project_title
    case_id = "00022"
# %%
    debug = False
    multiprocess = True
    num_processes=16
    I.Resampler.resample_cases(debug=debug, overwrite=True, multiprocess=multiprocess)
    fldr = Path("/s/fran_storage/datasets/preprocessed/fixed_spacings/nodes/spc_078_078_375/masks/")
    generate_bboxes_from_masks_folder(fldr)
# %%
    masks_folder = I.Resampler.resampling_output_folder / ("masks")
    print("Generating bbox info from {}".format(masks_folder))
    generate_bboxes_from_masks_folder(
        masks_folder,
        debug,
        num_processes,
    )

    bg_label=0
    mask_files = masks_folder.glob("*pt")
    arguments = [
        [x,  bg_label] for x in mask_files
    ]  # 0.2 factor for thresholding as kidneys are small on low-res imaging and will be wiped out by default threshold 3000
    bboxes = multiprocess_multiarg(
        func=bboxes_function_version,
        arguments=arguments,
        num_processes=num_processes,
        debug=debug,
    )

# %%
#     aa = pipeline2[0]([img,mask])
#     aa = pipeline2[1](aa)
#     aa = pipeline2[2](aa)
#     aa = pipeline2[3](aa)
#     aa = pipeline2[4](aa)
#     aa = pipeline2[5](aa)
#     aa = pipeline2[6](aa)
#     aa = pipeline2[7](aa)
# # %%
#     aa = pipeline2[8](aa)
#
# # %%
#     args = [[output_folder, patch_size, inf, stride] for inf in stage0_bbox]
#     multiprocess_multiarg(patch_generator_wrapper, args, debug=False)
#
#     spacing_dest = R.global_properties["spacings_median"]
#     img_fname = R.raw_dataset_properties[0]["properties"]["img_file"]
#     mask_fname = R.raw_dataset_properties[0]["properties"]["mask_file"]
#
# # %%
#
#     single_case_properties = [
#         p for p in R.raw_dataset_properties if p["case_id"] == "00063"
#     ][0]
#
