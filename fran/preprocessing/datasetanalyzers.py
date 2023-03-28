# %%
from fastai.vision.augment import store_attr
import numpy as np
import ast
from fran.transforms.totensor import ToTensorT
from fran.utils.helpers import *
import h5py

# sys.path += ["/home/ub/Dropbox/code/fran"]
from fran.utils.helpers import *
from fran.utils.fileio import *
import cc3d

from fran.utils.image_utils import get_img_mask_from_nii
from fran.utils.sitk_utils import SITKImageMaskFixer


def get_intensity_range(global_properties: dict) -> list:
    key_idx = [key for key in global_properties.keys() if "intensity_percentile" in key][0]
    intensity_range = global_properties[key_idx]
    return key_idx, intensity_range

@str_to_path()
def get_img_mask_filepairs(parent_folder: Union[str,Path]):
    '''
    param: parent_folder. Must contain subfolders labelled masks and images
    Files in either folder belonging to a given case should be identically named.
    '''

    imgs_folder = Path(parent_folder)/'images'
    masks_folder= Path(parent_folder)/'masks'
    imgs_all=list(imgs_folder.glob('*'))
    masks_all=list(masks_folder.glob('*'))
    assert (len(imgs_all)==len(masks_all)), "{0} and {1} folders have unequal number of files!".format(imgs_folder,masks_folder)
    img_mask_filepairs= []
    for img_fn in imgs_all:
            mask_fn = masks_folder/img_fn.name
            assert mask_fn.exists(), f"{mask_fn} doest not exist, corresponding tto {img_fn}"
            img_mask_filepairs.append([img_fn,mask_fn])
    return img_mask_filepairs


@str_to_path(0)
def verify_dataset_integrity(folder:Path, debug=False,fix=False):
    '''
    folder has subfolders images and masks
    '''
    print("Verifying dataset integrity")
    subfolder = list(folder.glob("mask*"))[0]
    args = [[fn,fix] for fn in subfolder.glob("*")]
    res = multiprocess_multiarg(verify_img_mask_match,args,debug=debug)
    errors = [item for item in res if re.search("mismatch", item[0],re.IGNORECASE)]
    if len(errors)>0:
        outname = folder/("errors.txt")
        print(f"Errors found saved in {outname}")
        save_list(errors,outname)
        res.insert(0,errors)
    else:
        print("All images and masks are verified for matching sizes and spacings.")
    return res


def verify_datasets_integrity(folders:list, debug=False,fix=False)->list:
    folders = listify(folders)
    res =[]
    for folder in folders:
        res.extend(verify_dataset_integrity(folder,debug,fix))
    return res
        
    

def verify_img_mask_match(mask_fn:Path,fix=False):
    img_fn = mask_fn.str_replace('masks','images')
    if '.pt' in mask_fn.name:
        return verify_img_mask_torch(mask_fn)
    else:
        S = SITKImageMaskFixer(img_fn,mask_fn)
        S.process(fix=fix)
        return S.log

@str_to_path()
def verify_img_mask_torch(mask_fn:Path):
    if isinstance(mask_fn,str): mask_fn = Path(mask_fn)
    img_fn = mask_fn.str_replace('masks','images')
    img,mask = list(map(torch.load,[img_fn,mask_fn]))
    if img.shape!=mask.shape:
        print(f"Image mask mismatch {mask_fn}")
        return '\nMismatch',img_fn,mask_fn,str(img.shape),str(mask.shape)

# %%
def get_cc3d_stats(mask, label, k_largest, dusting_threshold: int = 0):
    if torch.is_tensor(mask):
        mask = mask.numpy()
    mask_tmp = np.copy(mask.astype(np.uint8))
    mask_tmp[mask != label] = 0
    if dusting_threshold > 0:
        mask_tmp = cc3d.dust(
            mask_tmp, threshold=dusting_threshold, connectivity=26, in_place=True
        )
    mask_cc, N = cc3d.largest_k(
        mask_tmp, k=k_largest, return_N=True
    )  # up to 5 largest cysts or tumours max
    stats = cc3d.statistics(mask_cc)
    return stats


class BBoxesFromMask(object):
    """ """

    def __init__(
        self,
        filename,
        mask,
        proj_defaults,
        label_settings=None,
        dusting_threshold_factor=1,
    ):
        self.proj_defaults = proj_defaults
        self.mask = mask
        self.label_settings = label_settings
        if not self.label_settings:
            self.label_settings = proj_defaults.mask_labels
        self.dusting_threshold_factor = dusting_threshold_factor  # a multiplier when processing subsampled datasets having proportionately fewer voxels for dusting

        case_id = get_case_id_from_filename(None, filename)
        self._bboxes_info = {
            "case_id": case_id,
            "filename": filename,
        }

    def __call__(self):
        bboxes_all = []
        mask_all_fg = self.mask.copy()
        mask_all_fg[mask_all_fg > 1] = 1
        for label,label_info in self.label_settings.items():
            label = int(label)
            if label in self.mask:
                stats = {"label": label}
                stats.update(
                    get_cc3d_stats(
                        self.mask,
                        label,
                        k_largest=label_info["k_largest"],
                        dusting_threshold=int(
                            label_info["dusting_threshold"]
                            * self.dusting_threshold_factor
                        ),
                    )
                )
                bboxes_all.append(stats)
        stats = {"label": "all_fg"}
        stats.update(
            get_cc3d_stats(
                mask_all_fg,
                1,
                1,
                dusting_threshold=3000 * self.dusting_threshold_factor,
            )
        )
        bboxes_all.append(stats)
        self._bboxes_info["bbox_stats"] = bboxes_all
        return self._bboxes_info

    @property
    def bboxes_info(self):
        return self._bboxes_info


class SingleCaseAnalyzer:
    """
    Loads nifty -> nifty properties (spacings bbox)
    returns numpy array containing bbox voxels only
    """

    def __init__(
            self, project_title, case_files_tuple, percentile_range:list, outside_value=0
    ):
        """
        param: case_files_tuple are two nii_gz files (img,mask)
        """
        assert isinstance(case_files_tuple, list) or isinstance(
            case_files_tuple, tuple
        ), "case_files_tuple must be either a list or a tuple"
        self.case_id = get_case_id_from_filename(None, case_files_tuple[0])
        store_attr("case_files_tuple,outside_value,percentile_range")

    def load_case(self):
        self.img, self.mask, self._properties = get_img_mask_from_nii(
            self.case_files_tuple, outside_value=self.outside_value
        )

    def get_bbox_only_voxels(self):
        return self.img[self.mask != self.outside_value]


    @property
    def properties(self):
        return self._properties if self._properties else print("Run load_case() first")

    @property
    def case_id(self):
        return self._case_id
    @case_id.setter
    def case_id(self,value): self._case_id = value

class MultiCaseAnalyzer(object):
    """
    Input: raw_data_folder and project_title
    Outputs: (after running process_cases()) :
         1)bbox voxels stacked inside an h5py file,
         2)Dataset global properties (including mean and std of voxels inside bboxes)
    """

    def __init__(
        self, proj_defaults, outside_value=0, percentile_range: list = [0.5, 99.5],clip_range=None
    ):
        assert all(
            [isinstance(x, float) for x in percentile_range]
        ), "Provide float values for clip percentile_range"
        store_attr("outside_value,percentile_range,clip_range")
        self.project_title = proj_defaults.project_title
        self.properties_outfilename = proj_defaults.raw_dataset_properties_filename
        self.global_properties_outfilename = proj_defaults.global_properties_filename
        self.h5f_fname = proj_defaults.bboxes_voxels_info_filename
        self.list_of_raw_cases = get_img_mask_filepairs(
            parent_folder=proj_defaults.raw_data_folder
        )


    def store_projectwide_properties(self):
        with h5py.File(self.h5f_fname, "r") as h5f_file:
            cases = np.concatenate(
                [h5f_file[case][:] for case in h5f_file.keys()]
            )  # convert h5file cases into an array
        intensity_range = np.percentile(cases, self.percentile_range)
        global_properties = {}
        dataset_size = len(self.list_of_raw_cases)
        global_properties["project_title"] = self.project_title
        global_properties["dataset_size"] = dataset_size

        global_properties["mean_fg"] = np.double(
            cases.mean()
        )  # np.single is not JSON serializable
        global_properties["std_fg"] = np.double(cases.std())
        global_properties[
            percentile_range_to_str(self.percentile_range) + "_fg"
        ] = (
            intensity_range
        ).tolist()  # np.array is not JSON serializable
        global_properties["max_fg"] = np.double(cases.max())
        global_properties["min_fg"] = np.double(cases.min())

        all_spacings = np.zeros((dataset_size, 3))
        for ind, case_ in enumerate(self.case_properties):
            # if "case_id" in case_:
                spacing = case_["properties"]["itk_spacing"]
                all_spacings[ind, :] = spacing

        spacings_median = np.median(all_spacings, 0)
        global_properties["spacings_median"] = spacings_median.tolist()

        # global_properties collected. Now storing
        # self.case_properties.append(dataset_globals)
        print(
            "\nWriting dataset global_properties to json file: {}".format(
                self.properties_outfilename
            )
        )
        save_dict(self.case_properties, self.properties_outfilename)
        save_dict(global_properties, self.global_properties_outfilename)

    def get_nii_bbox_properties(
        self, num_processes=8, overwrite=False, multiprocess=True, debug=False
    ):
        """
        Stage 1: derives datase properties especially intensity_fg
        """

        self.case_properties = []
        if overwrite == True or not self.h5f_fname.exists():
            get_voxels = True
            print("Voxels inside bbox will be dumped into: {}".format(self.h5f_fname))
        else:
            get_voxels = False
            print("Voxels file: {0} exists".format(self.h5f_fname))
        args_list = [
            [case_tuple, self.project_title, self.outside_value, get_voxels]
            for case_tuple in self.list_of_raw_cases
        ]
        self.outputs = multiprocess_multiarg(
            func=case_analyzer_wrapper,
            arguments=args_list,
            num_processes=num_processes,
            multiprocess=multiprocess,
            debug=debug,
        )
        h5f = h5py.File(self.h5f_fname, "w") if get_voxels == True else None
        for output in self.outputs:
            self.case_properties.append(output["case"])
            if h5f:
                h5f.create_dataset(output["case"]["case_id"], data=output["voxels"])
        if h5f:
            h5f.close()

    def user_query_clip_range(self,intensity_percentile_range):
            try:
                self.clip_range = input("A Clip range has not been given. Press enter to accept clip range based on intensity-percentiles (i.e.{}) or give a new range now: ".format(intensity_percentile_range))
                if len(self.clip_range) == 0: self.clip_range = intensity_percentile_range
                else: self.clip_range = ast.literal_eval(self.clip_range) 
            except:
                print("A valid clip_range is not entered. Using intensity-default")
                self.clip_range = intensity_percentile_range

    def compute_dataset_mean(self,num_processes,multiprocess,debug):
        img_fnames = [case_[0] for case_ in self.list_of_raw_cases]
        args = [[fname, self.clip_range] for fname in img_fnames]

        print("Computing means from all nifty files (clipped to {})".format(self.clip_range))
        means_sizes = multiprocess_multiarg(
            get_means_voxelcounts,
            args,
            num_processes=num_processes,
            multiprocess=multiprocess,
            debug=debug,
        )
        means_sizes=  torch.tensor(means_sizes)
        weighted_mn = torch.multiply(
            means_sizes[:, 0], means_sizes[:, 1]
        ) 

        self.total_voxels = means_sizes[:, 1].sum()
        self.dataset_mean = weighted_mn.sum() / self.total_voxels

    def compute_dataset_std(self,num_processes,multiprocess,debug):
        img_fnames = [case_[0] for case_ in self.list_of_raw_cases]
        args = [[fname, self.dataset_mean, self.clip_range] for fname in img_fnames]
        print(
            "Computing std from all nifty files, using global mean computed above (clipped to {})".format(
                self.clip_range
            )
        )
        std_num = multiprocess_multiarg(get_std_numerator, args,num_processes=num_processes,multiprocess=multiprocess, debug=debug)
        std_num = torch.tensor(std_num)
        self.std = torch.sqrt(std_num.sum() / self.total_voxels)

    def compute_std_mean_dataset(
        self, num_processes=32, multiprocess=True, debug=False
    ):

        """
        Stage 2:
        Requires global_properties (intensity_percentile_fg range) for clipping ()
        """

        try:
            self.global_properties = load_dict(self.global_properties_outfilename)
        except:
            print(
                "Run process_cases first. Correct global_properties not found in file {}, or file does not exist".format(
                    self.properties_outfilename
                )
            )

        percentile_label, intensity_percentile_range=  get_intensity_range(self.global_properties)
        if not self.clip_range: self.user_query_clip_range(intensity_percentile_range)
        self.compute_dataset_mean(num_processes,multiprocess,debug)
        self.compute_dataset_std(num_processes,multiprocess,debug)
        self.global_properties['intensity_clip_range']= self.clip_range
        self.global_properties[percentile_label]= intensity_percentile_range
        self.global_properties["mean_dataset_clipped"] = self.dataset_mean.item()
        self.global_properties["std_dataset_clipped"] = self.std.item()
        self.global_properties["total_voxels"] = int(self.total_voxels)
        print(
            "Saving updated global_properties to file {}".format(
                self.properties_outfilename
            )
        )
        save_dict(self.global_properties, self.global_properties_outfilename,sort=True)


def case_analyzer_wrapper(
    case_files_tuple,
    project_title,
    outside_value,
    get_voxels=True,
    percentile_range=[0.5, 99.5],
):
    S = SingleCaseAnalyzer(
        project_title=project_title,
        case_files_tuple=case_files_tuple,
        outside_value=outside_value,
        percentile_range=percentile_range,
    )
    S.load_case()
    case_ = dict()
    case_["case_id"] = S.case_id
    if get_voxels == True:
        voxels = S.get_bbox_only_voxels()
        S.properties["mean_fg"] = int(voxels.mean())
        S.properties["min_fg"] = int(voxels.min())
        S.properties["max_fg"] = int(voxels.max())
        S.properties["std_fg"] = int(voxels.std())
        S.properties[percentile_range_to_str(percentile_range)] = np.percentile(
            voxels, percentile_range
        )
    case_["properties"] = S.properties
    output = {"case": case_, "voxels": voxels}
    return output


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
        img = torch.clip(img,min=clip_range[0],max=clip_range[1])
    var = (img - dataset_mean) ** 2
    var_sum = var.sum()
    return var_sum


def get_means_voxelcounts(img_fname, clip_range=None):
    img = ToTensorT(torch.float32)(img_fname)
    if clip_range is not None:
        img = torch.clip(img,min=clip_range[0],max=clip_range[1])
    return img.mean().item(), img.numel()


def bboxes_function_version(
    filename, proj_defaults, dusting_threshold_factor=1, label_settings=None
):

    if "nii.gz" in str(filename):
        mask_sitk = sitk.ReadImage(str(filename))
        mask = sitk.GetArrayFromImage(mask_sitk)
    elif ".pt" in str(filename):
        mask = torch.load(filename)
        if isinstance(mask, dict):
            mask = mask["mask"]
        mask = mask.numpy()

    A = BBoxesFromMask(
        filename, mask, proj_defaults, label_settings, dusting_threshold_factor
    )
    return A()


# %%
if __name__ == "__main__":
    
    from fran.utils.common import *
    P = Project(project_title="lits"); proj_defaults= P.proj_summary

# %% [markdown]
# %%
    M = MultiCaseAnalyzer(proj_defaults, outside_value=0)

# %%
    M.get_nii_bbox_properties(
        num_processes=24, debug=False, overwrite=True, multiprocess=True
    )

# %%

    M.store_projectwide_properties()
    M.compute_std_mean_dataset()
# %%
    M.global_properties = load_dict(M.global_properties_outfilename)
    percentile_label, intensity_percentile_range=  get_intensity_range(M.global_properties)
    clip_range=[-5,200]
# %%
    img_fnames = [case_[0] for case_ in M.list_of_raw_cases]
    img_fname = img_fnames[0]
# %%
    img = ToTensorT()(img_fname)
    if clip_range is not None:
        img = torch.clip(img,min=clip_range[0],max=clip_range[1])
    var = (img - dataset_mean) ** 2
 
# %%
    args = [[fname, M.dataset_mean, M.clip_range] for fname in img_fnames]
    print(
        "Computing std from all nifty files, using global mean computed above (clipped to {})".format(
            M.clip_range
        )
    )

    M.compute_std_mean_dataset(debug=True)
# %% [markdown]
    # # Resample Nifty to Torch
# %% [markdown]
    # ### KITS19

# %%
    R = ResampleDatasetNiftyToTorch(
        proj_defaults, minimum_final_spacing=0.0, enforce_isotropy=False
    )
# %%
    R.spacings[-1] = 1.5
# %%

    R.resample_cases(debug=False, overwrite=True, multiprocess=False)

# %%
    R.resample_cases(debug=False, num_processes=8, multiprocess=True, overwrite=True)

# %%
# %%

    res = multiprocess_multiarg(bboxes_function_version, arguments, 16, debug=False)

# %%

    stats_outfilename_kits21 = proj_defaults.stage0_folder / "bboxes_info"
    save_dict(res, stats_outfilename_kits21)

# %%
    # # Getting bbox properties from preprocessed images
    # ### KITS21 cropped nifty files bboxes
    M.std = torch.sqrt(std_num.sum() / M.total_voxels)

# %%
    P = Project(project_title="lits"); proj_defaults= P.proj_summary
    masks_folder_nii = proj_defaults.stage1_folder / ("cropped/images_nii/masks")
    masks_filenames = get_fileslist_from_path(masks_folder_nii, ext=".nii.gz")
    arguments = [[x, proj_defaults] for x in masks_filenames]

# %%

    res = multiprocess_multiarg(bboxes_function_version, arguments, 48, debug=False)
# %%

    import collections
    od = collections.OrderedDict(sorted(gp.items()))
# %%

    stats_outfilename_kits21 = (
        proj_defaults.stage1_folder / ("cropped/images_nii/masks")
    ).parent / ("bboxes_info")
    save_dict(res, stats_outfilename_kits21)
