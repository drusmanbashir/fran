# %%
from fastcore.all import store_attr
import ast
from fastcore.basics import GetAttr
import torch
import h5py
import numpy as np

from fran.preprocessing.datasetanalyzers import case_analyzer_wrapper, get_img_mask_filepairs, get_intensity_range, get_means_voxelcounts, get_std_numerator, percentile_range_to_str
from fran.utils.fileio import load_dict, save_dict
from fran.utils.helpers import multiprocess_multiarg


class GlobalProperties(GetAttr):
    _default = 'project'
    """
    Input: raw_data_folder and project_title
    Outputs: (after running process_cases()) :
         2)Dataset global properties (including mean and std of voxels inside bboxes)
    """

    def __init__(
        self, project, bg_label=0, percentile_range: list = [0.5, 99.5],clip_range=None
    ):
        assert all(
            [isinstance(x, float) for x in percentile_range]
        ), "Provide float values for clip percentile_range. Corresponding HUs will be stored in file for future processing."
        store_attr("project,bg_label,percentile_range,clip_range")
        self.list_of_raw_cases = get_img_mask_filepairs(
            parent_folder=self.project.raw_data_folder
        )
        self.case_properties = load_dict(self.raw_dataset_properties_filename)


    def store_projectwide_properties(self):
        '''
        Start here.
        '''
        
        with h5py.File(self.bboxes_voxels_info_filename, "r") as h5f_file:
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
        print(
            "\nWriting dataset global_properties to json file: {}".format(
                self.global_properties_filename
            )
        )
        save_dict(global_properties, self.global_properties_filename)

    def compute_std_mean_dataset(
        self, num_processes=32, multiprocess=True, debug=False
    ):

        """
        Stage 2:
        Requires global_properties (intensity_percentile_fg range) for clipping ()
        """

        try:
            self.global_properties = load_dict(self.global_properties_filename)
        except:
            print(
                "Run process_cases first. Correct global_properties not found in file {}, or file does not exist".format(
                    self.raw_dataset_properties_filename
                )
            )

        percentile_label, intensity_percentile_range=  get_intensity_range(self.global_properties)
        if not self.clip_range: self.user_query_clip_range(intensity_percentile_range)
        self._compute_dataset_mean(num_processes,multiprocess,debug)
        self._compute_dataset_std(num_processes,multiprocess,debug)
        self.global_properties['intensity_clip_range']= self.clip_range
        self.global_properties[percentile_label]= intensity_percentile_range
        self.global_properties["mean_dataset_clipped"] = self.dataset_mean.item()
        self.global_properties["std_dataset_clipped"] = self.std.item()
        self.global_properties["total_voxels"] = int(self.total_voxels)
        print(
            "Saving updated global_properties to file {}".format(
                self.global_properties_filename
            )
        )
        save_dict(self.global_properties, self.global_properties_filename,sort=True)




    def _compute_dataset_mean(self,num_processes,multiprocess,debug):
        img_fnames = [case_[0] for case_ in self.list_of_raw_cases]
        args = [[fname, self.clip_range] for fname in img_fnames]

        print("Computing means from all nifti files (clipped to {})".format(self.clip_range))
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

    def _compute_dataset_std(self,num_processes,multiprocess,debug):
        img_fnames = [case_[0] for case_ in self.list_of_raw_cases]
        args = [[fname, self.dataset_mean, self.clip_range] for fname in img_fnames]
        print(
            "Computing std from all nifti files, using global mean computed above (clipped to {})".format(
                self.clip_range
            )
        )
        std_num = multiprocess_multiarg(get_std_numerator, args,num_processes=num_processes,multiprocess=multiprocess, debug=debug)
        std_num = torch.tensor(std_num)
        self.std = torch.sqrt(std_num.sum() / self.total_voxels)

    def user_query_clip_range(self,intensity_percentile_range):
            try:
                self.clip_range = input("A Clip range has not been given. Press enter to accept clip range based on intensity-percentiles (i.e.{}) or give a new range now: ".format(intensity_percentile_range))
                if len(self.clip_range) == 0: self.clip_range = intensity_percentile_range
                else: self.clip_range = ast.literal_eval(self.clip_range) 
            except:
                print("A valid clip_range is not entered. Using intensity-default")
                self.clip_range = intensity_percentile_range



# %%
if __name__ == "__main__":
    from fran.utils.common import *
    P = Project(project_title="litsmallx");
    G = GlobalProperties(P)
    G.store_projectwide_properties()
# %%
    G.compute_std_mean_dataset()

# %%
