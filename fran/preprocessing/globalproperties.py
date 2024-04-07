# %%
from fastcore.all import store_attr
from fran.utils.helpers import pbar
import random
from monai.utils.misc import progress_bar
import pandas as pd
import ast
from fastcore.basics import GetAttr
from pathlib import Path
import torch
import numpy as np
import ipdb

from fran.utils.string import info_from_filename
tr = ipdb.set_trace


from fran.preprocessing.datasetanalyzers import get_img_mask_filepairs,case_analyzer_wrapper,  get_intensity_range, get_means_voxelcounts, get_std_numerator, import_h5py, percentile_range_to_str
from fran.utils.fileio import load_dict, save_dict
from fran.utils.helpers import multiprocess_multiarg

def unique_idx(total_len,start=1):
        for x in range(start, total_len+1):
            yield(x)


class GlobalProperties(GetAttr):
    _default = 'project'
    """
    Input: raw_data_folder and project_title
    Outputs: (after running process_cases()) :
         2)Dataset global properties (including mean and std of voxels inside bboxes)
    """

    def __init__(
        self, project, bg_label=0,max_cases:int =None, percentile_range: list = [0.5, 99.5],clip_range=None
    ):
        assert all(
            [isinstance(x, float) for x in percentile_range]
        ), "Provide float values for clip percentile_range. Corresponding HUs will be stored in file for future processing."
        store_attr("project,bg_label,percentile_range,clip_range")
        self.case_ids , self.img_filenames= self.collate_project_cases() #        self.h5ns =[]
        self.cases_for_sampling=self.sample_cases(max_cases)
        for ds in self.global_properties['datasources']:
            h5fn = Path(ds['h5_fname'])
            assert h5fn.exists(), "fg voxels not processed for datasource folder {}".format(ds['folder'])


    def collate_project_cases(self):
        cases = get_img_mask_filepairs(
            parent_folder=self.project.raw_data_folder
        )
        case_ids, img_filenames=[],[]
        for fpair in cases:
            case_id = info_from_filename(fpair[0].name, True)['case_id']
            case_ids.append(case_id)
            img_filenames.append(fpair[0])
        return case_ids,img_filenames


    def sample_cases(self,max_cases):
        if max_cases:
            cases_for_sampling= random.sample(self.case_ids,np.minimum(len(self.case_ids),max_cases))
        else:
            cases_for_sampling= self.case_ids
        return cases_for_sampling


    def retrieve_h5_voxels(self):
        voxels=[]
        h5py = import_h5py()
        for ds in self.global_properties['datasources']:
            ds_name =ds['ds']
            h5fn = ds['h5_fname']
            cases_ds = [ cid for cid in self.cases_for_sampling if cid.split("_")[0] == ds_name ]
            with h5py.File(h5fn, "r") as h5f_file:
                for cid in cases_ds:
                    cs = h5f_file[cid]
                    voxels.append(cs[:])
                    # props = {'case_id':cid, 'spacing': cs.attrs['spacing'] ,'labels':cs.attrs['labels']}
                    # self.case_properties.append(props)
                # cases_all.append(np.concatenate([h5f_file[case][:] for case in h5f_file.keys() if case in self.case_ids]))

        voxels = np.concatenate(voxels)
        return voxels

    def retrieve_h5_properties(self):
        h5py = import_h5py()
        case_properties=[]
        for ds in self.global_properties['datasources']:
            ds_name =ds['ds']
            h5fn = ds['h5_fname']
            cases_ds = [ cid for cid in self.cases_for_sampling if cid.split("_")[0] == ds_name ]
            with h5py.File(h5fn, "r") as h5f_file:
                for cid in pbar(cases_ds):
                    cs = h5f_file[cid]
                    props = {'case_id':cid, 'spacing': cs.attrs['spacing'] ,'labels':cs.attrs['labels']}
                    case_properties.append(props)

        assert(len(self.case_ids) == len(case_properties)), "Mismatch in case_ids and case_properties"
        return case_properties

    def store_projectwide_properties(self):
        '''
        Stage 1.
        Start here.

        '''
        cases = self.retrieve_h5_voxels()
        self.case_properties = self.retrieve_h5_properties()

        # convert h5file cases into an array
        intensity_range = np.percentile(cases, self.percentile_range)
        sample_size = len(self.case_ids)
        self.global_properties["sample_size"] = sample_size

        self.global_properties["mean_fg"] = np.double(
            cases.mean()
        )  # np.single is not JSON serializable
        self.global_properties["std_fg"] = np.double(cases.std())
        self.global_properties[
            percentile_range_to_str(self.percentile_range) + "_fg"
        ] = (
            intensity_range
        ).tolist()  # np.array is not JSON serializable
        self.global_properties["max_fg"] = np.double(cases.max())
        self.global_properties["min_fg"] = np.double(cases.min())

        all_spacings = np.zeros((sample_size, 3))
        for ind, case_ in enumerate(self.case_properties):
            # if "case_id" in case_:
                spacing = case_["spacing"]
                all_spacings[ind, :] = spacing

        spacing_median = np.median(all_spacings, 0)
        self.global_properties["spacing_median"] = spacing_median.tolist()

        # self.global_properties collected. Now storing
        print(
            "\nWriting dataset self.global_properties to json file: {}".format(
                self.global_properties_filename
            )
        )
        save_dict(self.global_properties, self.global_properties_filename)

    def compute_std_mean_dataset(
        self, num_processes=32, multiprocess=True, debug=False
    ):

        """
        Stage 2:
        Requires global_properties (intensity_percentile_fg range) for clipping ()
        """

        percentile_label, intensity_percentile_range=  get_intensity_range(self.global_properties)
        if not self.clip_range: self.clip_range =intensity_percentile_range
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


    # def collate_labels(self):


    def _compute_dataset_mean(self,num_processes,multiprocess,debug):
        args = [[fname, self.clip_range] for fname in self.img_filenames]

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
        args = [[fname, self.dataset_mean, self.clip_range] for fname in self.img_filenames]
        print(
            "Computing std from all nifti files, using global mean computed above (clipped to {})".format(
                self.clip_range
            )
        )
        std_num = multiprocess_multiarg(get_std_numerator, args,num_processes=num_processes,multiprocess=multiprocess, debug=debug)
        std_num = torch.tensor(std_num)
        self.std = torch.sqrt(std_num.sum() / self.total_voxels)

    # def user_query_clip_range(self,intensity_percentile_range):
    #         try:
    #             self.clip_range = input("A Clip range has not been given. Press enter to accept clip range based on intensity-percentiles (i.e.{}) or give a new range now: ".format(intensity_percentile_range))
    #             if len(self.clip_range) == 0: self.clip_range = intensity_percentile_range
    #             else: self.clip_range = ast.literal_eval(self.clip_range) 
    #         except:
    #             print("A valid clip_range is not entered. Using intensity-default")
    #             self.clip_range = intensity_percentile_range

    def collate_lm_labels(self):
        labels_all=[]
        lmgps = "lm_group"
        keys = [k for k in self.global_properties.keys() if lmgps in k]
        for key in keys:
            shared_labels_gps= self.global_properties[key]['ds']
            labs_gp = []
            for gp in shared_labels_gps:
                for c in self.case_properties:
                    if gp in c['case_id']:
                        labels = c['labels']
                        labels = self.serializable_obj(labels)
                        labs_gp.extend(labels)

            labs_gp = list(set(labs_gp))
            labels_all.extend(labs_gp)
            print(labs_gp)

            self.global_properties[key].update({'labels':labs_gp ,'num_labels':len(labs_gp)})
        self.global_properties['labels_all'] = list(set(labels_all))
        labels_tot= len(labels_all)

        if len(keys)>1: self._remap_labels(keys,labels_tot)
        self.maybe_append_imported_labels()
        self.save_global_properties()

    def serializable_obj(self,ints_list):
        ints_out = [int(x) for x in ints_list]
        return ints_out

    def _remap_labels(self, keys,labels_tot):
        uns = unique_idx(labels_tot)
        self.global_properties['labels_all']=[]
        for key in keys:
            gp_labels = self.global_properties[key]['labels']
            labels_neo = []
            for lab in gp_labels:
                entry = next(uns)
                labels_neo.append(entry)
                self.global_properties['labels_all'].append(entry)
            self.global_properties[key]['labels_neo']=labels_neo


    def maybe_append_imported_labels(self):
        for key in self.lm_group_keys:
            dici = self.global_properties[key]
            largest_label = self.global_properties['labels_all'][-1]
            if 'imported_labelsets' in dici.keys():
                n_labels_to_add = len(dici['imported_labelsets'])
                labels = list(range(largest_label+1,largest_label+1+n_labels_to_add))
                self.global_properties['labels_all'].extend(labels)



# %%
if __name__ == "__main__":
    from fran.utils.common import *
    P = Project(project_title="totalseg");
    G = GlobalProperties(P,max_cases=200)
# %%
    G.store_projectwide_properties()
# %%
    debug=False
    G.collate_lm_labels()
    G.compute_std_mean_dataset(debug=debug)

    type(G.global_properties['lm_group1']['labels'][0])
# %%
    G.h5ns =[]
    for ds in P.global_properties['datasources']:
        h5fn = Path(ds['folder'])/("fg_voxels.h5")
        assert h5fn.exists(), "fg voxels not processed for datasource folder {}".format(ds['folder'])
        G.h5ns.append(h5fn)



# %%
    cid ='litqsmall_00008'
    cid ='litqsmall_00007'
    aa= h5f_file[cid]
# %%
    f =  unique_idx(5)

    f.__next__()
# %%
    


    
    dss = G.global_properties['datasources']
    
    ds =dss[-2]
    ds_name =ds['ds']
    h5fn = ds['h5_fname']

    cases_ds = [ cid for cid in G.cases_for_sampling if ds_name in cid]
# %%
    import h5py
    voxels = []
    with h5py.File(h5fn, "r") as h5f_file:
                for cid in cases_ds:
                    cs = h5f_file[cid]
                    voxels.append(cs[:])

# %%
   
