# %%
import itertools as il
import json
import random
from pathlib import Path

import ipdb
import numpy as np
import torch
from fastcore.all import store_attr
from fastcore.basics import GetAttr
from utilz.helpers import PrintableDict, pbar
from utilz.string import info_from_filename

from fran.managers import _DS

tr = ipdb.set_trace


from utilz.fileio import save_dict
from utilz.helpers import multiprocess_multiarg

from fran.preprocessing import (get_img_mask_filepairs, get_intensity_range,
                                get_means_voxelcounts, get_std_numerator,
                                import_h5py, percentile_range_to_str)


def unique_idx(total_len, start=1):
    """
    Generate unique indices starting from a given value.

    Args:
        total_len: Total number of indices to generate
        start: Starting index value (default: 1)

    Yields:
        int: Sequential unique indices
    """
    for x in range(start, total_len + 1):
        yield (x)


DS = _DS()


class GlobalProperties(GetAttr):
    _default = "project"
    """
    Input: raw_data_folder and project_title
    Outputs: (after running process_cases()) :
         2)Dataset global properties (including mean and std of voxels inside bboxes)
    """

    def __init__(
        self,
        project,
        bg_label=0,
        max_cases: int = None,
        percentile_range: list = [0.5, 99.5],
        clip_range=None,
    ):
        assert all(
            [isinstance(x, float) for x in percentile_range]
        ), "Provide float values for clip percentile_range. Corresponding HUs will be stored in file for future processing."
        store_attr("project,bg_label,percentile_range,clip_range,max_cases")
        self.case_ids, self.img_fnames = (
            self.collate_project_cases()
        )  #        self.h5ns =[]
        self.cases_for_sampling = self.sample_cases()
        self.img_fnames_for_sampling = self.filter_img_fnames_for_sampling()
        for ds in self.global_properties["datasources"]:
            h5fn = Path(ds["h5_fname"])
            assert (
                h5fn.exists()
            ), "fg voxels not processed for datasource folder {}".format(ds["folder"])
        self.case_properties=self._retrieve_h5_properties()


    def __str__(self) -> str:
        repr = str(self.global_properties)
        return repr

    def __repr__(self) -> str:


        return self.__str__()

    def collate_project_cases(self):
        """
        Collates project cases by retrieving image-mask file pairs.

        Returns:
        - tuple: Contains lists of case IDs and corresponding image file paths.
        """
        cases = get_img_mask_filepairs(parent_folder=self.project.raw_data_folder)
        case_ids, img_filenames = [], []
        for fpair in cases:
            case_id = info_from_filename(fpair[0].name, True)["case_id"]
            case_ids.append(case_id)
            img_filenames.append(fpair[0])
        return case_ids, img_filenames

    def sample_cases(self):
        """
        Sample a subset of cases for processing if max_cases is specified.

        Returns:
            list: List of case IDs to be used for sampling
        """
        if self.max_cases:
            random.seed(
                42
            )  # this ensures same set of cases is created each time this code is run for a specific datasset
            cases_for_sampling = random.sample(
                self.case_ids, np.minimum(len(self.case_ids), self.max_cases)
            )
        else:
            cases_for_sampling = self.case_ids
        return cases_for_sampling

    def filter_img_fnames_for_sampling(self):
        """
        Filter image filenames to include only those corresponding to sampled cases.

        Returns:
            list: List of image filenames for sampled cases
        """
        img_fnames_to_sample = []
        for img_fname in self.img_fnames:
            img_case_id = info_from_filename(img_fname.name, True)["case_id"]
            bools = img_case_id in self.cases_for_sampling
            img_fnames_to_sample.append(bools)
        img_fnames_to_sample = list(il.compress(self.img_fnames, img_fnames_to_sample))
        return img_fnames_to_sample

    def _retrieve_h5_voxels(self):
        """
        Retrieves voxel data from HDF5 files within sampled cases.

        Returns:
        - np.array: Concatenated array of voxel data from the cases.
        """
        voxels = []
        h5py = import_h5py()
        for dsa in self.global_properties["datasources"]:
            h5fn = dsa["h5_fname"]
            ds_name = dsa["ds"]
            ds_name_final = DS.resolve_ds_name(ds_name)
            cases_ds = [
                cid
                for cid in self.cases_for_sampling
                if cid.split("_")[0] == ds_name_final
            ]
            with h5py.File(h5fn, "r") as h5f_file:
                for cid in cases_ds:
                    cs = h5f_file[cid]
                    voxels.append(cs[:])
        voxels = np.concatenate(voxels)
        return voxels

    def _retrieve_h5_properties(self):
        """
        Retrieves properties for each case from HDF5 files.

        Returns:
        - List of dicts: Case properties including dataset name, spacing, labels, etc.
        """
        h5py = import_h5py()  #
        case_properties = []
        for dsa in self.global_properties["datasources"]:
            ds_props = []
            h5fn = dsa["h5_fname"]
            ds_name = dsa["ds"]
            ds_name_final = DS.resolve_ds_name(ds_name)
            cases_ds = [
                cid
                for cid in self.cases_for_sampling
                if cid.split("_")[0] == ds_name_final
            ]

            print(len(cases_ds))
            with h5py.File(h5fn, "r") as h5f_file:  # this file has all cases of the ds
                for cid in pbar(cases_ds):
                    cs = h5f_file[cid]
                    props = {
                        "ds": ds_name_final,
                        "case_id": cid,
                        "spacing": cs.attrs["spacing"],
                        "labels": cs.attrs["labels"],
                    }
                    ds_props.append(props)
            assert len(cases_ds) == len(
                ds_props
            ), "Mismatch in case_ids and case_properties. "
            case_properties.extend(ds_props)
        assert len(self.cases_for_sampling) == len(
            case_properties
        ), "Mismatch in case_ids and case_properties"
        return case_properties

    def store_projectwide_properties(self):
        """
        Stage 1.
        Start here.

        Computes and stores global properties across the entire project.

        This includes statistical measures such as mean and standard deviation
        of intensities, bounds of intensity range, and spatial median values.

        """
        cases = self._retrieve_h5_voxels()
        # tr()
        # print("Do i need to retrieve h5 properties again?")
        # self.case_properties = self._retrieve_h5_properties()

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

        Computes the clipped mean and std of the dataset.

        Parameters:
        - num_processes (int): Number of processes for multiprocessing.
        - multiprocess (bool): Flag to enable/disable multiprocessing.
        - debug (bool): Flag to enable/disable debug mode.
        """

        assert "mean_fg" in self.global_properties.keys(),"mean_fg not found in global_properties. Run store_projectwide_properties() first."
        percentile_label, intensity_percentile_range = get_intensity_range(
            self.global_properties
        )
        if not self.clip_range:
            self.clip_range = intensity_percentile_range
        self._compute_dataset_mean(num_processes, multiprocess, debug)
        self._compute_dataset_std(num_processes, multiprocess, debug)
        self.global_properties["intensity_clip_range"] = self.clip_range
        self.global_properties[percentile_label] = intensity_percentile_range
        self.global_properties["mean_dataset_clipped"] = self.dataset_mean.item()
        self.global_properties["std_dataset_clipped"] = self.std.item()
        self.global_properties["total_voxels"] = int(self.total_voxels)
        print(
            "Saving updated global_properties to file {}".format(
                self.global_properties_filename
            )
        )
        save_dict(self.global_properties, self.global_properties_filename, sort=True)

    # def collate_labels(self):

    def _compute_dataset_mean(self, num_processes, multiprocess, debug):
        """
        Computes the mean intensity of the dataset after applying clipping.

        Parameters are same as the outer method `compute_std_mean_dataset`.
        """
        args = [[fname, self.clip_range] for fname in self.img_fnames_for_sampling]
        print(
            "Computing means from all nifti files (clipped to {})".format(
                self.clip_range
            )
        )
        means_sizes = multiprocess_multiarg(
            get_means_voxelcounts,
            args,
            num_processes=num_processes,
            multiprocess=multiprocess,
            debug=debug,
        )
        means_sizes = torch.tensor(means_sizes)
        weighted_mn = torch.multiply(means_sizes[:, 0], means_sizes[:, 1])

        self.total_voxels = means_sizes[:, 1].sum()
        self.dataset_mean = weighted_mn.sum() / self.total_voxels

    def _compute_dataset_std(self, num_processes, multiprocess, debug):
        """
        Computes the standard deviation of the dataset after applying clipping.

        Parameters are same as the outer method `compute_std_mean_dataset`.
        """
        args = [
            [fname, self.dataset_mean, self.clip_range]
            for fname in self.img_fnames_for_sampling
        ]
        print(
            "Computing std from all nifti files, using global mean computed above (clipped to {})".format(
                self.clip_range
            )
        )
        std_num = multiprocess_multiarg(
            get_std_numerator,
            args,
            num_processes=num_processes,
            multiprocess=multiprocess,
            debug=debug,
        )
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
    @property
    def unique_labels(self):
        pass

    def collate_lm_labels(self):
        """
        Collate and organize landmark labels from all datasets in the project.

        This method processes label groups, resolves dataset names, and creates
        a unified label mapping across all datasets.
        """
        labels_all = []
        # CODE: find relevance of lm_groups in modern version. Phase it out if redundant
        lmgps = "lm_group"
        keys = [k for k in self.global_properties.keys() if lmgps in k]
        for key in keys:
            shared_labels_gps = self.global_properties[key]["ds"]
            labs_gp = []
            for gp in shared_labels_gps:
                ds_name = DS.resolve_ds_name(gp)
                for c in self.case_properties:
                    if ds_name == c["ds"]:
                        labels = c["labels"]
                        labels = self.serializable_obj(labels)
                        labs_gp.extend(labels)

            labs_gp = list(set(labs_gp))
            dici = {"ds": ds_name, "label": labs_gp}
            labels_all.extend(labs_gp)
            print(labs_gp)
            self.global_properties[key].update(
                {"labels": labs_gp, "num_labels": len(labs_gp)}
            )
        self.global_properties["labels_all"] = list(set(labels_all))
        labels_tot = len(labels_all)
        if len(keys) > 1:
            self._remap_labels(keys, labels_tot)
        self.maybe_append_imported_labels()
        self.save_global_properties()

    def serializable_obj(self, ints_list):
        """
        Converts a list of labels into a fully serializable object.

        Parameters:
        - ints_list: List of label integers to convert.

        Returns:
        - List of int: List of converted integer labels.
        """
        ints_out = [int(x) for x in ints_list]
        return ints_out

    def _remap_labels(self, keys, labels_tot):
        """
        Remap labels to ensure unique indices across all label groups.

        Args:
            keys: List of label group keys to process
            labels_tot: Total number of labels to remap
        """
        uns = unique_idx(labels_tot)
        self.global_properties["labels_all"] = []
        for key in keys:
            gp_labels = self.global_properties[key]["labels"]
            labels_neo = []
            for lab in gp_labels:
                entry = next(uns)
                labels_neo.append(entry)
                self.global_properties["labels_all"].append(entry)
            self.global_properties[key]["labels_neo"] = labels_neo

    def maybe_append_imported_labels(self):
        """
        Append imported label sets to the global properties if they exist.

        This method extends the labels_all list with additional labels from
        imported labelsets found in label group configurations.
        """
        for key in self.lm_group_keys:
            dici = self.global_properties[key]
            largest_label = self.global_properties["labels_all"][-1]
            if "imported_labelsets" in dici.keys():
                n_labels_to_add = len(dici["imported_labelsets"])
                labels = list(
                    range(largest_label + 1, largest_label + 1 + n_labels_to_add)
                )
                self.global_properties["labels_all"].extend(labels)


# %%
if __name__ == "__main__":
# %%
#SECTION:--------------------------------------------- SETUPPPPPP--------------------------------------------------------------------------------------

    from fran.managers import Project
    from fran.utils.common import *

    P = Project(project_title="nodes")
    G = GlobalProperties(P, max_cases=50)
    conf = ConfigMaker(P, raytune=False, configuration_filename=None).config
    plan = conf["plan"]
# %%
    labs_gp =[]
    for cc in G.case_properties:
                    # if ds_name == cc["ds"]:
                        labels = cc["labels"]
                        labels = G.serializable_obj(labels)
                        labs_gp.extend(labels)

# %%


    labels_all = []
    lmgps = "lm_group"
    keys = [k for k in G.global_properties.keys() if lmgps in k]
    for key in keys:
        shared_labels_gps = G.global_properties[key]["ds"]
# %%
        labs_gp = []
        for cc in G.case_properties:
                    labels = cc["labels"]
                    ds = cc['ds']
                    labels = G.serializable_obj(labels)
                    labels = tuple(labels)
                    # if labels!=[1]:
                    #     tr()
                    dici = {"ds": ds, "label": labels}
                    labs_gp.append(dici)
# %%
    df = pd.DataFrame(labs_gp)
    df = df.drop_duplicates(subset=["ds", "label"], keep="first").reset_index(drop=True)

# %%
#         labs_gp = list(set(labs_gp))
#         dici = {"ds": ds_name, "label": labs_gp}
#         labels_all.extend(labs_gp)
#         print(labs_gp)
#         G.global_properties[key].update(
#             {"labels": labs_gp, "num_labels": len(labs_gp)}
#         )
#     G.global_properties["labels_all"] = list(set(labels_all))
#     labels_tot = len(labels_all)
#     if len(keys) > 1:
# # %%
#     if not "labels_all" in P.global_properties.keys():
#         P.set_lm_groups(plans["lm_groups"])
#         P.maybe_store_projectwide_properties(overwrite=True)
#     G.store_projectwide_properties()
# %%
    debug = True
    G.collate_lm_labels()
    G.compute_std_mean_dataset(debug=debug)

    type(G.global_properties["lm_group1"]["labels"][0])
# %%
    G.h5ns = []
    for ds in P.global_properties["datasources"]:
        h5fn = Path(ds["folder"]) / ("fg_voxels.h5")
        assert h5fn.exists(), "fg voxels not processed for datasource folder {}".format(
            ds["folder"]
        )
        G.h5ns.append(h5fn)

# %%
    cid = "litqsmall_00008"
    cid = "litqsmall_00007"
    aa = h5f_file[cid]
# %%
    f = unique_idx(5)

    f.__next__()
# %%

    dss = G.global_properties["datasources"]

    ds = dss[-2]
    ds_name = ds["ds"]
    h5fn = ds["h5_fname"]

    cases_ds = [cid for cid in G.cases_for_sampling if ds_name in cid]

    import h5py

    voxels = []
    with h5py.File(h5fn, "r") as h5f_file:
        for cid in cases_ds:
            cs = h5f_file[cid]
            voxels.append(cs[:])

# %%
# %%
    G = P.G

    cases_for_sampling = random.sample(
        G.case_ids, np.minimum(len(G.case_ids), G.max_cases)
    )
    img_fname = G.img_filenames[0]
    img_fnames_to_sample = []
# %%
# %%

    G.case_properties = G._retrieve_h5_properties()
    h5py = import_h5py()
    case_properties = []
    for dsa in G.global_properties["datasources"]:
        ds_props = []
        h5fn = dsa["h5_fname"]
        ds_name = dsa["ds"]
        ds = getattr(DS, ds_name)
        if ds["alias"] is not None:
            ds_name_final = ds["alias"]
        else:
            ds_name_final = ds_name
        cases_ds = [
            cid for cid in G.cases_for_sampling if cid.split("_")[0] == ds_name_final
        ]

        print(len(cases_ds))
        with h5py.File(h5fn, "r") as h5f_file:
            for cid in pbar(cases_ds):
                cs = h5f_file[cid]
                props = {
                    "case_id": cid,
                    "spacing": cs.attrs["spacing"],
                    "labels": cs.attrs["labels"],
                }
                ds_props.append(props)
        assert len(cases_ds) == len(
            ds_props
        ), "Mismatch in case_ids and case_properties"
        case_properties.extend(ds_props)
    assert len(G.case_ids) == len(
        case_properties
    ), "Mismatch in case_ids and case_properties"

# %%
    gp = "tcianodesshort"
# %%
    labels_all = []
    lmgps = "lm_group"
    keys = [k for k in G.global_properties.keys() if lmgps in k]
    key = keys[-1]
    gp = shared_labels_gps[-1]
# %%
# %%
    labels_all = []
    lmgps = "lm_group"
    keys = [k for k in G.global_properties.keys() if lmgps in k]
    for key in keys:
        shared_labels_gps = G.global_properties[key]["ds"]
        labs_gp = []
        for gp in shared_labels_gps:
            tr()
            ds_name = DS.resolve_ds_name(gp)
            for cc in G.case_properties:
                if ds_name == cc["ds"]:
                    labels = cc["labels"]
                    labels = G.serializable_obj(labels)
                    labs_gp.extend(labels)

        labs_gp = list(set(labs_gp))
        dici = {"ds": ds_name, "label": labs_gp}
        labels_all.extend(labs_gp)
        print(labs_gp)
        G.global_properties[key].update({"labels": labs_gp, "num_labels": len(labs_gp)})
    G.global_properties["labels_all"] = list(set(labels_all))
    labels_tot = len(labels_all)

# %%
