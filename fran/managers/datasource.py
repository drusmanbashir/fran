# %%
import sqlite3
import ipdb
from utilz.string import info_from_filename

tr = ipdb.set_trace

import os
import sys
from pathlib import Path

from utilz.helpers import *

sys.path += ["/home/ub/Dropbox/code"]
from utilz.fileio import *

if "XNAT_CONFIG_PATH" in os.environ:
    from xnat.object_oriented import *
common_vars_filename = os.environ["FRAN_COMMON_PATHS"]
COMMON_PATHS = load_yaml(common_vars_filename)
from contextlib import contextmanager

import ipdb
import SimpleITK as sitk
from fastcore.basics import GetAttr, Union
from label_analysis.helpers import to_binary, to_int
from utilz.fileio import load_dict, save_dict
from utilz.helpers import find_matching_fn, multiprocess_multiarg
from utilz.string import info_from_filename

from fran.preprocessing.datasetanalyzers import (case_analyzer_wrapper,
                                                 import_h5py)


MNEMONICS = ["liver", "lits", "lungs", "nodes", "bones", "lilu", "totalseg"]
class _DS():
    '''
    each folder has subfolder images and lms
    if a member has a 2-tuple value intead of 1, the send element is alias of the member dataset. This alias is used to match filenames with correct dataset
    '''
    def __init__(self) -> None:
        # for any datasource, an alias matching the {projectname}_ part of the filename should be given if the ds name is non-standard, e.g., drli_short
            self.lits={'ds': 'lits', 'folder': "/s/datasets_bkp/lits_segs_improved/", 'alias':None}
            self.lits_tmp={'ds': 'lits', 'folder': "/s/datasets_bkp/litstmp/", 'alias':"tmp"}
            self.litq={'ds':'litq', 'folder': "/s/xnat_shadow/litq", 'alias':None}
            self.nodes = {'ds': 'nodes', "folder":"/s/xnat_shadow/nodes", "alias":None}
            self.nodesthick={'ds':'nodesthick',"folder":"/s/xnat_shadow/nodesthick/", "alias": None}
            self.tcianode={'ds':'tcianode',"folder":"/s/xnat_shadow/tcianode",  'alias':None}
            self.tcianodeshort={'ds':'tcianodeshort',"folder":"/s/xnat_shadow/tcianodeshort",  'alias':"tcianode"}
            self.drli_short={'ds':'drli_short','folder': "/s/datasets_bkp/drli_short/","alias":'drli'}
            self.drli={'ds':'drli','folder': "/s/datasets_bkp/drli/","alias":None}
            self.litqsmall={'ds':'litqsmall','folder': "/s/datasets_bkp/litqsmall/","alias":None}
            self.lidc2={'ds':'lidc2','folder': "/s/xnat_shadow/lidc2","alias":None}
            self.lidctmp={'ds':'lidctmp','folder': "/s/xnat_shadow/lidctmp","alias":None}
            self.totalseg={'ds':'totalseg',"folder":"/s//xnat_shadow/totalseg","alias":None}
            self.task6={'ds':'task6','folder': "/s/datasets_bkp/Task06Lung/","alias":'lung'}

    def resolve_ds_name(self,ds:str):

        try:
            ds_dict = getattr(self,ds)
        except:
            return None # ds does not exist as a standard DS
        alias = ds_dict['alias']
        if alias is not None:
            return alias
        else:
            return ds

    def get_folder(self, ds:str):
        ds_dict = getattr(self,ds)
        return ds_dict['folder']
    def __str__(self):
        datasrcs = self.__dict__.keys()
        datasrcs = ",".join([k for k in datasrcs])
        return "Datasources: "+datasrcs

    def __repr__(self):
        datasrcs = self.__dict__.keys()
        datasrcs = ",".join([k for k in datasrcs])
        return "Datasources: "+datasrcs

class Datasource(GetAttr):
    """

    This class manages a dataset folder containing 'images' and 'lms' (label maps) subfolders,
    handles data preprocessing, integrity checking, and HDF5 storage of processed voxel data.
    
    Attributes
    ----------
    folder : Path
        Root folder containing 'images' and 'lms' subfolders
    name : str
        Dataset name (inferred from filenames or provided)
    alias : str, optional
        Alternative name for dataset matching
    bg_label : int
        Background label value (default: 0)
    test : bool
        Whether this is a test dataset
    h5_fname : Path
        Path to HDF5 file storing processed voxel data
    verified_pairs : list
        List of verified [image, label] file pairs
    new_cases : list
        List of new cases to be processed
        
    Main Methods
    -----------
    integrity_check()
        Verifies matching image-label pairs and equal counts
    process(return_voxels=True, num_processes=8, multiprocess=True, debug=False)
        Processes all new cases and extracts foreground voxel statistics
    dump_to_h5()
        Saves processed voxel data and metadata to HDF5 file
    relabel(remapping=None, target_label=None)
        Relabels all label maps according to specified mapping
    infer_dataset_name()
        Extracts dataset name from first image filename
        
    Case attributes:
    Each caser has following:
                    ds.attrs['spacing'] = list(output['case']['properties']['spacing'])
                    ds.attrs['labels'] = list(output['case']['properties']['labels'])
                    ds.attrs['numel_fg']= output['case']['properties']['numel_fg']
                    ds.attrs['mean_fg']= output['case']['properties']['mean_fg']
                    ds.attrs['min_fg']= output['case']['properties']['min_fg']
                    ds.attrs['max_fg']= output['case']['properties']['max_fg']
                    ds.attrs['std_fg']= output['case']['properties']['std_fg']


    Example
    -------
    >>> ds = Datasource(folder="/path/to/dataset", name="liver_ct")
    >>> ds.process(num_processes=4)  # Process all cases
    >>> print(f"Dataset has {len(ds)} cases")
    
    Notes
    -----
    - Folder structure must be: folder/images/*.nii.gz, folder/lms/*.nii.gz
    - Image and label filenames must match exactly
    - Creates 'fg_voxels.h5' file in dataset folder for processed data
    - Supports incremental processing (skips already processed cases)
    """
    
    def __init__(self, folder: Union[str, Path],name:str=None,alias=None, bg_label=0, test=False) -> None:
        """
        Initialize a Datasource for medical imaging data.
        
        Parameters
        ----------
        folder : Union[str, Path]
            Root folder containing 'images' and 'lms' subfolders with paired files
        name : str, optional
            Dataset name. If None, inferred from first image filename
        alias : str, optional
            Alternative dataset name for filename matching
        bg_label : int, default=0
            Background label value for processing
        test : bool, default=False
            Whether this dataset is for testing purposes
        """
        self.bg_label = bg_label
        self.folder = Path(folder)
        self.test=test
        self.alias= alias
        self.h5_fname = self.folder / "fg_voxels.h5"
        if name is None:
            self.name = self.infer_dataset_name()
        else:
            self.name = name

        self.integrity_check()
        self._filter_unprocessed_cases()

    def infer_dataset_name(self):
        subfolder = self.folder / ("images")
        fn = list(subfolder.glob("*"))[0]
        proj_title = info_from_filename(fn.name)["proj_title"]
        return proj_title

    def relabel(self, remapping: dict = None , target_label: int=None):
        assert remapping or target_label,"Must specify either a remapping_dict or specify a target_label so all labels are converted to it"
        '''
        scheme: if scheme is an int, all labels are converted to it. If dict, then explicit remapping is used based on dict.
        '''
        fldr_lms = self.folder/("lms")
        lm_fns = list(fldr_lms.glob("*"))
        for fn in lm_fns:
            print("Processing file: ",fn)
            lm = sitk.ReadImage(fn)
            if target_label:
                lm = to_binary(lm)
                if target_label!=1:
                    lm = relabel(lm,{1:target_label})
            else:
                lm = relabel(lm,remapping=remapping)
            lm = to_int(lm)
            sitk.WriteImage(lm,fn)

  
    def integrity_check(self):
        """
        verify name pairs
        any other verifications
        """

        images = list((self.folder / ("images")).glob("*"))
        lms = list((self.folder / ("lms")).glob("*"))
        assert (
            (a := len(images)) == (b := len(lms))
        ), "Different lengths of images {0}, and lms {1}.\nCheck your data folder".format(
            a, b
        )
        self.verified_pairs = []
        for img_fn in images:
            self.verified_pairs.append([img_fn, find_matching_fn(img_fn, lms,["all"])])
        print("Verified filepairs are matched")


    def _filter_unprocessed_cases(self):
        '''
        Loads h5_fname to get list of already completed cases.
        Any new cases will be processed and added to project.h5_fname
        '''
        
        h5py = import_h5py()
        try:
            with h5py.File(self.h5_fname, 'r') as h5f:
                prev_processed_cases = list(h5f.keys())
            # prev_processed_cases = set([b['case_id'] for b in self.raw_dataset_properties])
        except FileNotFoundError:
            print("First time preprocessing dataset. Will create new file: {}".format(self.h5_fname))
            self.raw_dataset_properties = []
            prev_processed_cases = set()
        all_case_ids = []
        for fns in self.verified_pairs:
            inf = info_from_filename(fns[0].name,full_caseid=True)
            case_id = inf['case_id']
            all_case_ids.append(case_id)
        assert (l1:=len(all_case_ids))==(l2:=len(set(all_case_ids))), "Duplicate case_ids found. Run fix_repeat_caseids() on parent folder"
        new_case_ids = set(all_case_ids).difference(prev_processed_cases)
        print("Found {0} new cases\nCases already processed in a previous session: {1}".format(len(new_case_ids), len(prev_processed_cases)))
        assert (l:=len(new_case_ids)) == (l2:=(len(all_case_ids)-len(prev_processed_cases))), "Difference in number of new cases"
        if len(new_case_ids) == 0: 
            print("No new cases found.")
            self.new_cases = []
        else:
            self.new_cases = [file_tuple for file_tuple in self.verified_pairs if info_from_filename(file_tuple[0].name,full_caseid=True)['case_id'] in new_case_ids] #file_tuple[0]


    def process(
        self, return_voxels=True, num_processes=8, multiprocess=True, debug=False, 
    ):
        """
        Stage 1: derives datase properties especially intensity_fg
        if return_voxels == True, returns voxels to be stored inside the h5 file
        """

        args_list = [
                [case_tuple,  self.bg_label, return_voxels]
                for case_tuple in self.new_cases
            ]
        self.outputs = multiprocess_multiarg(
            func=case_analyzer_wrapper,
            arguments=args_list,
            num_processes=num_processes,
            multiprocess=multiprocess,
            debug=debug,
        )

        for output in self.outputs:
            self.raw_dataset_properties.append(output["case"])
        self.dump_to_h5()


    def dump_to_h5(self):
        h5py = import_h5py()
        if self.h5_fname.exists():
            mode= 'a'
        else: mode = 'w'
        with h5py.File(self.h5_fname, mode) as h5f: 
            for output in self.outputs:
                try:
                    ds= h5f.create_dataset(output["case"]["case_id"], data=output["voxels"])
                    ds.attrs['spacing'] = list(output['case']['properties']['spacing'])
                    ds.attrs['labels'] = list(output['case']['properties']['labels'])
                    ds.attrs['numel_fg']= output['case']['properties']['numel_fg']
                    ds.attrs['mean_fg']= output['case']['properties']['mean_fg']
                    ds.attrs['min_fg']= output['case']['properties']['min_fg']
                    ds.attrs['max_fg']= output['case']['properties']['max_fg']
                    ds.attrs['std_fg']= output['case']['properties']['std_fg']

                except ValueError as e:
                    print(e)
                    print("Case id {} already exists in h5 file. Skipping".format(output['case']['case_id']))

    def _store_raw_dataset_properties(self):
        processed_props = [output['case'] for output in self.outputs]
        if self.h5_fname.exists():
            existing_props = load_dict(self.h5_fname)
            existing_total = len(existing_props)
            assert existing_total + len(processed_props) == len(self.project), "There is an existing raw_dataset_properties file. New cases are processed also, but their sum does not match the size of this project"
            raw_dataset_props = existing_props + processed_props
        else:
            raw_dataset_props = processed_props
        save_dict(raw_dataset_props, self.h5_fname)


    def extract_img_lm_fnames(self, ds):
        img_fnames = list((ds["source_path"] / ("images")).glob("*"))
        lm_fnames = list((ds["source_path"] / ("lms")).glob("*"))
        img_symlinks, lm_symlinks = [], []

        verified_pairs = []
        for img_fn in img_fnames:
            verified_pairs.append([img_fn, find_matching_fn(img_fn, lm_fnames,tags=["all"])])
        assert self.paths_exist(
            verified_pairs
        ), "(Some) paths do not exist. Fix paths and try again."
        print("self.populating raw data folder (with symlinks)")
        for pair in verified_pairs:
            img_symlink, lm_symlink = self.filepair_symlink(pair)
            img_symlinks.append(img_symlink)
            lm_symlinks.append(lm_symlink)
        return img_symlinks, lm_symlinks

    @property
    def images(self):
        images = [x[0] for x in self.verified_pairs]
        return images

    
    @property
    def labels(self):
        if not hasattr(self,"_labels"):
           import h5py
           labels = []
           with h5py.File(self.h5_fname, "r") as f:
                for case_id , obj in f.items():
                    labs = obj.attrs['labels']
                    labels.extend(tuple(labs))
           labels = set(labels)
           labels_list =[]
           for lab in labels:
                lab = int(lab)
                labels_list.append(lab)
           # labels = list(labels)
           self._labels = labels_list
             
        return self._labels


    @property
    def lms(self):
        lms = [x[1] for x in self.verified_pairs]
        return lms

    def __len__(self):
        return len(self.verified_pairs)

    def __repr__(self):
        s = "Dataset: {0}".format(self.name)
        return s

    def create_symlinks(self):
        pass


DS= _DS()


def val_indices(a, n):
    """
    Divide `a` elements into `n` roughly equal slices.

    Parameters
    ----------
    a : int
        The total number of items to divide.
    n : int
        The number of slices to divide `a` into.

    Returns
    -------
    list of slice
        A list of slices, each specifying a segment of `a`.
    """
    a = a - 1
    k, m = divmod(a, n)
    return [slice(i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(n)]


@contextmanager
def db_ops(db_name):
    """
    Context manager to open and manage a SQLite database connection.

    Parameters
    ----------
    db_name : str
        Path to the SQLite database file.

    Yields
    ------
    sqlite3.Cursor
        A cursor for executing SQL commands within the database.

    Example
    -------
    >>> with db_ops("database.db") as cursor:
    ...     cursor.execute("SELECT * FROM table")
    """
    conn = sqlite3.connect(db_name)
    try:
        cur = conn.cursor()
        yield cur
    except Exception as e:
        # do something with exception
        conn.rollback()
        raise e
    else:
        conn.commit()
    finally:
        conn.close()


if __name__ == '__main__':
# %%
   nodes_fldr = "/s/xnat_shadow/nodes"
   nodes_fn = "/s/xnat_shadow/nodes/fg_voxels.h5"
   ds = Datasource(nodes_fldr,"nodes")
   ds.process()
# %%
   import h5py
   ff = h5py.File(nodes_fn, "r")
   labels = []
   with h5py.File(nodes_fn, "r") as f:
        for case_id , obj in f.items():
            labs = obj.attrs['labels']
            labels.extend(tuple(labs))
   labels = set(labels)
#
# # %%
#     labs_list = []
#     for lab in labels:
#         labs_list.append(lab)
#     print(labs_list)
#
#    
#
# # %%
#    
#    labels = []
#    for fff in ff:
#        cc= ff['fff'] 
#        labels.append(fff.attrs['labels'])
#
#
# # %%
#     ds.raw_dataset_properties=[]
#     for output in ds.outputs:
#         ds.raw_dataset_properties.append(output["case"])
#         ds.dump_to_h5()
# #
#
#
# # %%3k3k3k3
