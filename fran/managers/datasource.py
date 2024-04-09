
# %%
import sqlite3
import ipdb
from fastcore.basics import GetAttr, Union
from label_analysis.totalseg import TotalSegmenterLabels
from monai.utils.enums import StrEnum
from fran.preprocessing.datasetanalyzers import case_analyzer_wrapper, import_h5py
from fran.utils.fileio import load_dict, save_dict
from fran.utils.helpers import find_matching_fn, multiprocess_multiarg
from fran.utils.string import (
    info_from_filename,
)

tr = ipdb.set_trace

from pathlib import Path
class DS(StrEnum):
    '''
    each folder has subfolder images and lms
    '''
    
    lits="/s/datasets_bkp/lits_segs_improved/"
    litq="/s/xnat_shadow/litq"
    drli_short="/s/datasets_bkp/drli_short/"
    drli="/s/datasets_bkp/drli/"
    litqsmall="/s/datasets_bkp/litqsmall/"
    lidc2="/s/xnat_shadow/lidc2"
    lidctmp="/s/xnat_shadow/lidctmp"
    totalseg = "/s//xnat_shadow/totalseg"
    task6="/s/datasets_bkp/Task06Lung/"

class Datasource(GetAttr):
    def __init__(self, folder: Union[str, Path],name:str=None,bg_label=0, test=False) -> None:
        """
        src_folder: has subfolders 'images' and 'lms'. Files in each are identically named
        """
        self.bg_label = bg_label
        self.folder = Path(folder)
        self.test=test
        self.h5_fname = self.folder / "fg_voxels.h5"
        if name is None:
            self.name = self.infer_dataset_name()
        else:
            self.name = name

        self.integrity_check()
        self.filter_unprocessed_cases()

    def infer_dataset_name(self):
        subfolder = self.folder / ("images")
        fn = list(subfolder.glob("*"))[0]
        proj_title = info_from_filename(fn.name)["proj_title"]
        return proj_title

    def integrity_check(self):
        """
        verify name pairs
        any other verifications
        """

        images = list((self.folder / ("images")).glob("*"))
        lms = list((self.folder / ("lms")).glob("*"))
        assert (
            a := len(images) == (b := len(lms))
        ), "Different lengths of images {0}, and lms {1}.\nCheck your data folder".format(
            a, b
        )
        self.verified_pairs = []
        for img_fn in images:
            self.verified_pairs.append([img_fn, find_matching_fn(img_fn, lms)])
        print("Verified filepairs are matched")


    def filter_unprocessed_cases(self):
        '''
        Loads project.raw_dataset_properties_filename to get list of already completed cases.
        Any new cases will be processed and added to project.raw_dataset_properties_filename
        '''
        
        h5py = import_h5py()
        try:
            with h5py.File(self.h5_fname, 'r') as h5f:
                prev_processed_cases = list(h5f.keys())
            # prev_processed_cases = set([b['case_id'] for b in self.raw_dataset_properties])
        except FileNotFoundError:
            print("First time preprocessing dataset. Will create new file: {}".format(self.raw_dataset_properties_filename))
            self.raw_dataset_properties = []
            prev_processed_cases = set()
        all_case_ids = []
        for fns in self.verified_pairs:
            inf = info_from_filename(fns[0].name,full_caseid=True)
            case_id = inf['case_id']
            all_case_ids.append(case_id)
        assert(len(all_case_ids)==len(set(all_case_ids))),"Some case_ids are repeated. Not implemented yet!"

        new_case_ids = set(all_case_ids).difference(prev_processed_cases)
        print("Found {0} new cases\nCases already processed in a previous session: {1}".format(len(new_case_ids), len(prev_processed_cases)))
        assert (l:=len(new_case_ids)) == (l2:=(len(all_case_ids)-len(prev_processed_cases))), "Difference in number of new cases"
        if len(new_case_ids) == 0: 
            print("No new cases found.")
            self.new_cases = []
        else:
            self.new_cases = [file_tuple for file_tuple in self.verified_pairs if info_from_filename(file_tuple[0].name,full_caseid=True)['case_id'] in new_case_ids] #file_tuple[0]


    def process_new_cases(
        self, return_voxels=True, num_processes=8, multiprocess=True, debug=False
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

                except ValueError:
                    print("Case id {} already exists in h5 file. Skipping".format(output['case']['case_id']))

    def store_raw_dataset_properties(self):
        processed_props = [output['case'] for output in self.outputs]
        if self.raw_dataset_properties_filename.exists():
            existing_props = load_dict(self.raw_dataset_properties_filename)
            existing_total = len(existing_props)
            assert existing_total + len(processed_props) == len(self.project), "There is an existing raw_dataset_properties file. New cases are processed also, but their sum does not match the size of this project"
            raw_dataset_props = existing_props + processed_props
        else:
            raw_dataset_props = processed_props
        save_dict(raw_dataset_props, self.raw_dataset_properties_filename)


    def extract_img_lm_fnames(self, ds):
        img_fnames = list((ds["source_path"] / ("images")).glob("*"))
        lm_fnames = list((ds["source_path"] / ("lms")).glob("*"))
        img_symlinks, lm_symlinks = [], []

        verified_pairs = []
        for img_fn in img_fnames:
            verified_pairs.append([img_fn, find_matching_fn(img_fn, lm_fnames)])
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



def get_ds_remapping(ds:str,global_properties):
        key = 'lm_group'
        keys=[]
        for k in global_properties.keys():
            if key in k:
                keys.append(k)

        for k in keys:
            dses  = global_properties[k]['ds']
            if ds in dses:
                labs_src = global_properties[k]['labels']
                if hasattr (global_properties[k],'labels'):
                    labs_dest = global_properties[k]['labels_neo']
                else:
                    labs_dest = labs_src
                remapping = {src:dest for src,dest in zip(labs_src,labs_dest)}
                return remapping
        raise Exception("No lm group for dataset {}".format(ds))

# %%
if __name__ == "__main__":
    debug=False
    D2 = Datasource(DS.drli)
    # D2.process_new_cases(debug=debug)


    import h5py
    f = h5py.File(D2.h5_fname,"r")
    case_ids = list(f.keys())
    cid = case_ids[0]
    f[cid].attrs.keys()



# %%
    TSL = TotalSegmenterLabels()
    lr= TSL.labels("lung","right")
    ll = TSL.labels("lung","left")

    remapping  = {l:0 for l in TSL.all}
    for l in lr:
        remapping[l]= 8

    for l in ll:
        remapping[l]= 9


# %%
# %%
