# %%
import shutil
import string

import ipdb
import SimpleITK as sitk
from fastcore.basics import GetAttr, Union
from label_analysis.helpers import (get_labels, single_label, to_binary,
                                    to_int, to_label)
from monai.utils.enums import StrEnum

from fran.preprocessing.datasetanalyzers import (case_analyzer_wrapper,
                                                 import_h5py)
from fran.utils.fileio import load_dict, save_dict
from fran.utils.helpers import find_matching_fn, multiprocess_multiarg
from fran.utils.string import info_from_filename, str_to_path

tr = ipdb.set_trace


def subscript_generator():
        letters = list(string.ascii_letters)
        while(letters):
            yield letters.pop(0)

from pathlib import Path


@str_to_path()
def fix_repeat_caseids(parent_folder):
        fldr_imgs = parent_folder/("images")
        fldr_lms = parent_folder/("lms")
        img_fns = list(fldr_imgs.glob("*"))
        lm_fns = list(fldr_lms.glob("*"))
        file_pairs = []
        
        for img_fn in img_fns:
            lm_fn = find_matching_fn(img_fn,lm_fns)
            file_pairs.append([img_fn,lm_fn])

        case_ids =[]
        files_to_alter = []

        for filepair in file_pairs:
            cid = info_from_filename(filepair[0].name,full_caseid=True)['case_id']
            subs = subscript_generator()
            while cid in case_ids:
                cid = info_from_filename(filepair[0].name,full_caseid=True)['case_id']+next(subs)
                # cid_new =next(subs)
            if cid!= info_from_filename(filepair[0].name,full_caseid=True)['case_id']:
                dici = {'filepair':filepair, 'new_cid':cid}
                files_to_alter.append(dici)
            case_ids.append(cid)
        print("Files with repeat case_id which will now be given unique case_id: ", len(files_to_alter))
        for f in files_to_alter:
            print("Fixing ",f['filepair'][0])
            change_caseid(f['filepair'],f['new_cid'])





class _DS():
    '''
    each folder has subfolder images and lms
    if a member has a 2-tuple value intead of 1, the send element is alias of the member dataset. This alias is used to match filenames with correct dataset
    '''
    def __init__(self) -> None:
            self.lits={'ds': 'lits', 'folder': "/s/datasets_bkp/lits_segs_improved/", 'alias':None}
            self.litq={'ds':'litq', 'folder': "/s/xnat_shadow/litq", 'alias':None}
            self.nodes = {'ds': 'nodes', "folder":"/s/xnat_shadow/nodes", "alias":None}
            self.nodesthick={'ds':'nodesthick',"folder":"/s/xnat_shadow/nodesthick/", "alias": None}
            self.tcianode={'ds':'tcianode',"folder":"/s/xnat_shadow/tcianode",  'alias':None}
            self.tcianodeshort={'ds':'tcianodeshort',"folder":"/s/xnat_shadow/tcianodeshort",  'alias':"tcianode"}
            self.drli_short={'ds':'drli_short','folder': "/s/datasets_bkp/drli_short/","alias":None}
            self.drli={'ds':'drli','folder': "/s/datasets_bkp/drli/","alias":None}
            self.litqsmall={'ds':'litqsmall','folder': "/s/datasets_bkp/litqsmall/","alias":None}
            self.lidc2={'ds':'lidc2','folder': "/s/xnat_shadow/lidc2","alias":None}
            self.lidctmp={'ds':'lidctmp','folder': "/s/xnat_shadow/lidctmp","alias":None}
            self.totalseg={'ds':'totalseg',"folder":"/s//xnat_shadow/totalseg","alias":None}
            self.task6={'ds':'task6','folder': "/s/datasets_bkp/Task06Lung/","alias":None}

    def resolve_ds_name(self,ds:str):
        ds_dict = getattr(self,ds)
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
    def __init__(self, folder: Union[str, Path],name:str=None,alias=None, bg_label=0, test=False) -> None:
        """
        src_folder: has subfolders 'images' and 'lms'. Files in each are identically named
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
            self.verified_pairs.append([img_fn, find_matching_fn(img_fn, lms)])
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
    from label_analysis.totalseg import TotalSegmenterLabels, relabel
    debug=False
    # D2 = Datasource(DS.drli)
    DS= _DS()
    D2 = Datasource(DS.get_folder("tcianodeshort"))
    D2.relabel(target_label=1)
    D2.process(debug=debug)
    # D2 = Datasource(DS.tcianode_short)
    # D2 = Datasource(DS.nodes)
    # fn = "/s/xnat_shadow/tcianodeshort/lms/tcianode_abd006.nii.gz"
    # lm = sitk.ReadImage(fn)
    # get_labels(lm)

# %%

    fix_repeat_caseids("/s/xnat_shadow/nodes")
# %%


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

    def change_caseid(filepair, new_cid):
        for fn in filepair:
            cid_old = info_from_filename(fn.name,full_caseid=True)['case_id']
            fn_out = fn.str_replace(cid_old,new_cid)
            shutil.move(fn,fn_out)


# %%
        fn ="/s/xnat_shadow/tcianodeshort/lms/tcianode_abd004.nii.gz"
        target_label=1
        fldr_lms = D2.folder/("lms")



# %%
# %%
