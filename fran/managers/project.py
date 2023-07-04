# %%
import math
import ipdb

from fran.utils.string import cleanup_fname, drop_digit_suffix, info_from_filename, strip_extension
tr = ipdb.set_trace

from pathlib import Path
import os, sys
import itertools as il
import functools as fl
from fastai.vision.augment import store_attr
from fran.utils.dictopts import dic_in_list
from fran.utils.helpers import *
import shutil
from fran.utils.helpers import DictToAttr, ask_proceed

sys.path += ["/home/ub/Dropbox/code"]
from types import SimpleNamespace
from fran.utils.fileio import *

if "XNAT_CONFIG_PATH" in os.environ:
    from xnat.object_oriented import *
common_vars_filename = os.environ["FRAN_COMMON_PATHS"]
from fran.utils.templates import mask_labels_template
from fran.utils.helpers import pat_full,pat_nodesc, pat_idonly


class Project(DictToAttr):
    def __init__(self, project_title):
        store_attr()
        self.set_folder_file_names()
        if  self.raw_dataset_info_filename.exists():
            self.raw_data_sources = load_dict(self.raw_dataset_info_filename)
        else:
            self.raw_data_sources=[]

    def create_project(self, datasets: list = None,label_dict_filename=None, test: bool = None):
        """
        param datasets: list of datasets to add to raw_data_folder
        param test: list of bool assigning some (or none) as test set(s)
        """

        if self.label_dict_filename.exists():
            ask_proceed("Project already exists. Proceed and make folders again (e.g., if some were deleted)?")(
                self._create_folder_tree
            )()
        else:
            self._create_folder_tree()
        if datasets:
            self.add_datasources(datasets, test)

        if not label_dict_filename:
            save_dict(mask_labels_template, self.label_dict_filename)
            print("Using a template mask_labels.json file. Amend {} later to match your target config.".format(self.label_dict_filename))
        else:
            shutil.copy(label_dict_filename,self.label_dict_filename)

    def _create_folder_tree(self):
        maybe_makedirs(self.project_folder)
        additional_folders=[
            self.raw_data_folder / ("images"),
            self.raw_data_folder / ("masks"),
        ]
        for folder in il.chain(self.folders,additional_folders):
            maybe_makedirs(folder)

    def populate_raw_data_folder(self):
        for ds in self.raw_data_sources:
            dataset_name = ds['dataset_name']
            test = ds['test']
            source_path = ds['source_path']
            images,masks=  self.extract_img_mask_fnames(ds)
            ds_new = {'dataset_name':dataset_name,'source_path':source_path, 'test':test, 'images':images,'masks':masks}
            self._add_dataset(ds_new)

    def extract_img_mask_fnames(self,ds):
            img_fnames = list((ds['source_path'] / ("images")).glob("*"))
            mask_fnames = list((ds['source_path'] / ("masks")).glob("*"))
            img_symlinks,mask_symlinks= [],[]
        
            verified_pairs=[]
            for img_fn in img_fnames:
                verified_pairs.append([img_fn, find_matching_fn(img_fn, mask_fnames)])
            assert (self.paths_exist(verified_pairs)), "(Some) paths do not exist. Fix paths and try again."
            print("self.populating raw data folder (with symlinks)")
            for pair in verified_pairs:
                img_symlink,mask_symlink = self.filepair_symlink(pair)
                img_symlinks.append(img_symlink)
                mask_symlinks.append(mask_symlink)

            return img_symlinks,mask_symlinks


    def filepair_symlink(self,pair:list):
        symlink_fnames= []
        for fn in pair:
            prnt = self.raw_data_folder/fn.parent.name
            fn_out = prnt/fn.name
            try:
                print("SYMLINK created: {0 -> {1}".format(fn,fn_out))
                fn_out.symlink_to(fn)
            except FileExistsError as e:
                print(f"SYMLINK {str(e)}. Skipping...")
            symlink_fnames.append(fn_out)
        return symlink_fnames
        


    def _add_dataset(self,dic):
        if dic_in_list(dic,self.datasets)==False:
            self.datasets.append(dic)
            save_dict(self.datasets,self.raw_dataset_info_filename)
        else: print("Dataset {} already registered with same fileset in project. Will not add".format(dic['dataset_name']))



    @property
    def folders(self):
        self._folders = []
        for key, value in self.__dict__.items():
            if isinstance(value, Path) and "folder" in key:
                self._folders.append(value)
        return self._folders


    @property
    def datasets(self):
        if not hasattr(self,'_datasets'): 
            try: 
                self._datasets = load_dict(self.raw_dataset_info_filename)
            except:  self._datasets = []
        return self._datasets
        

    def purge(self): 
        self.purge_raw_data_folder()
        files_to_del = [self.raw_dataset_info_filename]
        for fn in files_to_del:
            try: 
                print("Deleting {} (if it exists)".format(fn))
                fn.unlink()
            except: pass
    def purge_raw_data_folder(self):
        for f in il.chain.from_iterable([self.raw_data_imgs, self.raw_data_masks]):
            f.unlink()

    def _create_img_mask_symlinks(self, org_names, new_names):
        try:
            a = list(
                il.starmap(lambda x, y: y.symlink_to(x), zip(org_names, new_names))
            )
        except FileExistsError as e:
            print(f"SYMLINK {str(e)}. Skipping...")

    def paths_exist(self, paths: list):
        paths = il.chain.from_iterable(paths)
        all_exist = [[p, p.exists()] for p in paths]
        if not all([a[1] for a in all_exist]):
            print("Some files don't exist")
            print(all_exist)
            return False
        else:
            print("Matching image / mask file pairs found in all raw_data sources")
            return True

    def path_to_dataset(self,folder,test):
            folder = Path(folder)
            dd = {'dataset_name':self.get_dataset_name(folder),   'source_path':folder,'test':test}
            return dd

    def add_datasources(self, datasets: Union[list, str, Path], test=None) -> None:

        if not test:
            test = [
                False,
            ] * len(datasets)
        assert len(datasets) == len(
            test
        ), "datasets and test-status bool lists should have same length"

        datasets = [self.path_to_dataset(ds,ts) for ds,ts in zip(listify(datasets), listify(test))]
        datasets = self.filter_existing_datasets(datasets)
        self.raw_data_sources.extend(datasets)

    

    def add_datasources_xnat(self,xnat_proj:str):
        xnat_shadow_fldr="/s/xnat_shadow/"
        proj = Proj(xnat_proj)
        rc = proj.resource("IMAGE_MASK_FPATHS")
        csv_fn = rc.get("/tmp", extract=True)[0]
        df = pd.read_csv(csv_fn)


    def fold_update_needed(self):
        names = [a.name.split('.')[0] for a in self.raw_data_imgs]
        af= set(names)
        old= set(self.folds['all_cases'])
        dif = af.difference(old)
        old.difference(af)
        d = list(dif)
        n_new = len(d)
        print("New cases have been added (n={0})".format(n_new))
        print("Train valid indices need to be updated")



    def filter_existing_datasets(self,datasets:list):
        # removes duplicates
        names = [ll['dataset_name'] for ll in datasets]
        new_dsets=[]
        for name in names:
            is_new_dset = any([name == dset['dataset_name'] for dset in self.datasets])
            new_dsets.append(not is_new_dset)

        datasets =  list(il.compress(datasets,new_dsets))
        return datasets

    def fold_update_needed(self)->bool:
        n_new = len(self.new_case_ids)
        if n_new>0:
            print("New cases have been added (n={0})".format(n_new))
            print("Train valid indices need to be updated")
            return True
        else: return False

    def update_folds(self)->None:
        print("New cases: {0}\n{1}".format(str(len(self.new_case_ids)),self.new_case_ids))
        folds_new = create_folds(self.new_case_ids)
        self.folds = merge_dicts(self.folds,folds_new)
        save_dict(self.folds, self.validation_folds_filename)


    @ask_proceed("Create train/valid folds (80:20) ")
    def create_train_valid_folds(self):
        print("Existing datasets are :{}".format([[ds['dataset_name'],ds['test']] for ds in self.datasets]))
        print("A fresh train/valid split should be created everytime a new training dataset is added")
        train_val_list =  list(il.chain.from_iterable([ds['images'] for ds in self.datasets if ds['test']==False ]))
        test_list =list(il.chain.from_iterable([ds['images'] for ds in self.datasets if ds['test']==True]))
        json_fname = self.validation_folds_filename
        create_train_valid_test_lists_from_filenames(
             train_val_list,  test_list,0.2,  json_fname, shuffle=False
        )

    @property
    def new_case_ids(self):
        '''
        case ids which have been added to raw data but are not the validation folds
        '''
        names = il.chain.from_iterable([d['images'] for d in self.datasets  if d['test']==False])
        names = [a.name.split('.')[0] for a in names]
        files= set(names)
        folds_old= set(self.folds['all_cases'])
        added = files.difference(folds_old)
        removed = folds_old.difference(files)
        if len(removed)>0: 
            print("Some cases have been removed. The logic to remove them from folds is not implemeneted")
            tr()
        self._new_case_ids = list(added)
        return self._new_case_ids


    def get_dataset_name(self,folder):
        fnames = (folder/ ("images")).glob("*")
        fname = next(fnames)
        pat = r"([a-z]*[\d]*)[-_]"
        res=   re.match(pat, fname.name)[1] 
        return res

    def set_folder_file_names(self):

        common_paths= load_yaml(common_vars_filename)
        self.project_folder = Path(common_paths["projects_folder"]) /self.project_title
        self.cold_datasets_folder = Path(common_paths['cold_storage_folder'])/"datasets"
        self.fixed_dimensions_folder = Path(common_paths["rapid_access_folder"]) / self.project_title
        self.predictions_folder = Path(common_paths[
            "cold_storage_folder"
        ] )/ ("predictions/" + self.project_title)
        self.raw_data_folder = self.cold_datasets_folder / (
                    "raw_data/" + self.project_title
                )
        self.checkpoints_parent_folder = Path(common_paths['cold_storage_folder'])/("checkpoints/" + self.project_title)
        self.configuration_filename = self.project_folder/ ("experiment_configs.xlsx")

        self.fixed_spacings_folder = (
            self.cold_datasets_folder
            / ("preprocessed/fixed_spacings")
            / self.project_title
        )
        self.global_properties_filename = (
            self.project_folder / "global_properties"
        )
        self.patches_folder = self.fixed_dimensions_folder / ("patches")
        self.raw_dataset_properties_filename = (
            self.project_folder / "raw_dataset_properties"
        )

        self.bboxes_voxels_info_filename = self.raw_data_folder / ("bboxes_voxels_info")
        self.validation_folds_filename = self.project_folder/("validation_folds.json")
        self.whole_images_folder = self.fixed_dimensions_folder / ("whole_images")
        self.raw_dataset_info_filename = self.project_folder/("raw_dataset_srcs.pkl")
        self.log_folder = self.project_folder / ("logs")
        self.label_dict_filename =  self.project_folder/("mask_labels.json")


    def _raw_data_files(self, input):
        rdi = [list(a / (input).glob("*")) for a in self.raw_data_folder.glob("*")]
        return list(il.chain.from_iterable(rdi))

    @property
    def raw_data_imgs(self):
        self._raw_data_imgs = (self.raw_data_folder / ("images")).glob("*")
        return list(self._raw_data_imgs)

    @property
    def raw_data_masks(self):
        self._raw_data_masks = (self.raw_data_folder / ("masks")).glob("*")
        return list(self._raw_data_masks)

    @property
    def folds(self):
        if not hasattr(self,'_folds')   :
            self._folds = load_dict(self.validation_folds_filename)
            
        return self._folds

    @folds.setter
    def folds(self,value):
        self._folds = value


    def __len__(self): 
        self._len = len(self.raw_data_imgs)
        assert (self._len == len(self.folds['all_cases'])), "Have you accounted for all files in creating train/valid folds?"
        self._training_data_total = self._len
        return self._len

    def __repr__(self): 
        s = "Project {0}\n{1}".format(self.project_title, self.datasets)
        return s

    @ask_proceed("Remove all project files and folders?")
    def delete(self):
        for folder in self.folders:
            if folder.exists() and self.project_title in str(folder) :
                shutil.rmtree(folder)
        print("Done")

def create_train_valid_test_lists_from_filenames(train_val_list, test_list, pct_valid , json_filename, shuffle=False):
    train_val_ids = [strip_extension(fn.name) for fn in train_val_list]
    test_ids = [strip_extension(fn.name) for fn in test_list]
    folds_dict = create_folds(train_val_ids,test_ids,pct_valid,shuffle=shuffle)
    print("Saving folds to {}  ..".format(json_filename))
    save_dict(folds_dict,json_filename)

def create_folds(train_val_ids,test_ids=[], pct_valid=0.2,shuffle=False):
    if shuffle==True: 
        print("Shuffling all cases")
        random.shuffle(train_val_ids)
    else:    
        print("Putting cases in sorted order")
        train_val_ids.sort()
    final_dict= {"all_cases":train_val_ids+test_ids} 
    n_valid = int(pct_valid*len(train_val_ids))
    final_dict.update({"test_cases": test_ids})
    folds = int(1/pct_valid)
    print("Given proportion {0} of validation files yield {1} folds".format(pct_valid,folds))
    slices = [slice(fold*n_valid,(fold+1)*n_valid) for fold in range(folds)]
    val_cases_per_fold = [train_val_ids[slice] for slice in slices]
    for n in range(folds):
        train_cases_fold = list(set(train_val_ids)-set(val_cases_per_fold[n]))
        fold = {'fold_{}'.format(n):{'train':train_cases_fold, 'valid':val_cases_per_fold[n]}}
        final_dict.update(fold)
    return final_dict


# %%
if __name__ == "__main__":
    P = Project(project_title="l2")
    P.create_project(['/s/datasets_bkp/drli/', '/s/datasets_bkp/lits_segs_improved/', '/s/datasets_bkp/litqsmall/litqsmall', '/s/xnat_shadow/litq'])
    P.add_datasources("/s/datasets_bkp/litsmall")
    # P.create_project([ '/s/xnat_shadow/litq'])
    P.populate_raw_data_folder()
    P.raw_data_imgs
    P.update_folds()
# %%
    P.raw_data_imgs
    P.create_train_valid_folds()

# %%

    P.add_datasources(['/s/datasets_bkp/litsmall/'])
    P.populate_raw_data_folder()
# %%
# %%
    P.load_summary()
    pj = P
    pp(pj)
    len(P)
    P.save_summary()

# %%
       
