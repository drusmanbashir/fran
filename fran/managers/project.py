# %%
from pathlib import Path
import os, sys
import itertools as il
from fastai.vision.augment import store_attr
from fran.utils.dictopts import dic_in_list
from fran.utils.helpers import *

from fran.utils.helpers import DictToAttr, ask_proceed

sys.path += ["/home/ub/Dropbox/code"]
from types import SimpleNamespace
from fran.utils.fileio import *

common_paths_filename = os.environ["FRAN_COMMON_PATHS"]
from fran.utils.templates import mask_labels_template


class Project(DictToAttr):
    def __init__(self, project_title):
        store_attr()

    def create_project(self, datasets: list = None, test: bool = None):
        """
        param datasets: list of datasets to add to raw_data_folder
        param test: list of bool assigning some (or none) as test set(s)
        """

        if self.summary_filename.exists():
            ask_proceed("Project already exists. Proceed and make folders again (e.g., if some were deleted)?")(
                self._create_folder_tree
            )()
        else:
            self._create_folder_tree()
        if datasets:
            self.set_raw_data_sources(datasets, test)
        self._raw_data_folder = self.proj_summary.raw_data_folder


    def _create_folder_tree(self):
        folders = []
        maybe_makedirs(self.project_folder)
        for key, value in self.proj_summary.__dict__.items():
            if isinstance(value, Path) and "folder" in key:
                folders.append(value)
        additional_folders = [
            self.proj_summary.raw_data_folder / ("images"),
            self.proj_summary.raw_data_folder / ("masks"),
        ]
        folders.extend(additional_folders)
        for folder in folders:
            maybe_makedirs(folder)

    def populate_raw_data_folder(self):

        for ds in self.raw_data_sources:
            dataset_name = ds['dataset_name']
            test = ds['test']
            filenames = (ds['source_path'] / ("images")).glob("*")
            pairs = [[img_fn, img_fn.str_replace("images", "masks")] for img_fn in filenames]
            images,masks=[],[]
            if self.paths_exist(pairs) == True:
                print("self.pulating raw data folder (with symlinks)")
                for org_names in pairs:
                    case_filename = org_names[0].name
                    new_names= [
                        self.raw_data_folder / subfolder / case_filename
                        for subfolder in ["images", "masks"]
                    ]
                    images.append(new_names[0])
                    masks.append(new_names[1])

                    self._create_img_mask_symlinks(org_names, new_names)
            ds_new = {'dataset_name':dataset_name,'test':test, 'images':images,'masks':masks}
            self.add_dataset(ds_new)


    def add_dataset(self,dic):
        if dic_in_list(dic,self.datasets)==False:
            self.datasets.append(dic)
            save_dict(self.datasets,self.proj_summary.raw_dataset_info_filename)
        else: print("Dataset {} already registered with same fileset in project. Will not add".format(dic['dataset_name']))



    @property
    def label_dict(self):
        if not self.label_dict_filename.exists():
            save_dict(mask_labels_template, self.label_dict_filename)
            print("Using a template mask_labels.json file. Amend {} later to match your target config.".format(self.label_dict_filename))
        self._label_dict= load_dict(self.label_dict_filename)
        return self._label_dict


    @property
    def datasets(self):
        if not hasattr(self,'_datasets'): 
            try: 
                self._datasets = load_dict(self.proj_summary.raw_dataset_info_filename)
            except:  self._datasets = []
        return self._datasets
        

    def purge(self): 
        self.purge_raw_data_folder()
        files_to_del = self.proj_summary.raw_dataset_info_filename,self.summary_filename
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

    def set_raw_data_sources(self, datasets: Union[list, str, Path], test=None) -> None:
        if not hasattr(self.proj_summary, "raw_data_sources"):
            self.proj_summary.raw_data_sources= []

        datasets = listify(datasets)
        if not test:
            test = [
                False,
            ] * len(datasets)
        assert len(datasets) == len(
            test
        ), "datasets and test-status bool lists should have same length"

        datasets = [Path(ds) for ds in datasets]
        for ds, ts in zip(datasets,test):
            dd = {'dataset_name':self.get_dataset_name(ds),   'source_path':ds,'test':ts}
            self.proj_summary.raw_data_sources.extend([dd])
        self.proj_summary.raw_data_sources = self.unique_list_of_dicts(self.proj_summary.raw_data_sources)


    def unique_list_of_dicts(self,listi):
        # removes duplicates
        return [dict(t) for t in {tuple(d.items()) for d in listi}]

    def save_summary(self):
                save_pickle(self.proj_summary,self.summary_filename)

    def load_summary(self):
        self._proj_summary = load_pickle(self.summary_filename)

    def create_summary_dict(self):
                proj_summary = {"project_title": self.project_title}
                proj_summary["raw_data_folder"] = self.cold_datasets_folder / (
                    "raw_data/" + proj_summary["project_title"]
                )
                proj_summary["project_folder"] = self.project_folder
                
                proj_summary["bboxes_voxels_info_filename"] = (
                    proj_summary["raw_data_folder"] / "bboxes_voxels_info"
                )
                proj_summary["checkpoints_parent_folder"] = self.common_paths[
                    "cold_storage_folder"
                ] / ("checkpoints/" + proj_summary["project_title"])
                proj_summary["configuration_filename"] = proj_summary[
                    "project_folder"
                ] / ("experiment_configs.xlsx")
                proj_summary["fixed_dimensions_folder"] = (
                    self.common_paths["rapid_access_folder"]
                    / proj_summary["project_title"]
                )
                proj_summary["fixed_spacings_folder"] = (
                    self.cold_datasets_folder
                    / ("preprocessed/fixed_spacings")
                    / proj_summary["project_title"]
                )
                proj_summary["global_properties_filename"] = (
                    proj_summary["project_folder"] / "global_properties"
                )
                proj_summary["neptune_folder"] = self.common_paths["neptune_folder"]
                proj_summary["patches_folder"] = proj_summary[
                    "fixed_dimensions_folder"
                ] / ("patches")
                proj_summary["predictions_folder"] = self.common_paths[
                    "cold_storage_folder"
                ] / ("predictions/" + proj_summary["project_title"])
                proj_summary["raw_dataset_properties_filename"] = (
                    proj_summary["project_folder"] / "raw_dataset_properties"
                )
                proj_summary["validation_folds_filename"] = proj_summary[
                    "project_folder"
                ] / ("validation_folds.json")
                proj_summary["whole_images_folder"] = proj_summary[
                    "fixed_dimensions_folder"
                ] / ("whole_images")
                proj_summary['raw_dataset_info_filename'] = proj_summary['project_folder']/("raw_dataset_info.pkl")
                proj_summary["log_folder"] = proj_summary["project_folder"] / ("logs")

                proj_summary['mask_labels'] =self.label_dict


                return SimpleNamespace(**proj_summary)


    @ask_proceed("Create train/valid folds (80:20) ")
    def create_train_valid_folds(self):
        print("Existing datasets are :{}".format([[ds['dataset_name'],ds['test']] for ds in self.datasets]))
        print("A fresh train/valid split should be created everytime a new training dataset is added")
        train_val_list =  list(il.chain.from_iterable([ds['images'] for ds in self.datasets if ds['test']==False ]))
        test_list =list(il.chain.from_iterable([ds['images'] for ds in self.datasets if ds['test']==True]))
        json_fname = self.proj_summary.validation_folds_filename

        create_train_valid_test_lists_from_filenames(
             train_val_list,  test_list,0.2,  json_fname, shuffle=False
        )

    @property
    def summary_filename(self): return self.project_folder/("proj_summary.pkl")
    
    @property
    def label_dict_filename(self):return self.project_folder/("mask_labels.json")

    @property
    def project_folder(self):
        return self.common_paths["projects_folder"] /self.project_title

    @property
    def proj_summary(self):
        if not hasattr(self, "_proj_summary"):
            try: 
                self.load_summary()

            except FileNotFoundError:
                self._proj_summary = self.create_summary_dict()
        return self._proj_summary

    @property
    def common_paths(self):
        if not hasattr(self, "_common_paths"):
            output_dic_ = load_yaml(common_paths_filename)
            output_dic = {}
            for ke, val in output_dic_.items():
                output_dic[ke] = Path(val)
            self._common_paths = output_dic
        return self._common_paths

    @property
    def raw_data_sources(self):
        return self.proj_summary.raw_data_sources

    def get_dataset_name(self,folder):
        fnames = (folder/ ("images")).glob("*")
        fname = next(fnames)
        pat = r"([a-z]*[\d]*)[-_]"
        res=   re.match(pat, fname.name)[1] 
        return res

    @property
    def cold_datasets_folder(self):
        return self.common_paths["cold_storage_folder"] / (
                "datasets"
            )

    @property
    def raw_data_folder(self):
        return self.proj_summary.raw_data_folder

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
            self._folds = load_dict(self.proj_summary.validation_folds_filename)
            
        return self._folds


    def __len__(self): 
        self._len = len(self.raw_data_imgs)
        assert (self._len == len(self.folds['all_cases'])), "Have you accounted for all files in creawting train/valid folds?"
        self._proj_summary.training_data_total = self._len
        return self._len

    def __repr__(self): return "Project"

def create_train_valid_test_lists_from_filenames(train_val_list, test_list, pct_valid , json_filename, shuffle=False):
    pct_valid = 0.2
    train_val_ids = [get_case_id_from_filename(None,fn) for fn in train_val_list]
    test_ids = [get_case_id_from_filename(None,fn) for fn in test_list]
    if shuffle==True: 
        print("Shuffling all files")
        random.shuffle(train_val_ids)
    else:    
        print("Putting files in sorted order")
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
    print("Saving folds to {}  ..".format(json_filename))
    save_dict(final_dict,json_filename)


# %%
if __name__ == "__main__":
    P = Project(project_title="lits_tmp23456")
    P.create_project(['/media/ub/datasets_bkp/lits_short_curate/', '/s/datasets/drli_short/'])
    pj = P.proj_summary
    pp(pj)
    P.save_summary()
# %%
    P.set_raw_data_sources(["/s/datasets/drli_short/"])
    P.populate_raw_data_folder()
    P.raw_data_imgs
    P.create_train_valid_folds()
# %%
    P.load_summary()
    pj = P.proj_summary
    pp(pj)
    len(P)
    P.save_summary()

# %%

# %%
        
