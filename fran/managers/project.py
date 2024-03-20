# %%

import sqlite3
from fastcore.basics import listify
import ipdb
from fastcore.basics import GetAttr
from monai.utils.enums import StrEnum
from fran.preprocessing.datasetanalyzers import case_analyzer_wrapper
from fran.utils.string import (
    cleanup_fname,
    drop_digit_suffix,
    info_from_filename,
    strip_extension,
)

tr = ipdb.set_trace

from pathlib import Path
import os, sys
import itertools as il
from fastcore.basics import detuplify, store_attr
from fran.utils.dictopts import dic_in_list
from fran.utils.helpers import *
import shutil
from fran.utils.helpers import DictToAttr, ask_proceed

sys.path += ["/home/ub/Dropbox/code"]
from fran.utils.fileio import *

if "XNAT_CONFIG_PATH" in os.environ:
    from xnat.object_oriented import *
common_vars_filename = os.environ["FRAN_COMMON_PATHS"]
from fran.utils.templates import mask_labels_template
from fran.utils.helpers import pat_full, pat_nodesc, pat_idonly
from contextlib import contextmanager

class DS(StrEnum):
    lits="/s/datasets_bkp/lits_segs_improved/"
    litq="/s/xnat_shadow/litq"
    drli_short="/s/datasets_bkp/drli_short/"
    drli="/s/datasets_bkp/drli/"
    litqsmall="/s/datasets_bkp/litqsmall/"
    lidc2="/s/xnat_shadow/lidc2"
    lidctmp="/s/xnat_shadow/lidctmp"
    totalseg = "/s//xnat_shadow/totalseg"


def val_indices(a, n):
    a = a - 1
    k, m = divmod(a, n)
    return [slice(i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(n)]


@contextmanager
def db_ops(db_name):
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



class Project(DictToAttr):
    def __init__(self, project_title):
        store_attr()
        self.set_folder_file_names()
        self.db = self.project_folder / ("cases.db")

        if self.global_properties_filename.exists():
            self.global_properties = load_dict(self.global_properties_filename)
    def create_project(self, data_folders: list = None, test: list = None):
        """
        param datasets: list of datasets to add to raw_data_folder
        param test: list of bool assigning some (or none) as test set(s)
        """

        if self.db.exists():
            ask_proceed(
                "Project already exists. Proceed and make folders again (e.g., if some were deleted)?"
            )(self._create_folder_tree)()
            
        else:
            self.global_properties = {'project_title':self.project_title}
            self._create_folder_tree()

        self.create_table()
        if data_folders:
            self.add_data(data_folders, test)
    def save_global_properties(self):
            save_dict(self.global_properties, self.global_properties_filename)

    def sql_alter(self, sql_str):
        with db_ops(self.db) as cur:
            cur.execute(sql_str)

    def sql_query(self, sql_str,chain_output=False):
        with db_ops(self.db) as cur:
            res = cur.execute(sql_str)
            output = res.fetchall()
            if chain_output==True:
                output= list(il.chain.from_iterable(output))
        return output

    def vars_to_sql(self, dataset_name, img_fn, mask_fn, test):
        case_id = info_from_filename(img_fn.name,full_caseid=True)['case_id']
        fold = "NULL"
        img_sym = self.create_raw_ds_fname(img_fn)
        mask_sym = self.create_raw_ds_fname(mask_fn)
        cols = (
            dataset_name,
            case_id,
            str(img_fn),
            str(mask_fn),
            str(img_sym),
            str(mask_sym),
            fold,
            test,
        )
        return cols

    def create_table(self):
        tbl_name = "datasources"
        if not self.table_exists(tbl_name):
            self.sql_alter(
                "CREATE TABLE {} (ds, case_id, image,lm,img_symlink,lm_symlink,fold INTEGER,test)".format(
                    tbl_name
                )
            )

    def table_exists(self, tbl_name):
        ss = "SELECT name FROM sqlite_schema WHERE type='table' AND name ='{}'".format(
            tbl_name
        )
        aa = self.sql_query(ss)
        return True if len(aa) > 0 else False

    def add_data(self, data_folders, test=False):
        data_folders = listify(data_folders)
        test = [False] * len(data_folders) if not test else listify(test)
        assert len(data_folders) == len(
            test
        ), "Unequal lengths of datafolders and (bool) test status"
        for fldr, test in zip(data_folders, test):
            ds = Datasource(folder=fldr, test=test)
            ds = self.filter_existing_images(ds)
            self.populate_tbl(ds)
        self.populate_raw_data_folder()
        self.register_datasources()

    def _create_folder_tree(self):
        maybe_makedirs(self.project_folder)
        additional_folders = [
            self.raw_data_folder / ("images"),
            self.raw_data_folder / ("masks"),
        ]
        for folder in il.chain(self.folders, additional_folders):
            maybe_makedirs(folder)

    def populate_raw_data_folder(self):
        query_imgs = "SELECT image, img_symlink FROM datasources"
        query_masks = "SELECT mask, lm_symlink FROM datasources"
        pairs = self.sql_query(query_imgs)
        pairs.extend(self.sql_query(query_masks))
        for pair in pairs:
            self.filepair_symlink(*pair)

    def create_raw_ds_fname(self, fn):
        prnt = self.raw_data_folder / fn.parent.name
        fn_out = prnt / fn.name
        return fn_out

    def filepair_symlink(self,fn,fn_out):
        fn , fn_out = Path(fn), Path(fn_out)
        try:
            fn_out.symlink_to(fn)
            print("SYMLINK created: {0} -> {1}".format(fn, fn_out))
        except FileExistsError as e:
            print(f"SYMLINK {str(e)}. Skipping...")


    def delete_duplicates(self):
        ss = """DELETE FROM datasources WHERE rowid NOT IN (SELECT MIN(rowid) FROM datasources GROUP BY image)
        """
        self.sql_alter(ss)



    def purge(self):
        self.purge_raw_data_folder()
        files_to_del = [self.raw_dataset_info_filename]
        for fn in files_to_del:
            try:
                print("Deleting {} (if it exists)".format(fn))
                fn.unlink()
            except:
                pass

    def purge_raw_data_folder(self):
        for f in il.chain.from_iterable([self.raw_data_imgs, self.raw_data_masks]):
            f.unlink()


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

    def populate_tbl(self, ds):
        strs = [self.vars_to_sql(ds.name, *pair, ds.test) for pair in ds.verified_pairs]
        with db_ops(self.db) as cur:
            cur.executemany("INSERT INTO datasources VALUES (?,?,?,?,?,?,?,?)", strs)

    def add_datasources_xnat(self, xnat_proj: str):
        proj = Proj(xnat_proj)
        rc = proj.resource("IMAGE_MASK_FPATHS")
        csv_fn = rc.get("/tmp", extract=True)[0]
        pd.read_csv(csv_fn)

    def filter_existing_images(self, ds):
        ss = "SELECT image FROM datasources WHERE ds='{}'".format(ds.name)
        with db_ops(self.db) as cur:
            res = cur.execute(ss)
            pa = res.fetchall()
        existing_images = list(il.chain.from_iterable(pa))
        if len(existing_images) > 0:
            print(
                "Datasource {} exists already. Checking for new files in added folder".format(
                    ds.name
                )
            )
            existing_images = [x in existing_images for x in ds.images]
            ds.verified_pairs = list(il.compress(ds.verified_pairs, existing_images))
            if ln := len(ds) > 0:
                print("{} new files found. Adding to db.".format(ln))
            else:
                print("No new files to add from datasource {}".format(ds.name))
        return ds

    def set_folder_file_names(self):
        common_paths = load_yaml(common_vars_filename)
        self.project_folder = Path(common_paths["projects_folder"]) / self.project_title
        self.cold_datasets_folder = (
            Path(common_paths["cold_storage_folder"]) / "datasets"
        )
        self.fixed_dimensions_folder = (
            Path(common_paths["rapid_access_folder"]) / self.project_title
        )
        self.predictions_folder = Path(common_paths["cold_storage_folder"]) / (
            "predictions/" + self.project_title
        )
        self.raw_data_folder = self.cold_datasets_folder / (
            "raw_data/" + self.project_title
        )
        self.checkpoints_parent_folder = (
            Path(common_paths["checkpoints_parent_folder"]) / self.project_title
        )
        self.configuration_filename = self.project_folder / ("experiment_configs.xlsx")

        self.fixed_spacings_folder = (
            self.cold_datasets_folder
            / ("preprocessed/fixed_spacings")
            / self.project_title
        )
        self.global_properties_filename = self.project_folder / "global_properties.json"
        self.patches_folder = self.fixed_dimensions_folder / ("patches")
        self.raw_dataset_properties_filename = (
            self.project_folder / "raw_dataset_properties.pkl"
        )

        self.bboxes_voxels_info_filename = self.raw_data_folder / ("bboxes_voxels_info")
        self.validation_folds_filename = self.project_folder / ("validation_folds.json")
        self.whole_images_folder = self.fixed_dimensions_folder / ("whole_images")
        self.raw_dataset_info_filename = self.project_folder / ("raw_dataset_srcs.pkl")
        self.log_folder = self.project_folder / ("logs")

    @property
    def case_ids(self):
        ss = "SELECT case_id FROM datasources "  # only training cases
        qr = self.sql_query(ss)
        qr = list(il.chain.from_iterable(qr))
        return qr

    def get_unassigned_cases(self):
        ss = "SELECT case_id, image FROM datasources WHERE test=0 AND fold='NULL'"  # only training cases
        qr = self.sql_query(ss)
        qr = list(il.chain(qr))
        print("Cases not assigned to any training fold: {}".format(len(qr)))
        return qr

    def create_df_folds(self, cases):
        self.df_folds = pd.DataFrame(cases, columns=["case_id", "image"])
        self.df_folds["index"] = 0
        self.df_folds["fold"] = np.nan
        case_ids = self.df_folds["case_id"]
        case_ids = case_ids.sort_values()
        case_ids = case_ids.drop_duplicates()

        for ind, case_id in enumerate(case_ids):
            self.df_folds.loc[self.df_folds["case_id"] == case_id, "index"] = ind

    def update_tbl_folds(self):
        dds = []
        for n in range(len(self.df_folds)):
            dd = list(self.df_folds.loc[n, ["fold", "image"]])
            dd[0] = str(dd[0])
            dds.append(dd)
        ss = "UPDATE datasources SET fold= ? WHERE image=?"
        with db_ops(self.db) as cur:
            cur.executemany(ss, dds)

    # @ask_proceed("Create train/valid folds (80:20) ")
    def create_folds(self, pct_valid=0.2, shuffle=False):
        self.delete_duplicates()
        cases_unassigned = self.get_unassigned_cases()
        if len(cases_unassigned) > 0:
            self.create_df_folds(cases_unassigned)
            pct_valid = 0.2
            folds = int(1 / pct_valid)
            sl = val_indices(len(self.df_folds), folds)
            print(
                "Given proportion {0} of validation files yield {1} folds".format(
                    pct_valid, folds
                )
            )
            if shuffle == True:
                self.df_folds = self.df_folds.sample(frac=1).reset_index(drop=True)
            else:
                self.df_folds = self.df_folds.sort_values("index")
                self.df_folds = self.df_folds.reset_index(drop=True)

            for n in range(folds):
                self.df_folds.loc[sl[n], "fold"] = n

            self.update_tbl_folds()

    def register_datasources(self):
        ss = "SELECT DISTINCT ds FROM datasources ORDER BY ds"
        qr = self.sql_query(ss,True)
        dicis=[]
        for q in qr:
            fldr = Path(getattr(DS,q))
            h5_fname = fldr/("fg_voxels.h5")
            dici = {'ds':q, 'folder':str(fldr), 'h5_fname':str(h5_fname) }
            dicis.append(dici)
        self.global_properties['datasources'] = dicis
        self.save_global_properties()
        return dicis



    def get_train_val_files(self,fold):
        ss_train = "SELECT img_symlink FROM datasources WHERE fold<>{}".format(fold)
        ss_val = "SELECT img_symlink FROM datasources WHERE fold={}".format(fold)

        train_files,val_files = self.sql_query(ss_train,True),self.sql_query(ss_val,True)
        train_files =[Path(fn).name for fn in train_files]
        val_files =[Path(fn).name for fn in val_files]
        return train_files,val_files


    @ask_proceed("Remove all project files and folders?")
    def delete(self):
        for folder in self.folders:
            if folder.exists() and self.project_title in str(folder):
                shutil.rmtree(folder)
        print("Done")



    def set_label_groups(self,lm_groups:list= None):
        if lm_groups is None:
            self.global_properties['lm_group1']= {'ds': self.datasources}
        elif isinstance(lm_groups[0],Union[list,tuple]):  # list of list
            gps_all = list(il.chain.from_iterable(lm_groups))
            assert set(gps_all) == set(self.datasources),"Expected all datasets {} in lm_groups".format(self.datasources)
            for idx, grp in enumerate(lm_groups):
                for ds in grp:
                   assert ds in self.datasources, "{} not in dataset names".format(ds)
                self.global_properties[f'lm_group{idx+1}']={'ds': grp}
        else:
            assert set(lm_groups) == set(self.datasources),"Expected all datasets {} in lm_groups".format(self.datasources)
            self.global_properties[f'lm_group1']=lm_groups
        print("LM groups created")
        for key in self.global_properties.keys():
            if 'lm_group' in key:
                print(self.global_properties[key])
        self.save_global_properties()




    def __len__(self):
        ss = "SELECT COUNT (image )from datasources"
        qr = self.sql_query(ss)[0][0]
        return qr

    def __repr__(self):
        try:
            s = "Project {0}\n{1}".format(self.project_title, self.datasources)
        except:
            s= "Project {0}\n{1}".format(self.project_title,"Datasets Unknown")
        return s

    @property
    def df(self):
        ss = """select * FROM datasources"""
        with db_ops(self.db) as cur:
                qr = cur.execute(ss)
                colnames = [q[0] for q in qr.description]
                df = pd.DataFrame(qr, columns =colnames)
        return df



    @property
    def datasources(self):
        dses = [a['ds'] for a in self.global_properties['datasources']]
        return  dses


    @property
    def folders(self):
        self._folders = []
        for key, value in self.__dict__.items():
            if isinstance(value, Path) and "folder" in key:
                self._folders.append(value)
        return self._folders


    @property
    def raw_data_imgs(self):
        self._raw_data_imgs = (self.raw_data_folder / ("images")).glob("*")
        return list(self._raw_data_imgs)

    @property
    def raw_data_masks(self):
        self._raw_data_masks = (self.raw_data_folder / ("masks")).glob("*")
        return list(self._raw_data_masks)

    @property
    def lm_remap(self):
        '''
        tells whether postprocessing should remap labels (i.e., if more than 1 lm_group)
        '''
        
        key = 'lm_group'
        keys=[]
        for k in self.global_properties.keys():
            if key in k:
                keys.append(k)
        if len(keys)>1:
            return True
        else:
            return False



class Datasource(GetAttr):
    def __init__(self, folder: Union[str, Path],name:str=None,bg_label=0, test=False) -> None:
        """
        src_folder: has subfolders 'images' and 'masks'. Files in each are identically named
        """
        self.bg_label = bg_label
        self.folder = Path(folder)
        self.test=test
        self.h5_fname = self.folder / "fg_voxels.h5"
        self.raw_dataset_properties_filename = self.folder /"raw_dataset_properties.pkl"
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
        masks = list((self.folder / ("masks")).glob("*"))
        assert (
            a := len(images) == (b := len(masks))
        ), "Different lengths of images {0}, and masks {1}.\nCheck your data folder".format(
            a, b
        )
        self.verified_pairs = []
        for img_fn in images:
            self.verified_pairs.append([img_fn, find_matching_fn(img_fn, masks)])
        print("Verified filepairs are matched")


    def filter_unprocessed_cases(self):
        '''
        Loads project.raw_dataset_properties_filename to get list of already completed cases.
        Any new cases will be processed and added to project.raw_dataset_properties_filename
        '''
        
        try:
            self.raw_dataset_properties = load_dict(self.raw_dataset_properties_filename)
            prev_processed_cases = set([b['case_id'] for b in self.raw_dataset_properties])
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
        # self.store_raw_dataset_properties() # redundant


    def dump_to_h5(self):
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


    def extract_img_mask_fnames(self, ds):
        img_fnames = list((ds["source_path"] / ("images")).glob("*"))
        mask_fnames = list((ds["source_path"] / ("masks")).glob("*"))
        img_symlinks, lm_symlinks = [], []

        verified_pairs = []
        for img_fn in img_fnames:
            verified_pairs.append([img_fn, find_matching_fn(img_fn, mask_fnames)])
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
    def masks(self):
        masks = [x[1] for x in self.verified_pairs]
        return masks

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
    P= Project(project_title="totalseg")
    liver_ds = [DS.litq,DS.litqsmall,DS.lits,DS.drli]
    # P.create_project([DS.litq,DS.litqsmall,DS.lits,DS.drli,DS.lidc2])

    P.create_project([DS.drli_short,DS.lidctmp])
    P.create_project([DS.totalseg])
    # P.add_data(DS.litq)
# %%
    P.set_label_groups([['litq','litqsmall','drli','lits'],['lidc2']])
    P.set_label_groups([['drli'],['lidc2']])
    P.set_label_groups()
# %%
    # P.add_data(ds5)
    # P.create_project([ds,ds2,ds3,ds4])
    P.create_folds()
    len(P.raw_data_imgs)
    len(P)
# %%
    debug=True
    D2 = Datasource(DS.totalseg)

    D2.process_new_cases(debug=debug)


# %%
# %%
    import sqlite3
    db_name = "/s/fran_storage/projects/totalseg/cases.db"

    conn = sqlite3.connect(db_name)
    
    ss = """ALTER TABLE datasources RENAME COLUMN mask to lm"""
    conn.execute(ss)
# %%
    # %%
    ss = """select * FROM datasources"""
    qr = conn.execute(ss)
    pd.DataFrame(qr)
# %%

