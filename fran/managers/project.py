# %%
import sqlite3
from label_analysis.totalseg import TotalSegmenterLabels
from fastcore.basics import listify
import ipdb
from fastcore.basics import GetAttr
from monai.utils.enums import StrEnum
from fran.managers.datasource import Datasource, _DS
from fran.preprocessing.datasetanalyzers import case_analyzer_wrapper, import_h5py
from fran.preprocessing.globalproperties import GlobalProperties
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
from contextlib import contextmanager


DS= _DS()

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
    def create(self,mnemonic, datasources: list = None, test: list = None):
        """
        param datasets: list of datasets to add to raw_data_folder
        param test: list of bool assigning some (or none) as test set(s)
        """

        assert not self.db.exists(),   "Project already exists. Use 'add_data' if you want to add data. "
        self.global_properties = {'project_title':self.project_title, 'mnemonic':mnemonic}
        self._create_folder_tree()
        self.create_table()
        if datasources:
            self.add_data(datasources, test)
        self.save_global_properties()
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

    def vars_to_sql(self, ds_name, ds_alias, img_fn, lm_fn, test):
        case_id = info_from_filename(img_fn.name,full_caseid=True)['case_id']
        fold = "NULL"
        img_sym = self.create_raw_ds_fname(img_fn)
        lm_sym = self.create_raw_ds_fname(lm_fn)
        cols = (
            ds_name,
            ds_alias,
            case_id,
            str(img_fn),
            str(lm_fn),
            str(img_sym),
            str(lm_sym),
            fold,
            test,
        )
        return cols

    def create_table(self):
        tbl_name = "datasources"
        if not self.table_exists(tbl_name):
            self.sql_alter(
                "CREATE TABLE {} (ds,alias, case_id, image,lm,img_symlink,lm_symlink,fold INTEGER,test)".format(
                    tbl_name
                )
            )

    def table_exists(self, tbl_name):
        ss = "SELECT name FROM sqlite_schema WHERE type='table' AND name ='{}'".format(
            tbl_name
        )
        aa = self.sql_query(ss)
        return True if len(aa) > 0 else False

    def add_data(self, datasources, test=False):
        test = [False] * len(datasources) if not test else listify(test)
        assert len(datasources) == len(
            test
        ), "Unequal lengths of datafolders and (bool) test status"
        for ds_dict, test in zip(datasources, test):
            fldr = ds_dict['folder']
            ds = Datasource(folder=fldr, name = ds_dict['ds'],alias= ds_dict['alias'],test=test)
            ds = self.filter_existing_images(ds)
            self.populate_tbl(ds)
        self.populate_raw_data_folder()
        self.register_datasources(datasources)

    def _create_folder_tree(self):
        maybe_makedirs(self.project_folder)
        additional_folders = [
            self.raw_data_folder / ("images"),
            self.raw_data_folder / ("lms"),
        ]
        for folder in il.chain(self.folders, additional_folders):
            maybe_makedirs(folder)

    def populate_raw_data_folder(self):
        query_imgs = "SELECT image, img_symlink FROM datasources"
        query_lms = "SELECT lm, lm_symlink FROM datasources"
        pairs = self.sql_query(query_imgs)
        pairs.extend(self.sql_query(query_lms))
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
        for f in il.chain.from_iterable([self.raw_data_imgs, self.raw_data_lms]):
            f.unlink()


    def paths_exist(self, paths: list):
        paths = il.chain.from_iterable(paths)
        all_exist = [[p, p.exists()] for p in paths]
        if not all([a[1] for a in all_exist]):
            print("Some files don't exist")
            print(all_exist)
            return False
        else:
            print("Matching image / lm file pairs found in all raw_data sources")
            return True

    def populate_tbl(self, ds):

        strs = [self.vars_to_sql(ds.name,ds.alias, *pair, ds.test) for pair in ds.verified_pairs]
        with db_ops(self.db) as cur:
            cur.executemany("INSERT INTO datasources VALUES (?,?, ?,?,?,?,?,?,?)", strs)

    def add_datasources_xnat(self, xnat_proj: str):
        proj = Proj(xnat_proj)
        rc = proj.resource("IMAGE_lm_FPATHS")
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
        rapid_access_folder = Path(common_paths["rapid_access_folder"])/self.project_title
        self.project_folder = Path(common_paths["projects_folder"]) / self.project_title
        self.cold_datasets_folder = (
            Path(common_paths["cold_storage_folder"]) / "datasets"
        )
        self.fixed_spacing_folder = self.cold_datasets_folder/("preprocessed/fixed_spacing")/self.project_title
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

        self.global_properties_filename = self.project_folder / "global_properties.json"
        self.patches_folder = rapid_access_folder / ("patches")
        self.cache_folder= rapid_access_folder / ("cache")
        self.lbd_folder= rapid_access_folder / ("lbd")
        self.patches_folder= rapid_access_folder / ("patches")
        self.raw_dataset_properties_filename = (
            self.project_folder / "raw_dataset_properties.pkl"
        )

        self.bboxes_voxels_info_filename = self.raw_data_folder / ("bboxes_voxels_info")
        self.validation_folds_filename = self.project_folder / ("validation_folds.json")
        self.whole_images_folder = rapid_access_folder / ("whole_images")
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
    def _create_folds(self, pct_valid=0.2, shuffle=False):
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

    def register_datasources(self, datasources):
        dicis = []
        for ds in datasources:
                fldr = Path(ds['folder'])
                dataset_name = ds['ds']
                h5_fname = fldr/("fg_voxels.h5")
                dici = {'ds':dataset_name,'alias':ds['alias'],  'folder':str(fldr), 'h5_fname':str(h5_fname) }
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



    def set_lm_groups(self,lm_groups:list= None):
        if not lm_groups  or isinstance(lm_groups,float):
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

    def imported_labels(self, lm_group,input_fldr,labelsets):
        dici = self.global_properties[lm_group]
        dici['imported_folder1']=str(input_fldr)
        dici['imported_labelsets']= labelsets
        self.global_properties[lm_group]=dici
        self.save_global_properties()

    def maybe_store_projectwide_properties(self,clip_range=None,max_cases=250, overwrite=False):
        self._create_folds()
        self.G = GlobalProperties(self,max_cases=max_cases,clip_range=clip_range)
        if not 'labels_all' in self.global_properties.keys() or overwrite==True:
            self.G.store_projectwide_properties()
            self.G.compute_std_mean_dataset()
            self.G.collate_lm_labels()

    def add_plan(self,plan:dict, overwrite_global_properties=False):
        dss = plans['datasources']
        dss= dss.split(",")
        datasources = [getattr(DS,g) for g in dss]
        self.add_data(datasources)
        self.set_lm_groups(plans['lm_groups'])
        self.maybe_store_projectwide_properties(overwrite=overwrite_global_properties)




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
    def raw_data_lms(self):
        self._raw_data_lms = (self.raw_data_folder / ("lms")).glob("*")
        return list(self._raw_data_lms)

    @property
    def lm_remap(self):
        '''
        tells whether postprocessing should remap labels (i.e., if more than 1 lm_group)
        '''
        if len(self.lm_group_keys)>1:
            return True
        else:
            return False

    @property
    def lm_group_keys(self):
        lmgps = "lm_group"
        keys = [k for k in self.global_properties.keys() if lmgps in k]
        return keys



# %%
if __name__ == "__main__":
    from fran.utils.common import *

    P= Project(project_title="nodes3")

    P.create(mnemonic='nodes')
# %%
    conf = ConfigMaker(
        P, raytune=False, configuration_filename=None

    ).config
# %%


    plans = conf['plan1']
    P.add_plan(plans, overwrite_global_properties=False)
# %%
# %%
    # P.set_lm_groups([['litq','litqsmall','drli','lits'],['lidc2']])
    # P.set_lm_groups()

# %%
    # P.add_data(ds5)
    # P.create_project([ds,ds2,ds3,ds4])
    len(P.raw_data_imgs)
    len(P)
# %%
# %%
    P.imported_labels('lm_group2',Path("/s/fran_storage/predictions/totalseg/LITS-827/"),labelsets =[lr,ll])

# %%
    import sqlite3
    db_name = "/s/fran_storage/projects/litsmc/cases.db"

    conn = sqlite3.connect(db_name)
    
    ss = """ALTER TABLE datasources RENAME COLUMN lm to lm"""
    ss = """DELETE FROM datasources WHERE case_id='lits_115'"""
    cur = conn.cursor()
    cur.execute(ss)
    conn.commit()

# %%

    ss = """select * FROM datasources"""
    qr = conn.execute(ss)
    pd.DataFrame(qr)
# %%
    P._create_folds()
    max_cases = 100
    clip_range = [-300,300]

    P.G = GlobalProperties(P,max_cases=max_cases,clip_range=clip_range)
    if not 'labels_all' in P.global_properties.keys() or overwrite==True:
        P.G.store_projectwide_properties()
        P.G.compute_std_mean_dataset()
        P.G.collate_lm_labels()


# %%
    dicis = []
    for fldr  in data_folders:
            dataset_name = fldr.name
            fldr = Path(fldr)
            h5_fname = fldr/("fg_voxels.h5")
            dici = {'ds':dataset_name, 'folder':str(fldr), 'h5_fname':str(h5_fname) }
            dicis.append(dici)

# %%

    dss = plans['datasources']
    dss= dss.split(",")
    datasources = [getattr(DS,g) for g in dss]

    test=None
# %%

# %%

    strs = [P.vars_to_sql(ds.name,ds.alias, *pair, ds.test) for pair in ds.verified_pairs]
    with db_ops(P.db) as cur:
        cur.executemany("INSERT INTO datasources VALUES (?,?, ?,?,?,?,?,?,?)", strs)
# %%
