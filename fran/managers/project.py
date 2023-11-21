# %%
import sqlite3
from fastcore.basics import listify
import math
import ipdb
from fastcore.basics import GetAttr
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
import functools as fl
from fastcore.basics import detuplify, store_attr
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
from fran.utils.helpers import pat_full, pat_nodesc, pat_idonly
from contextlib import contextmanager


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
            self._create_folder_tree()

        self.create_table()
        if data_folders:
            datasources = self.add_data(data_folders, test)

    def sql_alter(self, sql_str):
        with db_ops(self.db) as cur:
            res = cur.execute(sql_str)

    def sql_query(self, sql_str,chain_output=False):
        with db_ops(self.db) as cur:
            res = cur.execute(sql_str)
            output = res.fetchall()
            if chain_output==True:
                output= list(il.chain.from_iterable(output))
        return output

    def vars_to_sql(self, dataset_name, img_fn, mask_fn, test):
        case_id = info_from_filename(img_fn.name)['case_id']
        fold = "NULL"
        img_sym = self.symlink_fname(img_fn)
        mask_sym = self.symlink_fname(mask_fn)
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
                "CREATE TABLE {} (ds, case_id, image,mask,img_symlink,mask_symlink,fold INTEGER,test)".format(
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
            ds = Datasource(fldr, test)
            ds = self.filter_existing_images(ds)
            self.populate_tbl(ds)
        self.populate_raw_data_folder()

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
        query_masks = "SELECT mask, mask_symlink FROM datasources"
        pairs = self.sql_query(query_imgs)
        pairs.extend(self.sql_query(query_masks))
        for pair in pairs:
            self.filepair_symlink(*pair)

    def symlink_fname(self, fn):
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
        xnat_shadow_fldr = "/s/xnat_shadow/"
        proj = Proj(xnat_proj)
        rc = proj.resource("IMAGE_MASK_FPATHS")
        csv_fn = rc.get("/tmp", extract=True)[0]
        df = pd.read_csv(csv_fn)

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
        self.global_properties_filename = self.project_folder / "global_properties"
        self.patches_folder = self.fixed_dimensions_folder / ("patches")
        self.raw_dataset_properties_filename = (
            self.project_folder / "raw_dataset_properties"
        )

        self.bboxes_voxels_info_filename = self.raw_data_folder / ("bboxes_voxels_info")
        self.validation_folds_filename = self.project_folder / ("validation_folds.json")
        self.whole_images_folder = self.fixed_dimensions_folder / ("whole_images")
        self.raw_dataset_info_filename = self.project_folder / ("raw_dataset_srcs.pkl")
        self.log_folder = self.project_folder / ("logs")



    def get_unassigned_cases(self):
        ss = "SELECT case_id,image FROM datasources WHERE test=0 AND fold='NULL'"  # only training cases
        qr = self.sql_query(ss)
        qr = list(il.chain(qr))
        print("Cases not assigned to any training fold: {}".format(len(qr)))
        return qr

    def create_df(self, cases):
        self.df = pd.DataFrame(cases, columns=["case_id", "image"])
        self.df["index"] = 0
        self.df["fold"] = np.nan
        case_ids = self.df["case_id"]
        case_ids = case_ids.sort_values()
        case_ids = case_ids.drop_duplicates()

        for ind, case_id in enumerate(case_ids):
            self.df.loc[self.df["case_id"] == case_id, "index"] = ind

    def update_tbl_folds(self):
        dds = []
        for n in range(len(self.df)):
            dd = list(self.df.loc[n, ["fold", "image"]])
            dd[0] = str(dd[0])
            dds.append(dd)
        ss = "UPDATE datasources SET fold= ? WHERE image=?"
        with db_ops(self.db) as cur:
            cur.executemany(ss, dds)

    @ask_proceed("Create train/valid folds (80:20) ")
    def create_folds(self, pct_valid=0.2, shuffle=False):
        self.delete_duplicates()
        cases_unassigned = self.get_unassigned_cases()
        if len(cases_unassigned) > 0:
            self.create_df(cases_unassigned)
            pct_valid = 0.2
            folds = int(1 / pct_valid)
            sl = val_indices(len(self.df), folds)
            print(
                "Given proportion {0} of validation files yield {1} folds".format(
                    pct_valid, folds
                )
            )
            if shuffle == True:
                self.df = self.df.sample(frac=1).reset_index(drop=True)
            else:
                self.df = self.df.sort_values("index")
                self.df = self.df.reset_index(drop=True)

            for n in range(folds):
                self.df.loc[sl[n], "fold"] = n

            self.update_tbl_folds()

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



    def __len__(self):
        ss = "SELECT COUNT (image )from datasources"
        qr = self.sql_query(ss)[0][0]
        return qr

    def __repr__(self):
        try:
            s = "Project {0}\n{1}".format(self.project_title, self.dataset_names)
        except:
            s= "Project {0}\n{1}".format(self.project_title,"Datasets Unknown")
        return s


    @property
    def folders(self):
        self._folders = []
        for key, value in self.__dict__.items():
            if isinstance(value, Path) and "folder" in key:
                self._folders.append(value)
        return self._folders

    @property
    def dataset_names(self):
        ss = "SELECT DISTINCT ds FROM datasources ORDER BY ds"
        qr = self.sql_query(ss,True)
        return qr


    @property
    def raw_data_imgs(self):
        self._raw_data_imgs = (self.raw_data_folder / ("images")).glob("*")
        return list(self._raw_data_imgs)

    @property
    def raw_data_masks(self):
        self._raw_data_masks = (self.raw_data_folder / ("masks")).glob("*")
        return list(self._raw_data_masks)

class Datasource(GetAttr):
    def __init__(self, folder: Union[str, Path], test=False) -> None:
        """
        src_folder: has subfolders 'images' and 'masks'. Files in each are identically named
        """
        self.folder = Path(folder)
        self.name = self.folder.name
        self.integrity_check()
        store_attr(but="folder")

    def integrity_check(self):
        """
        verify name pairs
        any other verifications
        """

        images = list((self.folder / ("images")).glob("*.*"))
        masks = list((self.folder / ("masks")).glob("*.*"))
        assert (
            a := len(images) == (b := len(masks))
        ), "Different lengths of images {0}, and masks {1}.\nCheck your data folder".format(
            a, b
        )
        self.verified_pairs = []
        for img_fn in images:
            self.verified_pairs.append([img_fn, find_matching_fn(img_fn, masks)])
        pp(self.verified_pairs)

    def extract_img_mask_fnames(self, ds):
        img_fnames = list((ds["source_path"] / ("images")).glob("*"))
        mask_fnames = list((ds["source_path"] / ("masks")).glob("*"))
        img_symlinks, mask_symlinks = [], []

        verified_pairs = []
        for img_fn in img_fnames:
            verified_pairs.append([img_fn, find_matching_fn(img_fn, mask_fnames)])
        assert self.paths_exist(
            verified_pairs
        ), "(Some) paths do not exist. Fix paths and try again."
        print("self.populating raw data folder (with symlinks)")
        for pair in verified_pairs:
            img_symlink, mask_symlink = self.filepair_symlink(pair)
            img_symlinks.append(img_symlink)
            mask_symlinks.append(mask_symlink)
        return img_symlinks, mask_symlinks

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


# %%
if __name__ == "__main__":
    P = Project(project_title="litsg")
    ds = "/s/xnat_shadow/litq"
    ds2="/s/datasets_bkp/litqmall"
    ds3="/s/datasets_bkp/drli"
    ds4="/s/datasets_bkp/lits_segs_improved/"
    ds5="/s/datasets_bkp/drli_short/"
    # P.create_project([ds,ds2,ds3,ds4])
    P.create_project([ds5])
    P.create_folds()
    len(P.raw_data_imgs)
    len(P)
# %%

