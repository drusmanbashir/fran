# %%
import sqlite3
from send2trash import send2trash
import ipdb

from batchgenerators.utilities.file_and_folder_operations import List
from fastcore.basics import listify
from utilz.config_parsers import ConfigMaker
from utilz.string import (
    info_from_filename,
)

tr = ipdb.set_trace

from pathlib import Path
import os, sys
import itertools as il
from fastcore.basics import store_attr
from utilz.helpers import *
from utilz.helpers import DictToAttr, ask_proceed

sys.path += ["/home/ub/Dropbox/code"]
from utilz.fileio import *

if "XNAT_CONFIG_PATH" in os.environ:
    from xnat.object_oriented import *
common_vars_filename = os.environ["FRAN_COMMON_PATHS"]
COMMON_PATHS = load_yaml(common_vars_filename)
from contextlib import contextmanager
import shutil
import string

import ipdb
import SimpleITK as sitk
from fastcore.basics import GetAttr, Union
from label_analysis.helpers import (to_binary, to_int)

from fran.preprocessing.datasetanalyzers import (case_analyzer_wrapper,
                                                 import_h5py)
from utilz.fileio import load_dict, save_dict
from utilz.helpers import find_matching_fn, multiprocess_multiarg
from utilz.string import info_from_filename, str_to_path

tr = ipdb.set_trace
from pathlib import Path

def subscript_generator():
        letters = list(string.ascii_letters)
        while(letters):
            yield letters.pop(0)




def change_caseid(filepair, new_cid):
        for fn in filepair:
            cid_old = info_from_filename(fn.name,full_caseid=True)['case_id']
            fn_out = fn.str_replace(cid_old,new_cid)
            print("--New filename: ",fn_out)
            shutil.move(fn,fn_out)

@str_to_path()
def fix_repeat_caseids(parent_folder):
        ## parent_folder mist have subfolders images and lms
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
            print("Fixing: ",f['filepair'][0])
            change_caseid(f['filepair'],f['new_cid'])





class _DS():
    '''
    each folder has subfolder images and lms
    if a member has a 2-tuple value intead of 1, the send element is alias of the member dataset. This alias is used to match filenames with correct dataset
    '''
    def __init__(self) -> None:
        # for any datasource, an alias matching the {projectname}_ part of the filename should be given if the ds name is non-standard, e.g., drli_short
            self.lits={'ds': 'lits', 'folder': "/s/datasets_bkp/lits_segs_improved/", 'alias':None}
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



class Project(DictToAttr):
    """
    Represents a project which includes managing data sources, manipulating project-wide settings, 
    and interacting with a database. 

    This class supports various operations needed to manage data workflows efficiently, using 
    methods that simplify folder structure creation, data addition, and SQL queries.

    Attributes
    ----------
    project_title : str
        The title of the project.

    Main Methods
    -------
    create(mnemonic, datasources=list, test=list)
        Create the project structure and initialize with data.
    add_data(datasources, test=False)
        Adds data sources to the project.
    maybe_store_projectwide_properties()

    Example
    -------
    Typical use case for setting up a project and adding data:

    >>> from fran.utils.common import DS

    >>> P = Project(project_title="nodes")
    >>> # Create the project structure
    >>> P.create(mnemonic='nodes')

    >>> # Add data sources
    >>> P.add_data([DS.nodes, DS.nodesthick])

    >>> # Process and store project-wide properties. This also creates  5 folds .  Has to be run atleast once
    >>>conf = ConfigMaker(
        P, raytune=False, configuration_filename=None

    ).config
    >>> # Now add a main plan. this creates lm_groups. Also computes dataset properties, mean, std. Vital to do this once. The plan should be the main, default plan with all datasources included.
    >>>plans = conf['plan1']
    >>>P.add_main_plan(plans)
    """


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
        """
        Execute an SQL query and fetch all results.

        This method uses a database context manager to execute the provided SQL query string and 
        fetches the results. Optionally, it can flatten the output into a single list if the 
        query returns a nested list of tuples.

        Parameters
        ----------
        sql_str : str
            The SQL query string to be executed.
        chain_output : bool, optional
            If True, the output list of tuples is flattened into a single list using itertools' chaining. 
            Default is False.

        Returns
        -------
        list
            A list of results from the SQL query execution. If `chain_output` is set to True, the list is 
            flattened, otherwise, it returns a list of tuples.

        Examples
        --------
        >>> results = project.sql_query("SELECT * FROM table_name")
        >>> results_flat = project.sql_query("SELECT id FROM table_name", chain_output=True)

        Notes
        -----
        - This method relies on a context manager `db_ops` which is assumed to be defined within the class 
          or accessible in the same module.
        """
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
        """
        Create the `datasources` table in the database if it doesn't exist.
        """
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

    def add_data(self, datasources :List, test=False):
        """
        Add multiple datasources to the project database.

        Parameters
        ----------
        datasources : list of Datasource
            List of datasource objects to add.
        test : bool or list of bool, optional
            Boolean flags indicating which datasources are for testing.
        """
        #list of DS objects, e.g., DS.nodes
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
        """
        Populate the raw data folder with symbolic links to image and label map files.
        """
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
        """
        Delete duplicate entries in the `datasources` table, keeping the first instance of each.
        """
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
        """
        Verify if all paths in a list exist.

        Parameters
        ----------
        paths : list of Path
            Paths to check.

        Returns
        -------
        bool
            True if all paths exist, False otherwise.
        """
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
        """
        Populate the database with verified datasource pairs.

        Parameters
        ----------
        ds : Datasource
            Datasource object with verified file pairs.
        """
        strs = [self.vars_to_sql(ds.name,ds.alias, *pair, ds.test) for pair in ds.verified_pairs]
        with db_ops(self.db) as cur:
            cur.executemany("INSERT INTO datasources VALUES (?,?, ?,?,?,?,?,?,?)", strs)

    def add_datasources_xnat(self, xnat_proj: str):
        """
        Add datasources from an XNAT project.

        Parameters
        ----------
        xnat_proj : str
            The XNAT project name.
        """
        proj = Proj(xnat_proj)
        rc = proj.resource("IMAGE_lm_FPATHS")
        csv_fn = rc.get("/tmp", extract=True)[0]
        pd.read_csv(csv_fn)

    def filter_existing_images(self, ds):
        """
        Filter out images that already exist in the database.

        Parameters
        ----------
        ds : Datasource
            Datasource to filter.

        Returns
        -------
        Datasource
            Filtered datasource with only new images.
        """

        print("="*50)
        print("Filtering datasource: ",ds.name)
        ss = "SELECT image FROM datasources WHERE ds='{}'".format(ds.name)
        with db_ops(P.db) as cur:
            res = cur.execute(ss)
            pa = res.fetchall()
        existing_images = list(il.chain.from_iterable(pa))
        existing_images = [Path(x) for x in existing_images]
        if len(existing_images) > 0:
            print(
                "Datasource {} exists already. Checking for new files in added folder".format(
                    ds.name
                )
            )
            remaining_images_bool = [x not in existing_images for x in ds.images]
            ds.verified_pairs= list(il.compress(ds.verified_pairs, remaining_images_bool))
            if (ln := len(ds)) > 0:
                print("{} new files found. Adding to db.".format(ln))
            else:
                print("No new files to add from datasource {}".format(ds.name))

        print("="*50)

        return ds

    def set_folder_file_names(self):
        rapid_access_folder = Path(COMMON_PATHS["rapid_access_folder"])/self.project_title
        self.project_folder = Path(COMMON_PATHS["projects_folder"]) / self.project_title
        self.cold_datasets_folder = (
            Path(COMMON_PATHS["cold_storage_folder"]) / "datasets"
        )
        self.fixed_spacing_folder = self.cold_datasets_folder/("preprocessed/fixed_spacing")/self.project_title
        self.fixed_size_folder = self.cold_datasets_folder/("preprocessed/fixed_size")/self.project_title
        self.predictions_folder = Path(COMMON_PATHS["cold_storage_folder"]) / (
            "predictions/" + self.project_title
        )
        self.raw_data_folder = self.cold_datasets_folder / (
            "raw_data/" + self.project_title
        )
        self.checkpoints_parent_folder = (
            Path(COMMON_PATHS["checkpoints_parent_folder"]) / self.project_title
        )
        self.configuration_filename = self.project_folder / ("experiment_configs.xlsx")

        self.global_properties_filename = self.project_folder / "global_properties.json"
        self.patches_folder = rapid_access_folder / ("patches")
        self.cache_folder= rapid_access_folder / ("cache")
        self.lbd_folder= rapid_access_folder / ("lbd")
        self.pbd_folder= rapid_access_folder / ("pbd")
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
        """
        Register new datasources and save to global properties.

        Parameters
        ----------
        datasources : list of Datasource
            List of datasource dictionaries.

        Returns
        -------
        list of dict of new datasources.
            
        """
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


    #NOTE: Later functions patch repeated case ids (e.g., LBGgenerator) so that there is 49,49a, 49b also lm_fnames have substrings 'label-' etc. Fix databases so that after LBD generates new tables are added. Consider updating case ids for repeat ids perhaps
    def get_train_val_files(self, fold: int = None, ds: Union[str, List[str]] = None):
        """
        Retrieves the file paths (img_symlink) for training and validation sets based on the given fold,
        optionally filtering by the provided datasource(s).

        Parameters
        ----------
        fold : int, optional
            The fold number used to split the data into training and validation sets. 
            If None, all folds are returned in a single list
            
        ds : str or list of str, optional
            A string or list representing one or more datasources. If it contains commas, 
            it is treated as a list of datasources.

        Returns
        -------
        tuple of lists
            - train_files: A list of file paths (img_symlink) assigned to training.
            - val_files: A list of file paths (img_symlink) assigned to validation.
        """
        if not ds: # default datasources are all datasources in the project
            ds = self.datasources

        # Build SQL queries
        ss_train = self.build_sql_query(fold, ds, is_validation=False)
        train_files = self.fetch_files(ss_train)


        if fold:
            ss_val = self.build_sql_query(fold, ds, is_validation=True)
            val_files = self.fetch_files(ss_val)
            return train_files, val_files
        else:
            return train_files


    def build_sql_query(self, fold: int, ds: Union[str, List[str]], is_validation: bool) -> str:
        """
        Builds the SQL query for fetching files based on the fold and datasource.

        Parameters
        ----------
        fold : int, optional
            The fold number. If None, the fold condition is ignored.
        ds : str or list of str, optional
            Datasource(s) to filter by. If None, all datasources are selected.
        is_validation : bool
            Whether to build the query for the validation set (True) or training set (False).

        Returns
        -------
        str
            The constructed SQL query.
        """
        query = "SELECT img_symlink FROM datasources"
        
        conditions = []

        # Add fold condition
        if isinstance(fold, int):
            fold_condition = "fold = {}".format(fold) if is_validation else "fold <> {}".format(fold)
            conditions.append(fold_condition)

        # Add datasource condition
        if ds:
            ds_condition = self.build_ds_condition(ds)
            conditions.append(ds_condition)

        # Append conditions to the query
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        return query


    def build_ds_condition(self, ds: Union[str, List[str]]) -> str:
        """
        Builds the datasource condition for the SQL query.

        Parameters
        ----------
        ds : str or list of str
            Datasource(s) to filter by.

        Returns
        -------
        str
            The datasource condition for the SQL query.
        """
        if isinstance(ds, str) and "," in ds:
            ds_list = [d.strip() for d in ds.split(",")]
        else:
            ds_list = [ds] if isinstance(ds, str) else ds

        return "ds IN ({})".format(", ".join("'{}'".format(d) for d in ds_list))


    def fetch_files(self, query: str) -> List[str]:
        """
        Executes the SQL query and returns the list of file names.

        Parameters
        ----------
        query : str
            The SQL query to execute.

        Returns
        -------
        list of str
            The list of file names (img_symlink) extracted from the query result.
        """
        result = self.sql_query(query, True)
        return [Path(fn).name for fn in result]

    @ask_proceed("Remove all project files and folders?")
    def delete(self):
        for folder in self.folders:
            if folder.exists() and self.project_title in str(folder):
                # shutil.rmtree(folder)
                send2trash(folder)
        print("Done")



    def set_lm_groups(self,lm_groups:list= None):
        """
        Defines and assigns label groups (lm_groups) to the project. The idea behind lm_groups is that labels in each group are treated uniquely and any overlapping labels with another (preceding) gruops are relabelled serially starting at the last label of the previous lm_group

        Parameters
        ----------
        lm_groups : list, optional
            A list defining groups of datasets for label management. The groups can be either:
            - `None` or a single list, which will create a default group using all datasets.
            - A list of lists/tuples, where each inner list represents a group of datasets.
            - A single list containing all datasets, which assigns all datasets to a single group.
            
            For example:
            - If `lm_groups` is `None`, all datasets are assigned to a single group (`lm_group1`).
            - If `lm_groups` contains lists like `[['group1', 'group2'], ['group3']]`, 
              the datasets will be split across multiple groups (`lm_group1`, `lm_group2`, etc.).

        Raises
        ------
        AssertionError
            If the datasets in `lm_groups` do not match the datasets registered in the project.

        Notes
        -----
        - This function organizes datasets into label groups for training and validation purposes.
        - The label groups are saved to the project's global properties and can be retrieved later for use in different stages of the project, such as data preprocessing or model training.

        Example
        -------
        >>> project.set_lm_groups([['ds1', 'ds2'], ['ds3']])
        This will assign `ds1` and `ds2` to `lm_group1`, and `ds3` to `lm_group2`.
        """
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

    def imported_labels(self, lm_group: str,input_fldr: Path,labelsets: list):
        """
        Save information about imported labels for a specific label group.

        Parameters
        ----------
        lm_group : str
            Label group identifier.
        input_fldr : Path
            Folder containing imported labels.
        labelsets : list
            List of imported label sets.
        """
        dici = self.global_properties[lm_group]
        dici['imported_folder1']=str(input_fldr)
        dici['imported_labelsets']= labelsets
        self.global_properties[lm_group]=dici
        self.save_global_properties()

    def maybe_store_projectwide_properties(self,clip_range=None,max_cases=250, overwrite=False):
        """
        Store global properties like dataset mean and standard deviation.

        Parameters
        ----------
        clip_range : tuple, optional
            Range for intensity clipping.
        max_cases : int, optional
            Maximum cases for computing properties.
        overwrite : bool, optional
            Whether to overwrite existing properties.
        """
        from fran.preprocessing.globalproperties import GlobalProperties
        self._create_folds()
        self.G = GlobalProperties(self,max_cases=max_cases,clip_range=clip_range)
        if not 'labels_all' in self.global_properties.keys() or overwrite==True:
            self.G.store_projectwide_properties()
            self.G.compute_std_mean_dataset()
            self.G.collate_lm_labels()

    def add_plan(self,plan:dict):
        """
        Adds a plan to the project, which defines datasets, label groups, and preprocessing steps.
        Each time this is called with a plan contaiing new datasources, the database will be updated accordingly.

        Parameters:
        ----------
        plan : dict
            A dictionary defining the project plan, typically loaded from an Excel sheet.
            Must include 'datasources' (list of dataset names) and 'lm_groups' (list of label groups).
        overwrite_global_properties : bool, optional
            Whether to overwrite existing global properties (default is True). Set it to False if you already computed dataset mean, std, and now only want to add datasources
        """
        dss = plan['datasources']
        dss= dss.split(",")
        datasources = [getattr(DS,g) for g in dss]
        self.add_data(datasources)
        self.set_lm_groups(plan['lm_groups'])
        self.maybe_store_projectwide_properties(overwrite=True)




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
    
    @property
    def has_folds(self):
        '''
        checks if folds have been created for this project
        '''
        
        ss = """SELECT fold FROM datasources"""
        result = self.sql_query(ss, True)
        cc =[a=='NULL' for a in result]
        all_bool= not all(cc)
        return all_bool



# %%
# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------

if __name__ == "__main__":
    from fran.utils.common import *

    P= Project(project_title="totalseg")
    # P.delete()
    P.create(mnemonic='totalseg')
    # P.add_data([DS.nodes,DS.nodesthick])
    P.add_data([DS.totalseg])

# %%
    conf = ConfigMaker(
        P, raytune=False, configuration_filename=None

    ).config

    plans= conf['plan4']
    plans = conf['plantmp']

    P.add_plan(plans)
# %%
    # P.maybe_store_projectwide_properties(overwrite=True)
# %%
    # P.set_lm_groups([['litq','litqsmall','drli','lits'],['lidc2']])
    # P.set_lm_groups()

# %%
    # P.add_data(ds5)
    # P.create_project([ds,ds2,ds3,ds4])
    len(P.raw_data_imgs)
    len(P)
# %%

    P.train_v
    if P.has_folds:
        P.get_train_val_files(0,conf['plan']['datasources'])
        aa = P.get_train_val_files(None,"nodes")
# %%
    P.imported_labels('lm_group2',Path("/s/fran_storage/predictions/totalseg/LITS-827/"),labelsets =[lr,ll])

# %%
    import sqlite3
    db_name = "/s/fran_storage/projects/litsmc/cases.db"
    db_name = "/s/fran_storage/projects/nodes/cases.db"
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    
# %%
    ss = """ALTER TABLE datasources RENAME COLUMN lm to lm"""
    ss = """DELETE FROM datasources WHERE case_id='lits_115'"""

    ss = """SELECT case_id FROM datasources WHERE fold IS NOT NULL"""


# %%
    ss_train = "SELECT img_symlink FROM datasources WHERE fold<>{} ".format(1)
    dss = conf['plan']['datasources']
    dss = ("nodesthick","nodes")


# %%
    ds = plans['datasources']
    fold=0
    ss_train = "SELECT img_symlink FROM datasources WHERE fold<>{}".format(fold)
    # ss_train = "SELECT img_symlink FROM datasources WHERE fold<>{} AND ds = 'nodes' ".format(fold)
    ss_val = "SELECT img_symlink FROM datasources WHERE fold={}".format(fold)
    if isinstance(ds,str) and "," not in ds:
        # Convert the list of datasources into a SQL-friendly string format
        # ds_filter = ds
        ss_train += " AND ds IN ({})".format(ds)
        ss_val += " AND ds IN ({})".format(ds)
    elif isinstance(ds,str):
        ss_train += " AND ds = '{}'".format(ds)
        ss_val += " AND ds = '{}'".format(ds)
        # ds = "('{}')".format(ds)
    # else:
        # Append the datasource filter to the SQL queries
        # ss_train += " AND ds IN ({})".format(ds_filter)
        # ss_val += " AND ds IN ({})".format(ds_filter)
    train_files,val_files = P.sql_query(ss_train,True),P.sql_query(ss_val,True)
    train_files =[Path(fn).name for fn in train_files]
    val_files =[Path(fn).name for fn in val_files]


    # ss_train = "SELECT img_symlink FROM datasources WHERE fold<>{} AND ds in  ('nodesthick', 'nodes') ".format(10)

# %%
    fold = 0
    ss_train = """
    SELECT img_symlink 
    FROM datasources 
    WHERE fold <> {} 
    AND LOWER(TRIM(ds)) IN ('nodesthick', 'nodes')
    """.format(fold)
# %%
    dss = None
    ss_train = "SELECT img_symlink FROM datasources WHERE fold<>{0} AND ds in  ('{1}')".format(10,dss)
    aa = cur.execute(ss_train)
    bb= pd.DataFrame(aa)
    bb
# %%
    conn.commit()
    aa = conn.execute(ss_train)


# %%

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

    fold = 0
    ds = DS.nodes
    ss_train = P.build_sql_query(fold, ds, is_validation=False)

    ss_val = P.build_sql_query(fold, ds, is_validation=True)
    query = ss_val

    result = P.sql_query(query, True)
    result = P.sql_query(ss, True)
    # Execute SQL queries
    train_files = P.fetch_files(ss_train)
    val_files = P.fetch_files(ss_val)

    ss = """SELECT NOT EXISTS (SELECT 1 FROM datasources WHERE fold IS NOT NULL) AS all_nulls"""
# %%
# %%

    strs = [P.vars_to_sql(ds.name,ds.alias, *pair, ds.test) for pair in ds.verified_pairs]
    with db_ops(P.db) as cur:
        cur.executemany("INSERT INTO datasources VALUES (?,?, ?,?,?,?,?,?,?)", strs)
# %%

    ds_dict =DS.nodes
    fldr = ds_dict['folder']
    test=False
    ds = Datasource(folder=fldr, name = ds_dict['ds'],alias= ds_dict['alias'],test=test)

    ss = "SELECT image FROM datasources WHERE ds='{}'".format(ds.name)
    with db_ops(P.db) as cur:
        res = cur.execute(ss)
        pa = res.fetchall()
    existing_images = list(il.chain.from_iterable(pa))
    existing_images = [Path(x) for x in existing_images]
    if len(existing_images) > 0:
        print(
            "Datasource {} exists already. Checking for new files in added folder".format(
                ds.name
            )
        )
        remaining_images_bool = [x not in existing_images for x in ds.images]
        ds.verified_pairs= list(il.compress(ds.verified_pairs, remaining_images_bool))
        if (ln := len(ds)) > 0:
            print("{} new files found. Adding to db.".format(ln))
        else:
            print("No new files to add from datasource {}".format(ds.name))
# %%
        datasources= [DS.drli_short]
        test=False
        test = [False] * len(datasources) if not test else listify(test)
        assert len(datasources) == len(
            test
        ), "Unequal lengths of datafolders and (bool) test status"
        for ds_dict, test in zip(datasources, test):
            fldr = ds_dict['folder']
            ds = Datasource(folder=fldr, name = ds_dict['ds'],alias= ds_dict['alias'],test=test)
            ds = P.filter_existing_images(ds)
            P.populate_tbl(ds)
        P.populate_raw_data_folder()
        P.register_datasources(datasources)


# %%
