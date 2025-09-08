# %%
import logging
from fastcore.all import is_close, test_eq
from fastcore.basics import  GetAttr, store_attr
import numpy as np
from fran.preprocessing.helpers import get_label_stats, import_h5py, percentile_range_to_str
from fran.transforms.imageio import LoadSITKd
from fran.transforms.totensor import ToTensorT
from utilz.helpers import *

# sys.path += ["/home/ub/Dropbox/code/fran"]
from utilz.helpers import *
from utilz.fileio import *

from utilz.imageviewers import ImageMaskViewer
from utilz.string import  headline, info_from_filename

class SingleCaseAnalyzer:
    """
    Wraps LoadSITKd to load a single case

    """

    def __init__(
            self,  case_files_tuple, percentile_range:list, bg_label=0
    ):
        """
        param: case_files_tuple are two nii_gz files (img,mask)
        """
        assert isinstance(case_files_tuple, list) or isinstance(
            case_files_tuple, tuple
        ), "case_files_tuple must be either a list or a tuple"
        case_files_tuple = [Path(fn) for fn in case_files_tuple]
        case_props =  info_from_filename(case_files_tuple[0].name)
        self.case_id = "_".join([case_props['proj_title'],case_props["case_id"]])
        store_attr("case_files_tuple,bg_label,percentile_range")

    def load_case(self):
        L= LoadSITKd(keys=['image','mask'], image_only=True,ensure_channel_first=False,simple_keys=True,lm_key='mask')
        dici = {'image':self.case_files_tuple[0],'mask':self.case_files_tuple[1]}
        dd = L(dici)
        # self.properties['itk_spacing']= self.img.meta['']
        self.img,self.mask = dd['image'],dd['mask']
        self.set_properties()

    @property   
    def properties(self):
        return self._properties
        
    def set_properties(self):
        self._properties = {}
        excluded = 'filename_or_obj','original_channel_dim'
        for k in self.img.meta.keys():
            if k not in excluded:
                v1 = self.img.meta[k]
                v2 = self.mask.meta[k]
                if isinstance(v1,np.ndarray):
                    assert is_close(v1,v2,eps=1e-4),"Metadata mismatch for key: {0} . Filename: {1}".format(k,self.mask.meta['filename_or_obj'])
                elif isinstance(v1,str):
                    test_eq(v1,v2)
                self._properties[k] = v1

        self._properties['img_fname'] = self.img.meta['filename_or_obj']
        self._properties['lm_fname'] = self.mask.meta['filename_or_obj']
        self._properties['labels'] = self.mask.meta['labels']
        self._properties["case_id"]= self.case_id



    def get_bbox_only_voxels(self):
        return self.img[self.mask != self.bg_label]


    @property
    def case_id(self):
        return self._case_id
    @case_id.setter
    def case_id(self,value): self._case_id = value


class MultiCaseAnalyzer(GetAttr):
    _default = "project"

    def __init__(self, project,bg_label=0) -> None:
        store_attr()
        self.project_title = project.project_title
        self.h5f_fname = project.bboxes_voxels_info_filename
        self.filter_unprocessed_cases()


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
        ss = "SELECT ds, case_id, lm_symlink, img_symlink FROM datasources" 
        res= self.project.sql_query(ss)
        all_cases= []
        for r in res:
            case_ = {"case_id":r[0]+"_"+r[1], "lm_symlink":r[2] , "img_symlink":r[3]}
            all_cases.append(case_)
        all_case_ids = set([c['case_id'] for c  in all_cases])
        new_cases = all_case_ids.difference(prev_processed_cases)
        headline("Found {0} new cases\nCases already processed in a previous session: {1}".format(len(new_cases), len(prev_processed_cases)))
        assert (l:=len(new_cases)) == (l2:=(len(all_case_ids)-len(prev_processed_cases))), "Difference in number of new cases"
        if len(new_cases) == 0: 
            print("No new cases found.")
            self.new_cases = []
        else:
            self.new_cases = [[c['img_symlink'],c['lm_symlink']] for c in all_cases if c['case_id'] in new_cases]

    def process_new_cases(
        self, return_voxels=True, num_processes=8, multiprocess=True, debug=False
    ):
        """
        Stage 1: derives datase properties especially intensity_fg
        if return_voxels == True, returns voxels to be stored inside the h5f file
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

    def dump_to_h5f(self):
        h5py = import_h5py()

        if self.h5f_fname.exists():
            mode= 'a'
        else: mode = 'w'
        with h5py.File(self.h5f_fname, mode) as h5f: 
            for output in self.outputs:
                try:
                    h5f.create_dataset(output["case"]["case_id"], data=output["voxels"])
                except ValueError:
                    print("Case id {} already exists in h5f file. Skipping".format(output['case']['case_id']))

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


def case_analyzer_wrapper(
    case_files_tuple,
    bg_label,
    get_voxels=True,
    percentile_range=[0.5, 99.5],
):
    S = SingleCaseAnalyzer(
        case_files_tuple=case_files_tuple,
        bg_label=bg_label,
        percentile_range=percentile_range,
    )
    S.load_case()
    case_ = dict()
    case_["case_id"] = S.case_id
    voxels=None
    if get_voxels == True:
        voxels = S.get_bbox_only_voxels().float()
        S.properties['numel_fg']= voxels.numel()
        if voxels.numel() == 0:
            S.properties["mean_fg"] = np.nan
            S.properties["min_fg"] = np.nan
            S.properties["max_fg"] = np.nan
            S.properties["std_fg"] = np.nan
            S.properties[percentile_range_to_str(percentile_range)] = np.nan
        else:
            S.properties["mean_fg"] = int(voxels.mean())
            S.properties["min_fg"] = int(voxels.min())
            S.properties["max_fg"] = int(voxels.max())
            S.properties["std_fg"] = int(voxels.std())
            S.properties[percentile_range_to_str(percentile_range)] = np.percentile(
            voxels, percentile_range
        )
    case_["properties"] = S.properties
    output = {"case": case_, "voxels": voxels}
    return output


# %%
if __name__ == "__main__":
    
    from fran.utils.common import *
    from fran.managers import Project
    P2 = Project(project_title="totalseg")
    fn = "/s/fran_storage/datasets/preprocessed/fixed_spacings/litsmc/resampling_configs.pkl"
    fn = Path(fn)
    dd = load_dict(fn)
    bboxes_info= load_dict("/s/fran_storage/datasets/preprocessed/fixed_spacings/litsmc/spc_080_080_150/bboxes_info_bk.pkl")

    existing_cases = set([b['case_id'] for b in bboxes_info])

    ss = "SELECT ds, case_id, label_symlink FROM datasources" 
    raw_info = load_dict(P2.raw_dataset_properties_filename)
# %%


    return_voxels=True
    num_processes=8
    multiprocess=True
    debug=True
    args_list = [
            [case_tuple,  ds.bg_label, return_voxels]
            for case_tuple in ds.new_cases
        ]

# %%
    case_tuple = ds.new_cases[0]
    ds.outputs = multiprocess_multiarg(
        func=case_analyzer_wrapper,
        arguments=args_list,
        num_processes=num_processes,
        multiprocess=multiprocess,
        debug=debug,
        logname = "dd.log"
        )
# %%
    case_analyzer_wrapper(case_files_tuple = case_tuple, bg_label=0)
# %%
#SECTION:-------------------- TS


# %%
    cases = []
    res= P.sql_query(ss)
    for dad in res:
        case_ = {"case_id":dad[0]+"_"+dad[1], "label_symlink":dad[2]}
        cases.append(case_)
# %%
    cases = set(cases)
    new_cases = cases.difference(existing_cases)
    assert (l:=len(new_cases)) == (l2:=(len(cases)-len(existing_cases))), "Difference in number of new cases"
    label_fns = [c['label_symlink'] for c in cases if c['case_id'] in new_cases]
# %%
    fn = Path(label_fns[0])
    aa = bboxes_function_version(fn, 0)
    fn2 = fn.str_replace("lms","images")
    img = torch.load(fn2)
    mask=torch.load(fn)
    # ImageMaskViewer([img,mask])
# %% [markdown]
    b = bboxes_function_version(fn,proj_defaults)
# %%
    A = BBoxesFromMask(
            fn,0
        )
    aa = A()
# %%
    P = Project("lidc")
    P.create("lungs")
    M = MultiCaseAnalyzer(P)
    M.process_new_cases(debug=True, num_processes=2, multiprocess=True)
    M.dump_to_h5f()
    M.store_raw_dataset_properties()

    dd = load_dict(P.raw_dataset_properties_filename)

# %%
    bb = aa['bbox_stats'][1]
    bbox = bb['bounding_boxes'][1]
    img2 , mask2  = img[bbox],mask[bbox]

    img2,mask2 = img2.permute(2,1,0), mask2.permute(2,1,0)
    ImageMaskViewer([img2,mask2])
# %%
    bboxes_all = []
    label_all_fg = A.mask.copy()
    label_all_fg[label_all_fg > 1] = 1
    for label,label_info in A.label_settings.items():
            label = int(label)
            if label in A.mask:
                stats = {"label": label}
                stats.update(
                    get_label_stats(
                        A.mask,
                        label,
                        k_largest=label_info["k_largest"],
                        dusting_threshold=int(
                            label_info["dusting_threshold"]
                            * A.dusting_threshold_factor
                        ),
                    )
                )
                bboxes_all.append(stats)
    stats = {"label": "all_fg"}
    stats.update(
            get_label_stats(
                label_all_fg,
                1,
                1,
                dusting_threshold=3000 * A.dusting_threshold_factor,
            )
        )
    bboxes_all.append(stats)
    A._bboxes_info["bbox_stats"] = bboxes_all
# %%

    P = Project(project_title="litsmc"); proj_defaults= P
    M = GlobalProperties(proj_defaults, bg_label=0)

# %%
    debug_ = True
    M.get_nii_bbox_properties(
        num_processes=1, debug=debug_, overwrite=True, multiprocess=True
    )

# %%

    M.store_projectwide_properties()
    debug=False
    M.compute_std_mean_dataset(num_processes=1, debug=debug )
# %%
    M.global_properties = load_dict(M.global_properties_outfilename)
    percentile_label, intensity_percentile_range=  get_intensity_range(M.global_properties)
    clip_range=[-5,200]
# %%
    img_fnames = [case_[0] for case_ in M.list_of_raw_cases]
    img_fname = img_fnames[0]
# %%
    img = ToTensorT()(img_fname)
    if clip_range is not None:
        img = torch.clip(img,min=clip_range[0],max=clip_range[1])
    var = (img - dataset_mean) ** 2
 
# %%
    args = [[fname, M.dataset_mean, M.clip_range] for fname in img_fnames]
    print(
        "Computing std from all nifti files, using global mean computed above (clipped to {})".format(
            M.clip_range
        )
    )

    M.compute_std_mean_dataset(debug=True)
# %% [markdown]
    # # Resample nifti to Torch
# %% [markdown]
    # ### KITS19

# %%
    R = ResampleDatasetniftiToTorch(
        proj_defaults, minimum_final_spacing=0.0, enforce_isotropy=False
    )
# %%
    R.spacings[-1] = 1.5
# %%

    R.resample_cases(debug=False, overwrite=True, multiprocess=False)

# %%
    R.resample_cases(debug=False, num_processes=8, multiprocess=True, overwrite=True)

# %%
# %%

    res = multiprocess_multiarg(bboxes_function_version, arguments, 16, debug=False)

# %%

    stats_outfilename_kits21 = proj_defaults.stage0_folder / "bboxes_info"
    save_dict(res, stats_outfilename_kits21)

# %%
    # # Getting bbox properties from preprocessed images
    # ### KITS21 cropped nifti files bboxes
    M.std = torch.sqrt(std_num.sum() / M.total_voxels)

# %%
    P = Project(project_title="lits"); proj_defaults= P
    masks_folder_nii = proj_defaults.stage1_folder / ("cropped/images_nii/masks")
    masks_filenames = get_fileslist_from_path(masks_folder_nii, ext=".nii.gz")
    arguments = [[x, proj_defaults] for x in masks_filenames]

# %%

    res = multiprocess_multiarg(bboxes_function_version, arguments, 48, debug=False)
# %%

    import collections
    od = collections.OrderedDict(sorted(gp.items()))
# %%

# %%
    fn = "hhm_jack.nii.gz"
    load_image(fn)
    get_extension(fn)

# %%
    stats_outfilename_kits21 = (
        proj_defaults.stage1_folder / ("cropped/images_nii/masks")
    ).parent / ("bboxes_info")
    save_dict(res, stats_outfilename_kits21)

# %%
    overwrite = True
    num_processes = 26
    multiprocess = True
    debug=True

# %%

    P = Project(project_title="litsmc"); proj_defaults= P
    logname = proj_defaults.log_folder / ("log.txt")
    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
# %%
    M.compute_std_mean_dataset(debug=False)
    M.case_properties = []
    if overwrite == True or not M.h5f_fname.exists():
        get_voxels = True
        print("Voxels inside bbox will be dumped into: {}".format(M.h5f_fname))
    else:
        get_voxels = False
        print("Voxels file: {0} exists".format(M.h5f_fname))
    args_list = [
        [case_tuple,  M.bg_label, get_voxels]
        for case_tuple in M.list_of_raw_cases
    ]
# %%
    M.outputs = multiprocess_multiarg(
        func=case_analyzer_wrapper,
        arguments=args_list,
        num_processes=num_processes,
        multiprocess=multiprocess,
        debug=debug,
        logname = logname
    )
# %%
    project_title = "litsmc"
    bg_label = 0
    c1 = Path("/s/xnat_shadow/lidctmp/images/lidc2_0069.nii.gz")
    c2 = Path("/s/xnat_shadow/lidctmp/masks/lidc2_0069.nii.gz")
    tup = [c1, c2]


    S = SingleCaseAnalyzer(
        project_title=project_title,
        case_files_tuple=tup,
        bg_label=bg_label,
        percentile_range=[.5, 99.5],
    )

    S.load_case()
    S.properties
# %%
    L= LoadSITKd(keys=['image','mask'], image_only=True,ensure_channel_first=False,simple_keys=True)
    dici = {'image':c1,'mask':c2}
    dd = L(dici)

# %%
    S.load_case()
    case_ = dict()
    case_["case_id"] = S.case_id
    if get_voxels == True:
        voxels = S.get_bbox_only_voxels()
        S.properties["mean_fg"] = int(voxels.mean())
        S.properties["min_fg"] = int(voxels.min())
        S.properties["max_fg"] = int(voxels.max())
        S.properties["std_fg"] = int(voxels.std())
        S.properties[percentile_range_to_str(percentile_range)] = np.percentile(
            voxels, percentile_range
        )
    case_["properties"] = S.properties
    output = {"case": case_, "voxels": voxels}


    case_props =  info_from_filename(fn.name)
    case_id = "_".join([case_props['proj_title'],case_props["case_id"]])
    print(case_id)
# %%


    db_name = "/s/fran_storage/projects/lilun/cases.db"
    conn = sqlite3.connect(db_name)
    
    ss = """ALTER TABLE datasources RENAME COLUMN mask_symlink to lm_symlink"""
    conn.execute(ss)
# %%
