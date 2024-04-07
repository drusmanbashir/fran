# %%
import argparse
import ast
import shutil
from fran.managers.project import DS

from label_analysis.totalseg import TotalSegmenterLabels

from fran.preprocessing.datasetanalyzers import *
from fran.preprocessing.fixed_spacing import ResampleDatasetniftiToTorch
from fran.preprocessing.globalproperties import GlobalProperties
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator
from fran.preprocessing.patch import PatchDataGenerator, PatchGenerator
from fran.utils.fileio import *
from fran.utils.helpers import *

common_vars_filename = os.environ["FRAN_COMMON_PATHS"]


@str_to_path(0)
def verify_dataset_integrity(folder:Path, debug=False,fix=False):
    '''
    folder has subfolders images and masks
    '''
    print("Verifying dataset integrity")
    subfolder = list(folder.glob("mask*"))[0]
    args = [[fn,fix] for fn in subfolder.glob("*")]
    res = multiprocess_multiarg(verify_img_label_match,args,debug=debug)
    errors = [item for item in res if re.search("mismatch", item[0],re.IGNORECASE)]
    if len(errors)>0:
        outname = folder/("errors.txt")
        print(f"Errors found saved in {outname}")
        save_list(errors,outname)
        res.insert(0,errors)
    else:
        print("All images and masks are verified for matching sizes and spacings.")
    return res

def user_input(inp:str, out=int):
    tmp = input(inp)
    try:
        tmp = ast.literal_eval(tmp)
        tmp = out(tmp)
    except:
        tmp = None
    return tmp


class PreprocessingManager():
    #dont use getattr
    def __init__(self, args):
        self.assimilate_args(args)
        P = Project(project_title=args.project_title); 
        self.project= P
        conf = ConfigMaker(
            P, raytune=False, configuration_filename=None, configuration_mnemonic='liver'
        ).config

        # args.overwrite=False
        plan = conf[self.plan]
        self.spacing = ast.literal_eval(plan['spacing'])

        self.Resampler = ResampleDatasetniftiToTorch(
                    project=self.project,
                    spacing=self.spacing,

                    device='cpu'
                )



        # 
        print("Project: {0}".format(self.project_title))


    def verify_dataset_integrity(self):
        verify_dataset_integrity(self.project.raw_data_folder,debug=self.debug,fix = not self.no_fix)

    def analyse_dataset(self):
            if self._analyse_dataset_questions() == True:
                self.GlobalP= GlobalProperties(self.project, bg_label=0,clip_range=self.clip_range)
                self.GlobalP.store_projectwide_properties()
                self.GlobalP.compute_std_mean_dataset(debug=self.debug)
                self.GlobalP.collate_lm_labels()

    def _analyse_dataset_questions(self):

        global_properties = load_dict(self.project.global_properties_filename)
        if not 'total_voxels' in global_properties.keys():
             return True
        else:
            reanalyse = input(
                "Dataset global properties already computed. Re-analyse dataset (Y/y)?"
            )
            if reanalyse.lower() == "y":
                return True

    def resample_dataset(self, generate_bboxes=True):
        self.get_resampling_configs()
        self.Resampler.create_dl()
        self.Resampler.process()
        if generate_bboxes==True:
            self.Resampler.generate_bboxes_from_masks_folder(
                    debug=self.debug, bg_label=0,num_processes=self.num_processes
                    )


    def generate_TSlabelboundeddataset(self,organ,imported_folder,keep_imported_labels=False,lm_group="lm_group1"):
        '''
        requires resampled folder to exist. Crops within this folder
        '''
        imported_folder=Path(imported_folder)
        
        TSL = TotalSegmenterLabels()
        if organ=="lungs":
            imported_labelsets = TSL.labels("lung", "right"), TSL.labels("lung", "left")
            remapping = TSL.create_remapping(imported_labelsets, [8, 9])
        self.L = LabelBoundedDataGenerator(
            project=self.project,
            expand_by=20,
            spacing=self.spacing,
            lm_group=lm_group,
            imported_folder=imported_folder,
            imported_labelsets=imported_labelsets,
            keep_imported_labels=keep_imported_labels,
            remapping=remapping,
        )

        self.L.setup()
        self.L.process()




    @ask_proceed("Generating low-res whole images to localise organ of interest")
    def generate_whole_images_dataset(self):
        if not hasattr(self, "spacing"):
            self.set_spacing()
        output_shape = ast.literal_eval(
            input(
                "Enter whole image matrix shape as list/tuple/number(e.g., [128,128,96]): "
            )
        )
        if isinstance(output_shape, (int, float)):
            output_shape = [
                output_shape,
            ] * 3
        self.WholeImageTM = WholeImageTensorMaker(
            self.project,
            source_spacing=self.spacing,
            output_size=output_shape,
            num_processes=self.num_processes,
        )
        arglist_imgs, arglist_masks= self.WholeImageTM.get_args_for_resizing()
        for arglist in [arglist_imgs,arglist_masks]: 
            res= multiprocess_multiarg(func=resize_and_save_tensors,arguments=arglist,num_processes=self.num_processes,debug=self.debug)
        print("Now call bboxes_from_masks_folder")
        generate_bboxes_from_masks_folder(self.WholeImageTM.output_folder_masks,0,self.debug,self.num_processes)

    def set_patches_config(self,spacing_ind=0,patch_overlap=.25,expand_by=20):

        spacing_config = self.resampling_configs[spacing_ind]
        self.spacing, self.fixed_spacing_folder , self.lbd_output_folder = spacing_config.values()
        self.lbd_output_folder= Path(self.lbd_output_folder)
        patches_output_folder = self.create_patches_output_folder(
            self.lbd_output_folder, self.patch_size
        )
        patches_config_fn = patches_output_folder / "patches_config.json"
        if patches_config_fn.exists()==True:
            print("Patches configs already exist. Loading from file")
            patches_config = load_dict(patches_config_fn)
        else:
            patches_config = {
                
                "patch_overlap": patch_overlap,
                "expand_by": expand_by,
            }
        return patches_config, patches_output_folder

    def generate_hires_patches_dataset(self,spacing_ind, patch_overlap=.25,expand_by=0,debug=False,overwrite=False,mode=None):
        patches_config , patches_output_folder= self.set_patches_config(spacing_ind,patch_overlap,expand_by)

        if mode is None:
            mode = self.mode
        PG = PatchDataGenerator(self.project,self.lbd_output_folder, self.patch_size,**patches_config,mode=mode)
        PG.create_patches(overwrite=overwrite,debug=debug)
        print("Generating boundingbox data")
        PG.generate_bboxes(debug=debug)

        resampled_dataset_properties_fn_org = self.lbd_output_folder / (
            "resampled_dataset_properties.json"
        )
        resampled_dataset_properties_fn_dest = (
            patches_output_folder.parent / resampled_dataset_properties_fn_org.name
        )
        if not resampled_dataset_properties_fn_dest.exists():
            shutil.copy(
                resampled_dataset_properties_fn_org,
                resampled_dataset_properties_fn_dest,
            )


    def get_resampling_configs(self):
        try:
            resampling_configs = load_dict(
                self.project.fixed_spacing_folder / ("resampling_configs")
            )
            print(
                "Based on earlier pre-processing, following data-spacing configs are available:"
            )
            for indx, config in enumerate(resampling_configs):
                print("Index: {0}, config {1}".format(indx, config["spacing"]))
            return resampling_configs
        except:
            print("No resampling configs exist.")
            return []

    def create_patches_output_folder(self, fixed_spacing_folder, patch_size):
        
        patches_fldr_name = "dim_{0}_{1}_{2}".format(*patch_size)
        output_folder = (
            self.project.patches_folder / fixed_spacing_folder.name / patches_fldr_name
        )
        # maybe_makedirs(output_folder)
        return output_folder

    def assimilate_args(self, args):
        for key, value in vars(args).items():
            setattr(self, key, value)

    def maybe_change_default_spacing(self, vals):
        def _accept_defaults():
            print("Accepting defaults")

        try:
            if isinstance(vals, str):
                vals = ast.literal_eval(vals)
            if all([isinstance(vals, (list, tuple)), len(vals) == 3]):
                self.Resampler.spacing = vals
            elif isinstance(vals, (int, float)):
                vals = [
                    vals,
                ] * 3
                self.Resampler.spacing = vals
            else:
                _accept_defaults()
        except:
            _accept_defaults()

    @property
    def resampling_configs(self):
        return self.get_resampling_configs()



def do_resampling(R, args):
    dim0 = input("Change dim0 to (press enter to leave unchanged)")
    dim1 = input("Change dim2 to (press enter to leave unchanged)")
    dim2 = input("Change dim3 to (press enter to leave unchanged)")
    spacing = [
        float(a) if len(a) > 0 else b for a, b in zip([dim0, dim1, dim2], R.spacing)
    ]
    R.spacing = spacing
    R.resample_cases(debug=False, overwrite=args.overwrite, multiprocess=True)


def do_low_res(proj_defaults):
    low_res_shape = get_list_input(
        text="Enter low-res image shape (e.g., '128,128,128')", fnc=str_to_list_int
    )
    stage0_files = list(Path(proj_defaults.stage0_folder / "volumes").glob("*.pt"))

    stage1_subfolder = (
        proj_defaults.stage1_folder
        / str(low_res_shape).strip("[]").replace(", ", "_")
        / "volumes"
    )
    maybe_makedirs(stage1_subfolder)

    args = [[fn, stage1_subfolder, low_res_shape, False] for fn in stage0_files]
    multiprocess_multiarg(resample_img_mask_tensors, args, debug=False)



# %%

if __name__ == "__main__":
    from fran.utils.common import *
    parser = argparse.ArgumentParser(description="Resampler")

    parser.add_argument("-t", help="project title", dest="project_title")
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        help="number of parallel processes",
        default=8,
    )
    parser.add_argument("-e", "--enforce-isotropy", action="store_true")
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("-c", "--clip-centre", action='store_true', help="Clip and centre data now or during training?")
    parser.add_argument("-r", "--clip-range", nargs='+', help="Give clip range to compute dataset std and mean")
    parser.add_argument("-m", "--mode", default= "fgbg", help = "Mode of Patch generator, 'fg' or 'fgbg'")
    parser.add_argument("-p", "--patch-size", nargs="+", default=[192,192,128] ,help="e.g., [192,192,128]if you want a high res patch-based dataset")
    parser.add_argument("-s", "--spacing", nargs='+', help="Give clip range to compute dataset std and mean")
    parser.add_argument("-i", "--imported-folder")
    parser.add_argument("-po", "--patch-overlap" ,help="Generating patches will overlying by this fraction range [0,.9). Default is 0.25 ", default=0.25, type=float)
    parser.add_argument("-hp", "--half_precision" ,action="store_true")
    parser.add_argument("-nf", "--no-fix", action="store_false",help="By default if img/mask sitk arrays mismatch in direction, orientation or spacing, FRAN tries to align them. Set this flag to disable")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument(
        "-np", "--no-pbar", dest="pbar", action="store_false", help="Switch off progress bar"
    )



    args = parser.parse_known_args()[0]
# %%
    args.project_title = "litsmc"

    # args.num_processes = 1
    args.debug=True
    # args.clip_range=[-100,200]

    args.plan = "plan1"
# %%

    P= Project(project_title=args.project_title)

# %%
    conf = ConfigMaker(
        P, raytune=False, configuration_filename=None, configuration_mnemonic='liver'
    ).config
# %%

    plans = conf['plan1']
    dss = plans['datasources']
    dss= dss.split(",")
    datasources = [getattr(DS,g) for g in dss]
    P.create_project(datasources)
    P.set_lm_groups(plans['lm_groups'])
    P.maybe_store_projectwide_properties(overwrite=False)

# %%
    I = PreprocessingManager(args)
    # I.spacing = 
# %%
    I.resample_dataset()
    # I.generate_TSlabelboundeddataset("lungs","/s/fran_storage/predictions/totalseg/LITS-827")
    I.generate_hires_patches_dataset(spacing_ind=1,debug=True,overwrite=True)
# %%
    
    fixed_spacing_folder =Path(I.resampling_configs[1]['resampling_output_folder'])
    PG = PatchDataGenerator(I.project,fixed_spacing_folder, I.patch_size,**patches_config,mode=I.mode)



# %%
    overwrite=True
    debug=True
    PG.create_patches(overwrite=overwrite,debug=debug)

# %%
    bb= PG.fixed_sp_bboxes[0]
    args =             [
                PG.dataset_properties,
                PG.output_folder,
                PG.patch_size,
                bb,
                patch_overlap,
                PG.expand_by,
                PG.mode
            ]
# %%
    P = PatchGenerator(
        PG.dataset_properties, PG.output_folder, PG.patch_size, bb, patch_overlap, expand_by, 'fg'
    )
# %%
    P.create_patches_from_all_bboxes()

# %%
    PG.generate_bboxes(debug=debug)
# %%
    # I.verify_dataset_integrity()

    # I.analyse_dataset()
    I.resample_dataset(generate_bboxes=True)

# %%
    I.generate_whole_images_dataset()

# %%
    im1 = "/home/ub/tmp/imgs/litq_72b_20170224_old.pt"
    im2 = "/s/fran_storage/datasets/preprocessed/fixed_spacing/lilun3/spc_074_074_160/images/litq_72b_20170224.pt"
    im1 = torch.load(im1)
    im2 = torch.load(im2)
    ImageMaskViewer([im1,im2], data_types=['image','image'])
# %%

    spacing_ind = 0
    patch_overlap=.25
    expand_by = 20
    patches_config , patches_output_folder= I.set_patches_config(spacing_ind,patch_overlap,expand_by)
    PG = PatchDataGenerator(I.project,I.fixed_spacing_folder, I.patch_size,**patches_config)
    print("Generating boundingbox data")
    PG.generate_bboxes(debug=debug)
# %%

    patch_overlap=0.25
    expand_by = 20
    patches_config , patches_output_folder= I.set_patches_config(0,patch_overlap,expand_by)
# %%

    resampling_configs = I.get_resampling_configs()
    spacing_config = resampling_configs[spacing_ind]


    value= spacing_config['spacing']
# %%
    folder_name_from_list(
            prefix="spc",
            parent_folder=I.project.lbd_folder,
            values_list=value,
        )
# %%
    spacing_ind=1
    patch_overlap=.25
    expand_by=0
    patches_config , patches_output_folder= I.set_patches_config(spacing_ind,patch_overlap,expand_by)

    if mode is None:
        mode = I.mode
        PG = PatchDataGenerator(I.project,I.lbd_output_folder, I.patch_size,**patches_config,mode=mode)

# ii = "/s/fran_storage/datasets/preprocessed/fixed_spacing/lax/spc_080_080_150/images/lits_5.pt"
# torch.load(ii).dtype
# %%

        I.get_resampling_configs()
        I.Resampler.create_dl()
        I.Resampler.process()
# %%
