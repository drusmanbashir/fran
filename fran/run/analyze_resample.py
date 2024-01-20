# %%
import argparse
import ast
import shutil
from fran.preprocessing.globalproperties import GlobalProperties
from fran.preprocessing.stage1_preprocessors import *
from fran.preprocessing.datasetanalyzers import *
from fran.preprocessing.stage0_preprocessors import ResampleDatasetniftiToTorch, generate_bboxes_from_masks_folder, verify_dataset_integrity
from fran.utils.helpers import *
from fran.utils.fileio import *


common_vars_filename = os.environ["FRAN_COMMON_PATHS"]

def user_input(inp:str, out=int):
    tmp = input(inp)
    try:
        tmp = ast.literal_eval(tmp)
        tmp = out(tmp)
    except:
        tmp = None
    return tmp


class InteractiveAnalyserResampler:
    def __init__(self, args):
        self.assimilate_args(args)

        P = Project(project_title=args.project_title); 
        self.proj_defaults= P
        # 
        print("Project: {0}".format(self.project_title))


    @ask_proceed("Verify dataset integry? ..Recommended if this is your first resampling run.")
    def verify_dataset_integrity(self):
        verify_dataset_integrity(self.proj_defaults.raw_data_folder,debug=self.debug,fix = not self.no_fix)


    def analyse_dataset(self):
            self.MultiAnalyser = MultiCaseAnalyzer(self.proj_defaults,bg_label=0)
            if len(self.MultiAnalyser.new_cases)>0:
                self.MultiAnalyser.process_new_cases(
                    num_processes=self.num_processes,
                    debug=self.debug,
                    multiprocess=True,
                )
                self.MultiAnalyser.dump_to_h5f()
                self.MultiAnalyser.store_raw_dataset_properties()
            if self._analyse_dataset_questions() == True:
                self.GlobalP= GlobalProperties(self.proj_defaults, bg_label=0)
                self.GlobalP.store_projectwide_properties()
                self.GlobalP.compute_std_mean_dataset()

    def _analyse_dataset_questions(self):
        do_analysis = not self.proj_defaults.global_properties_filename.exists()
        if do_analysis==True: return True
        else:
            reanalyse = input(
                "Dataset global properties file exists already. Re-analyse dataset (Y/y)?"
            )
            if reanalyse.lower() == "y":
                return True

    def resample_dataset(self):
        @ask_proceed("Resample dataset?")
        def _inner():
            self.Resampler = ResampleDatasetniftiToTorch(
                self.proj_defaults,
                minimum_final_spacing=0.5,
                enforce_isotropy=self.enforce_isotropy,
                half_precision=self.half_precision,
                clip_centre=self.clip_centre
            )


            vals = input("Press enter to accept defaults or a list/float for new values: ")
            self.maybe_change_default_spacings(vals)
            self.spacings = self.Resampler.spacings
            self.Resampler.resample_cases(
                multiprocess=True,
                num_processes=self.num_processes,
                overwrite=self.overwrite,
                debug=self.debug,
            )
            self.Resampler.generate_bboxes_from_masks_folder(
                debug=self.debug, bg_label=0,num_processes=self.num_processes
            )

        self.get_resampling_configs()
        _inner()
    @ask_proceed("Generating low-res whole images to localise organ of interest")
    def generate_whole_images_dataset(self):
        if not hasattr(self, "spacings"):
            self.set_spacings()
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
            self.proj_defaults,
            source_spacings=self.spacings,
            output_size=output_shape,
            num_processes=self.num_processes,
        )
        arglist_imgs, arglist_masks= self.WholeImageTM.get_args_for_resizing()
        for arglist in [arglist_imgs,arglist_masks]: 
            res= multiprocess_multiarg(func=resize_and_save_tensors,arguments=arglist,num_processes=self.num_processes,debug=self.debug)
        print("Now call bboxes_from_masks_folder")
        generate_bboxes_from_masks_folder(self.WholeImageTM.output_folder_masks,0,self.debug,self.num_processes)

    @ask_proceed("Generating hi-res patches. I recommend adding extra size to generated patches. The extra voxels will benefit affine transformation and will be cropped out before feeding the data to the NN")
    def generate_hires_patches_dataset(self,debug=False,overwrite=False):
        self.set_patch_size()
        self.set_spacings()
        patches_output_folder = self.create_patches_output_folder(
            self.fixed_sp_folder, self.patch_size
        )

        patches_config = self.get_patches_config(patches_output_folder / "patches_config.json")

        PG = PatchGeneratorDataset(self.proj_defaults,self.fixed_sp_folder, self.patch_size,**patches_config)
        PG.create_patches(overwrite=overwrite,debug=debug)
        print("Generating boundingbox data")
        PG.generate_bboxes(debug=debug)

        resampled_dataset_properties_fn_org = self.fixed_sp_folder / (
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


    def get_patches_config(self, patches_config_fn):
        if patches_config_fn.exists()==True:
            print("Patches configs already exist. Loading from file")
            patches_config = load_dict(patches_config_fn)
        else:
            expand_by = user_input(
                "Optionally add surrounding anatomy with target organ? Enter int (e.g., 10 for 10mm): "
            )
            patch_overlap = user_input(
                "Select patch overlap factor: [0,0.9) (current default: {})".format(
                    self.patch_overlap
                ),
                float,
            )
            patches_config = {
                "patch_overlap": patch_overlap,
                "expand_by": expand_by,
            }
        return patches_config

    def get_resampling_configs(self):
        try:
            resampling_configs = load_dict(
                self.proj_defaults.fixed_spacings_folder / ("resampling_configs")
            )
            print(
                "Based on earlier pre-processing, following data-spacing configs are available:"
            )
            for indx, config in enumerate(resampling_configs):
                print("Index: {0}, config {1}".format(indx, config["spacings"]))
            return resampling_configs
        except:
            print("No resampling configs exist.")
            return []


    def set_spacings(self):
        resampling_configs = self.get_resampling_configs()
        while True:
            try:
                indx = ast.literal_eval(
                    input("Enter valid index for desired spacing spec: ")
                )
                spacings_config = resampling_configs[indx]
            except (ValueError, IndexError):
                continue
            else:
                break
        self.spacings, self.fixed_sp_folder = spacings_config.values()

    def set_patch_size(self):
        patch_size = ast.literal_eval(
            input("What patch size? Enter list or int (e.g., [160,160,64]): ")
        )
        if isinstance(patch_size, int):
            patch_size = [
                patch_size,
            ] * 3
        print("Patch size set to: {}".format(patch_size))
        self.patch_size = patch_size

    def create_patches_output_folder(self, fixed_sp_folder, patch_size):
        patches_fldr_name = "dim_{0}_{1}_{2}".format(*patch_size)
        output_folder = (
            self.proj_defaults.patches_folder / fixed_sp_folder.name / patches_fldr_name
        )
        # maybe_makedirs(output_folder)
        return output_folder

    def assimilate_args(self, args):
        for key, value in vars(args).items():
            setattr(self, key, value)

    def maybe_change_default_spacings(self, vals):
        def _accept_defaults():
            print("Accepting defaults")

        try:
            vals = ast.literal_eval(vals)
            if all([isinstance(vals, (list, tuple)), len(vals) == 3]):
                self.Resampler.spacings = vals
            elif isinstance(vals, (int, float)):
                vals = [
                    vals,
                ] * 3
                self.Resampler.spacings = vals
            else:
                _accept_defaults()
        except:
            _accept_defaults()


def do_resampling(R, args):
    dim0 = input("Change dim0 to (press enter to leave unchanged)")
    dim1 = input("Change dim2 to (press enter to leave unchanged)")
    dim2 = input("Change dim3 to (press enter to leave unchanged)")
    spacings = [
        float(a) if len(a) > 0 else b for a, b in zip([dim0, dim1, dim2], R.spacings)
    ]
    R.spacings = spacings
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
    parser.add_argument("-po", "--patch-overlap" ,help="Generating patches will overlying by this fraction range [0,.9). Default is 0.25 ", default=0.25, type=float)
    parser.add_argument("-hp", "--half_precision" ,action="store_true")
    parser.add_argument("-nf", "--no-fix", action="store_false",help="By default if img/mask sitk arrays mismatch in direction, orientation or spacings, FRAN tries to align them. Set this flag to disable")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument(
        "-np", "--no-pbar", dest="pbar", action="store_false", help="Switch off progress bar"
    )


    args = parser.parse_known_args()[0]
# %%
    # args.project_title = "litsmc"
    # args.num_processes = 1
    # args.debug=False
    # args.overwrite=False
    I = InteractiveAnalyserResampler(args)
# %%
    I.verify_dataset_integrity()

    I.analyse_dataset()
    I.resample_dataset()

    I.generate_whole_images_dataset()
    I.generate_hires_patches_dataset(debug=True)

# %%

# ii = "/s/fran_storage/datasets/preprocessed/fixed_spacings/lax/spc_080_080_150/images/lits_5.pt"
# torch.load(ii).dtype
# %%
