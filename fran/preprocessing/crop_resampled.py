
import argparse
from os import maybe_makedirs
from fran.preprocessing.stage1_preprocessors import patch_generator_wrapper
from fran.preprocessing.datasetanalyzers import *

from fran.utils.fileio import load_dict
from fran.utils.helpers import *
# %%

def generate_patches(proj_defaults):
    ps = "Enter desired patch-size (e.g. '64,160,160')"
    patch_size=get_list_input(ps, str_to_list_int)
    
    fldr_name ="patches_{0}_{1}_{2}".format(*patch_size)
    output_folder= proj_defaults.stage2_folder/fldr_name/("volumes")
    maybe_makedirs(output_folder)
    stage0_bboxes_fn=proj_defaults.stage0_folder/("bboxes_info")
    stage0_bboxes = load_dict(stage0_bboxes_fn)
    args = [[output_folder,patch_size,inf] for inf in stage0_bboxes]
    multiprocess_multiarg(patch_generator_wrapper,args,debug=False)
    
# %%
    #generating bboxes from cropped files
    cropped_tnsrs_filenames = list(output_folder.glob("*pt"))
    arguments =[[x,proj_defaults] for x in cropped_tnsrs_filenames]
    res_cropped= multiprocess_multiarg(func=bboxes_function_version,arguments=arguments,num_processes=16,debug=False)
    save_dict(res_cropped,output_folder.parent/"bboxes_info")

if __name__ == "__main__":

    common_vars_filename=os.environ['FRAN_COMMON_PATHS']

    parser = argparse.ArgumentParser(description="Resampler")
    parser.add_argument('t',help='project title')
    parser.add_argument('n',type=int, help='num processes', default= 8)

    args = parser.parse_args()

    P = Project(project_title="lits"); proj_defaults= P
    output_folder = proj_defaults.stage2_folder
    spacings= load_dict( proj_defaults.resampled_dataset_properties_filename)['preprocessed_dataset_spacings']
    global_props =  load_dict(proj_defaults.raw_dataset_properties_filename)[-1]
    print("Pre-processed dataset spacings are: {}".format(spacings))

# %%

    patches_gen= input ("Proceed with patch generation(Y/y)?")
    if patches_gen.lower() == 'y': generate_patches(proj_defaults)
    else: print("Nothing to do.")

