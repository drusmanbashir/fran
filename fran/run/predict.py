from ray.util.multiprocessing import Pool as rPool
from fran.inference.scoring import compute_dice_fran
from utilz.helpers import *
import os

def case_processed_already(img_fn,output_folder):
    img_fn_no_ext = img_fn.name.split(".")[0]
    matching_output_files = [fn for fn in output_folder.glob("*") if img_fn_no_ext in str(fn)]
    if len(matching_output_files) == 0:     
        print("\n\n **********  {0} prediction already exists in folder {1}".format(img_fn, output_folder))
        return False
    else: return True

import argparse

from fran.inference.cascade import EndToEndPredictor

def run_prediction(E,proj_defaults,run_name_l,run_name_p,use_neptune,img_fn,gt_fn=None,overwrite=False,output_folder=None):
    # E = EndToEndPredictor(proj_defaults,run_name_l,run_name_p,use_neptune=use_neptune)
    if overwrite==False and case_processed_already(img_fn,E.output_folder): 
        return img_fn,None
    E.predict(img_fn=img_fn , save_localiser=True)
    if gt_fn:
        return img_fn, compute_dice_fran(gt_fn,E.pred_final,E.n_classes)
    else: return img_fn, None


def main (args):

    project_title = args.t
    P = Project(project_title=project_title); proj_defaults= P
    print("Project: {0}".format(project_title))

    run_name_l= args.l
    run_name_p  =args.p
    use_neptune = not args.no_neptune
    imgs_folder,masks_folder = Path(args.images_folder), Path(args.masks_folder)
    img_fnames = list(imgs_folder.glob("*"))
    n_processes = len(args.gpus)
    overwrite=args.overwrite
    if masks_folder: 
        img_mask_pairs = [[fn,masks_folder/(fn.name)] for fn in img_fnames]
    else:
        img_mask_pairs = [[fn,None] for fn in img_fnames]

    E = EndToEndPredictor(proj_defaults,run_name_l,run_name_p,use_neptune=use_neptune)
    args = [[E, proj_defaults,run_name_l,run_name_p,use_neptune,*img_mask_pair,overwrite,None] for img_mask_pair in img_mask_pairs]
    res = multiprocess_multiarg(func=run_prediction,arguments=args,num_processes=n_processes,io=True)
    tr()
    save_dict(res,"result")




if __name__ == "__main__":
    from fran.utils.common import *
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("-t", help="project title")
    parser.add_argument("-l", help="localiser run-name")
    parser.add_argument("-p", help="patch-predictor run-name")
    parser.add_argument("-i","--images-folder" , help="folder containing files on which to run inference",required=True)
    parser.add_argument("-m","--masks-folder" , help="folder containing files on which to run inference")
    parser.add_argument("-o","--overwrite" , action='store_true')
    parser.add_argument("--no-neptune",action ='store_true')

    parser.add_argument("--gpus", help="gpu labels (list of int). Default is 1 gpu", nargs='+', type=int)
# %%
    args = parser.parse_known_args()[0]
# %%
    args.t = 'lits'
    args.l = 'LITS-206'
    args.p = 'LITS-133'
    args.images_folder = "/s/fran_storage/datasets/raw_data/lits/fold_o_validation/images/"
    args.masks_folder = "/s/fran_storage/datasets/raw_data/lits/fold_o_validation/masks/"
    args.gpus = [0,1]
# %%

    main(args)
