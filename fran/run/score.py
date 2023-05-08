
# %%
from fran.inference.scoring import compute_dice_fran
from fran.run.predict import case_processed_already
from fran.utils.helpers import *
import argparse

from fran.inference.inference_base import EndToEndPredictor

def run_prediction(proj_defaults,run_name_l,run_name_p,use_neptune,img_fn,gt_fn,overwrite):
    E = EndToEndPredictor(proj_defaults,run_name_l,run_name_p,use_neptune=use_neptune)
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

    args.p
    masks_folder = Path(args.masks_folder)
    mask_fnames = list(masks_folder.glob("*"))
    predictions_folder ="/s/fran_storage/predictions/lits/segmentations_LITS-206/"
    predictions_folder = Path(predictions_folder)
    prediction_fnames = list(predictions_folder.glob("*"))
    pred_gt_pairs = []
    for fn in mask_fnames:
        gt =[ f for f in prediction_fnames if fn.name in str(f)][0]
        pred_gt_pairs. append([fn,gt])
    res= []
    n_classes = 3
    n_processes = args.n_processes
    m_args = [[*fpair, n_classes] for fpair in pred_gt_pairs]
    res = multiprocess_multiarg(func=compute_dice_fran,arguments=m_args,num_processes=n_processes,debug=True)
    save_dict(res,"result")
# %%




if __name__ == "__main__":
    from fran.utils.common import *
    parser = argparse.ArgumentParser(description="Trainer")
    parser.add_argument("-t", help="project title")
    parser.add_argument("-p", help="patch-predictor run-name")
    parser.add_argument("-m","--masks-folder" , help="folder containing ground-truth labels")
    parser.add_argument("--n-processes",default = 8)
# %%
    args = parser.parse_known_args()[0]
# %%
    args.t = 'lits'
    args.p = 'LITS-133'
    args.images_folder = "/s/fran_storage/datasets/raw_data/lits/fold_o_validation/images/"
    args.masks_folder = "/s/fran_storage/datasets/raw_data/lits/fold_o_validation/masks/"
# %%

    main(args)
