# %
import os
import argparse
# from fran.inference.transforms import *
from fran.utils.common import *
from fran.transforms.spatialtransforms import *
from fran.managers.trainer import *
from fran.managers.tune import *
from fran.inference.inference_base import *
from fran.utils.imageviewers import *
import itertools as il

    # ImageMaskViewer([img_np,mask_np])


tot_gpus=2
n_lists = 4

ray.init(num_gpus=tot_gpus)
# %%
@ray.remote(num_gpus=tot_gpus/n_lists)
class EnsembleActor:
    def setup(self,proj_defaults,run_name_w,runs_ensemble ):
        device = os.environ["CUDA_VISIBLE_DEVICES"]
        self.En = EnsemblePredictor(proj_defaults,run_name_w,runs_ensemble,device=device)
    def process(self,fnames):
        for img_fn in fnames:
            img_fn = Path(img_fn)
            self.En.run(img_fn)
def slice_list(listi,start_end:list):
    # print(listi)
    return listi[start_end[0]:start_end[1]]
# %%
def main_old(args):


    mo_df = pd.read_csv(Path("/s/datasets_bkp/litq/complete_cases/cases_metadata.csv"))

    # runs_ensemble=["LITS-444","LITS-443","LITS-439","LITS-436","LITS-445"]
    runs_ensemble=["LITS-451","LITS-452","LITS-453"]
    # runs_ensemble=["LITS-265","LITS-255","LITS-270","LITS-271","LITS-272"]
    device = 0
    E = EnsemblePredictor(proj_defaults,run_name_w,runs_ensemble,device=device)
    
    for n in range(len(mo_df)):
        Path(mo_df.mask_filenames[n] )
        img_fn =Path(mo_df.image_filenames[n])
        E.run(img_fn)

def main(args):


    # runs_ensemble=["LITS-408","LITS-385","LITS-383","LITS-357"]

    runs_ensemble=["LITS-444","LITS-443","LITS-439","LITS-436","LITS-445"]
    # runs_ensemble=["LITS-265","LITS-255","LITS-270","LITS-271","LITS-272"]
    mo_df = pd.read_csv(Path("/s/datasets_bkp/litq/complete_cases/cases_metadata.csv"))
    fnames = list(mo_df.mask_filenames)
    fpl= int(len(mo_df)/n_lists)
    inds = [[fpl*x,fpl*(x+1)] for x in range(n_lists-1)]
    inds.append([fpl*(n_lists-1),None])

    chunks = list(il.starmap(slice_list,zip([fnames]*n_lists,inds)))
    # runs_ensemble=["LITS-408","LITS-385","LITS-383","LITS-357"]
    Es = []
    for x  in range(n_lists):
     Es.append(EnsembleActor.remote()) 
# %%
    for E, fs in zip(Es,chunks):
           E.setup.remote(proj_defaults, run_name_w,runs_ensemble)
           E.process(fs)
 
# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ensemble Predictor")


    common_paths_filename=os.environ['FRAN_COMMON_PATHS']
    P = Project(project_title="lits"); proj_defaults= P.proj_summary
    args = parser.parse_known_args()[0]
    args.debug = True
    #note: horseshoe kidney  Path('/s/datasets/raw_database/raw_data/kits21/images/kits21_00005.nii.gz')

    run_name_w= "LITS-276" # best trial

    main_old(args)


# %%
    # img = img.to('cuda')

# %%
#     mask_sitk = sitk.ReadImage(mask_fn)
#     mask_np = sitk.GetArrayFromImage(mask_sitk)
#     mask_np = mask_np.astype(np.uint8)
#
#     img_sitk = sitk.ReadImage(img_fn)
#     img_np = sitk.GetArrayFromImage(img_sitk)
#
#     mask_np = mask_np.astype(np.uint8)
# %%

#
#     mask_pt = ToTensor.encodes(mask_fn)
#     preds_avg= pred_mean(preds)
#     preds_voted = pred_voted(preds_int)
# # %%
#     n_classes = 3
#     scores=[]
#
#     for pred_pt in [*preds, *preds_int,preds_avg,preds_voted]:
#         pred_onehot,mask_onehot = [one_hot(x,classes=n_classes,axis=0).unsqueeze(0) for x in [pred_pt,mask_pt]]
#         aa = compute_dice(pred_onehot,mask_onehot, include_background=False)
#         print(aa)
#         scores.append(aa)
#
# # %%
#         n=0
#         ImageMaskViewer([img.detach()[0][0],preds[n][0][1]])
#
# # %%
#         s = sliding_window_inference(inputs = img_input,roi_size = p.patch_size,sw_batch_size =1,predictor=p.model)
# # %%
#
