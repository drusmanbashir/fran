# %
import os
# from fran.inference.transforms import *
from fran.inference.scoring import compute_dice_fran
from monai.engines.evaluator import EnsembleEvaluator
from monai.transforms.post.array import VoteEnsemble
from fran.utils.common import *
from fran.inference.inference_raytune_models import ModelFromTuneTrial
from fran.transforms.totensor import ToTensorF
from fran.transforms.spatialtransforms import *
from monai.data import GridPatchDataset, PatchIter
from monai.inferers import SlidingWindowInferer
from fran.managers.trainer import *
from fran.managers.tune import *
from fran.inference.inference_base import *
from fran.utils.imageviewers import *
from torch.utils.data import DataLoader as DataLoaderT
# %%
if __name__ == "__main__":


    common_paths_filename=os.environ['FRAN_COMMON_PATHS']
    P = Project(project_title="lits"); proj_defaults= P.proj_summary

    configs_excel = ConfigMaker(proj_defaults.configuration_filename,raytune=False).config
# %%
    train_list, valid_list, test_list = get_fold_case_ids(
            fold=configs_excel['metadata']["fold"],
            json_fname=proj_defaults.validation_folds_filename,
        )

    

    mask_files = list((proj_defaults.raw_data_folder/("masks")).glob("*"))
    img_files= list((proj_defaults.raw_data_folder/("images")).glob("*"))
    masks_valid = [filename for filename in mask_files if  get_case_id_from_filename(None, filename) in valid_list]
    imgs_valid =  [proj_defaults.raw_data_folder/"images"/mask_file.name for mask_file in masks_valid]
    imgs_test =  [filename for filename in img_files if  get_case_id_from_filename(None, filename) in test_list]

# %%
    mo_df = pd.read_csv(Path("/media/ub/datasets_bkp/litq/complete_cases/cases_metadata.csv"))
    n= 0
    img_fn =Path(mo_df.image_filenames[n])
    mask_fn =Path(mo_df.mask_filenames[n] )
# %%
    #note: horseshoe kidney  Path('/s/datasets/raw_database/raw_data/kits21/images/kits21_00005.nii.gz')

    run_name_w= "LITS-118" # best trial
    runs_ensemble=["LITS-265","LITS-255","LITS-270","LITS-271","LITS-272"]


# %%

    device='cuda'
    Nep = NeptuneManager(proj_defaults)



# %%
    E = EndToEndPredictor(proj_defaults,run_name_w,runs_ensemble[0],use_neptune=True,device=device,save_localiser=True)
    E.get_localiser_bbox(img_fn)
    E.bboxes = w.bboxes
    E.run_patch_prediction(img_fn)
    bboxes = w.get_bbox_from_pred()
    w = E.predictor_w
    w.decode_pipeline = Pipeline(w.encode_pipeline[:2])
    y = w.decode_pipeline.decode(w.pred)
    y = w.run()
    d = [x for x in w.encode_pipeline][::-1]
# %%
    ImageMaskViewer([w.img_np_orgres[bboxes[0]],w.img_np_orgres[bboxes[0]]])
# %%
    x = w.pred.clone()
    for dec in d[:-2]:
        x = dec.decodes(x)

# %%
    dec = d[-2]
    y = dec.decodes(x)
    y = d[-1].decodes(y)
# %%
    x = w.encode_pipeline.decode(w.pred)
# %%
    w = E.predictor_w
    img = w.img_np_orgres.copy()
    img = img.transpose(2,0,1)
    ImageMaskViewer([x[0][bboxes[0]],w.pred[0]],data_types=['mask','mask'])
    ImageMaskViewer([img,x[0]])
# %%
    a= E.load_model_neptune(Nep,runs_ensemble[0],device=device) 
# %%
    patch_size = [192,192,96]
    out_channels = 3
    resample_spacings = [1,1,2]
# %%
    E.expand_bbox=0.1
    E.proj_defaults=proj_defaults
    E.resample_spacings=resample_spacings
    E.patch_size=patch_size
    E.global_properties = load_dict(proj_defaults.global_properties_filename)
    E.load_case(img_fn,mask_fn,bboxes=E.bboxes)
# %%
    imgs = E.imgs
    imgs = [torch.tensor(img,device=device).unsqueeze(0).unsqueeze(0) for img in imgs]
    img = imgs[0]
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
    # ImageMaskViewer([img_np,mask_np])
    # ImageMaskViewer([img_np,preds[1]])

# %%
    overlap=.25
    # S = SlidingWindowInferer(roi_size=patch_size,mode='gaussian',progress=True,overlap=overlap,sw_batch_size=4)
# %%
    preds = []
    preds_int = []
    for n in range(len(runs_ensemble)):
        run_name_p = runs_ensemble[n]
        model,patch_size,resample_spacings,out_channels= E.load_model_neptune(Nep,run_name_p,device=device) 


        patch_overlap = [int(x*overlap) for x in patch_size]
        P = PatchPredictor(proj_defaults,resample_spacings,patch_size,patch_overlap,device='cpu',softmax=True)
        P.load_model(model,run_name_p,)

        P.load_case(img_filename=img_fn,mask_filename=mask_fn,bboxes=bboxes)
        P.make_prediction()
        P.retain_k_largest_components() # time consuming functio
        preds.append(torch.tensor(P.pred_orgres))
        preds_int.append(torch.tensor(P.pred_final))
# # %%
#         with torch.no_grad():
#             pred = S(inputs=img,network=P.model)
#             preds.append(S(inputs = img,network=model) )
#         del P
#         del model
#         torch.cuda.empty_cache()
# %%
    def pred_mean(preds:list):
        '''
        preds are supplied as raw model output
        '''
    
        pred_avg = torch.stack(preds)
        pred_avg = torch.mean(pred_avg,dim=0)
        return pred_avg
# %%
    def pred_voted(preds_int:list):
        V = VoteEnsemble(num_classes=3)
        preds_int_vote = [pred.unsqueeze(0) for pred in preds_int]
        out = V(preds_int_vote)
        out.squeeze_(0)
        return out

# %%


    mask_pt = ToTensor.encodes(mask_fn)
    preds_avg= pred_mean(preds)
    preds_voted = pred_voted(preds_int)
# %%
    n_classes = 3
    scores=[]

    for pred_pt in [*preds, *preds_int,preds_avg,preds_voted]:
        pred_onehot,mask_onehot = [one_hot(x,classes=n_classes,axis=0).unsqueeze(0) for x in [pred_pt,mask_pt]]
        aa = compute_dice(pred_onehot,mask_onehot, include_background=False)
        print(aa)
        scores.append(aa)

# %%
        n=0
        ImageMaskViewer([img.detach()[0][0],preds[n][0][1]])

# %%
        s = sliding_window_inference(inputs = img_input,roi_size = p.patch_size,sw_batch_size =1,predictor=p.model)
# %%

