# %
import os
# from fran.inference.transforms import *
from fran.inference.scoring import compute_dice_fran
from monai.engines.evaluator import EnsembleEvaluator
from monai.transforms.post.array import VoteEnsemble
from fran.utils.common import *
from fran.inference.inference_raytune_models import ModelFromTuneTrial
from fran.transforms.spatialtransforms import *
from monai.data import GridPatchDataset, PatchIter
from monai.inferers import SlidingWindowInferer
# def load_model(file, model, opt, with_opt=True, device=None, strict=True):
#     "Load `model` from `file` along with `opt` (if available, and if `with_opt`)"
#     distrib_barrier()
#     if isinstance(device, int): device = torch.device('cuda', device)
#     elif device is None: device = 'cpu'
#     state = torch.load(file, map_location=device)
#     hasopt = set(state)=={'model', 'opt'}
#     model_state = state['model'] if hasopt else state
#     get_model(model).load_state_dict(model_state, strict=strict)
#     if hasopt and with_opt:
#         try: opt.load_state_dict(state['opt'])
#         except:
#             if with_opt: warn("Could not load the optimizer state.")
#     elif with_opt: warn("Saved filed doesn't contain an optimizer state.")
#

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


    n= 16
    img_fn = imgs_valid[n]
    mask_fn = img_fn.str_replace("images","masks")
# %%
    #note: horseshoe kidney  Path('/s/datasets/raw_database/raw_data/kits21/images/kits21_00005.nii.gz')

    run_name_w= "LITS-118" # best trial
    runs_ensemble=["LITS-265","LITS-255","LITS-270","LITS-271"]


# %%


    @patch_to(EndToEndPredictor)
    def unload_previous(self):
        attributes = ["backsampled_pred", "mask_cc","mask_sitk", "subject"]
        try:
            for att in attributes:
                delattr(self,att)
        except:
            pass



    @patch_to(EndToEndPredictor)
    def create_encode_pipeline(self):
            T = TransposeSITKToNp()
            R = ResampleToStage0(self.img_sitk,self.resample_spacings)
            B = BBoxesToPatchSize(self.patch_size,self.sz_dest,self.expand_bbox)
            C = ClipCenter(clip_range=self.global_properties['intensity_clip_range'],mean=self.global_properties['mean_fg'],std=self.global_properties['std_fg'])
            self.encode_pipeline = Pipeline([T,R,B,C])

    @patch_to(EndToEndPredictor)
    def create_dl_tio(self):
            self.img_transformed , self.bboxes_transformed= self.encode_pipeline([self.img_np_orgres, self.bboxes])
            self.dls=[]
            self.imgs=[]
            for bbox in self.bboxes_transformed:
                self.imgs.append(self.img_transformed[bbox])

    @patch_to(EndToEndPredictor)
    def load_case(self, img_filename,mask_filename=None, case_id=None, bboxes=None):
        self.unload_previous()
        self.gt_fn = mask_filename
        self.img_filename = img_filename
        self.case_id = get_case_id_from_filename(self.proj_defaults.project_title,self.img_filename) if not case_id else case_id
        self.img_sitk= sitk.ReadImage(str(self.img_filename))
        self.img_np_orgres=sitk.GetArrayFromImage(self.img_sitk)
        self.size_source = self.img_sitk.GetSize()
        self.sz_dest = get_sitk_target_size_from_spacings(self.img_sitk,self.resample_spacings)
        self.bboxes =bboxes 
        self.create_encode_pipeline()
        self.create_dl_tio()


    device='cpu'
    Nep = NeptuneManager(proj_defaults)

    E = EndToEndPredictor(proj_defaults,run_name_w,runs_ensemble[0],use_neptune=True,device=device)
    E.get_localiser_bbox(img_fn,mask_fn)
    bboxes = E.bboxes
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
    overlap=.25
    # S = SlidingWindowInferer(roi_size=patch_size,mode='gaussian',progress=True,overlap=overlap,sw_batch_size=4)
# %%
    preds = []
    preds_int = []
    for n in range(len(runs_ensemble)):
        run_name_p = runs_ensemble[n]
        model,patch_size,resample_spacings,out_channels= E.load_model_neptune(Nep,run_name_p,device=device) 


        patch_overlap = [int(x*overlap) for x in patch_size]
        P = PatchPredictor(proj_defaults,resample_spacings,patch_size,patch_overlap,device='cpu')
        P.load_model(model,run_name_p,)

        P.load_case(img_filename=img_fn,mask_filename=mask_fn,bboxes=bboxes)
        P.make_prediction()
        P.retain_k_largest_components() # time consuming functio
        preds.append(torch.tensor(P.pred_orgres))
        preds_int.append(torch.tensor(P.pred_final))
# %%
        with torch.no_grad():
            pred = S(inputs=img,network=P.model)
            preds.append(S(inputs = img,network=model) )
        del P
        del model
        torch.cuda.empty_cache()
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



    mask_pt = ToTensor.encodes(mask_fn)
# %%
    n_classes = 3
    scores=[]

    for pred_pt in il.chain.from_iterable([preds, preds_int,pred_avg]):
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

