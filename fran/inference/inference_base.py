
# jupyter: ---
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from fran.inference.scoring import compute_dice_fran
from fran.transforms.intensitytransforms import ClipCenter, clip_image, standardize
from monai.metrics.meandice import compute_dice
from fran.managers.base import load_checkpoint
import os

from fran.transforms.totensor import ToTensorI

from fran.utils.sitk_utils import align_sitk_imgs
import sys

from fran.inference.helpers import *
sys.path+=["/home/ub/Dropbox/code/fran/"]
from fran.managers.trainer import *
from fran.managers.tune import ModelFromTuneTrial
from fran.transforms.inferencetransforms import *
from fastcore.transform import NoneType, Transform
from fran.utils.imageviewers import *

# %%
from fran.inference.helpers import get_sitk_target_size_from_spacings
from fran.utils.helpers import *

import torchio as tio
from fastcore.basics import store_attr

from fran.utils.imageviewers import ImageMaskViewer
import torch.nn.functional as F
import cc3d
# from experiments.kits21.kits21_repo.evaluation.metrics import *
# HEC_SD_TOLERANCES_MM = {
#     'kidney_and_mass': 1.0330772532390826,
#     'mass': 1.1328796488598762,
#     'tumor': 1.1498198361434828,
# }
#
import ast

class ArrayToSITKF(Transform):

    def __init__(self,img_sitk): store_attr()
    def encodes(self,pred_pt)->list:
            assert pred_pt.ndim==4, "This requires 4d array, NxDxWxH"
            preds_out = []
            for pred in pred_pt: 
                pred_ = sitk.GetImageFromArray(pred)
                pred_ = align_sitk_imgs(pred_,self.img_sitk)
                preds_out.append(pred_)
            return preds_out

class ArrayToSITKI(Transform):

    def __init__(self,img_sitk): store_attr()
    def encodes(self,pred):
                assert all([pred.ndim==3,'int' in str(pred.dtype)]), "This requires 3d int array, DxWxH"
                pred_ = sitk.GetImageFromArray(pred)
                pred_ = align_sitk_imgs(pred_,self.img_sitk)
                return pred_

def get_k_organs(mask_labels):
            k_components = [tissue['k_largest'] for tissue in mask_labels if tissue['label']==1][0]
            return k_components


class FillBBoxPatches(ItemTransform):
        '''
        Based on size of original image and n_channels output by model, it creates a zerofilled tensor. Then it fills locations of input-bbox with data provided
        '''
        def __init__(self,img_size,out_channels): self.output_img= torch.zeros(out_channels,*img_size)
        def decodes(self,x):
            patches,bboxes = x
            for bbox,pred_patch in zip(bboxes,patches):
                for n in range(self.output_img.shape[0]):
                    self.output_img[n][bbox]=pred_patch[n]
            return self.output_img


class PatchPredictor(object):
    def __init__(self, proj_defaults,out_channels, resample_spacings,  patch_size: list = [128,128,128] ,patch_overlap:Union[list,float,int]=0.5, grid_mode="crop",expand_bbox=0.1,k_components=None, 
                     batch_size=8,stride=None,device=None,softmax=True,dusting_threshold=10):

        '''
        params:
        patch_overlap: float : [0,1] percent overlap of patch_size
                        int : single number -> [int,int,int]
                        list : e.g., [1,256, 256]
        '''
        store_attr(but='new_subfolder')
        if not k_components:
            self.k_components = get_k_organs(proj_defaults.mask_labels)

        assert grid_mode in ["crop","average"], "grid_mode should be either 'crop' or 'average' "
        self.grid_mode = grid_mode
        global_properties_fname = self.proj_defaults.global_properties_filename
        if len(patch_size)==2: #2d patch size
            patch_size = [1]+patch_size
        if isinstance(patch_overlap,float):
            self.patch_overlap = [int(x*patch_overlap) for x in patch_size]
        elif isinstance(patch_overlap,int):
            self.patch_overlap = [patch_overlap]*3
        self.global_properties = load_dict(global_properties_fname)
        if device is None:
            self.device = get_available_device()
    def load_case(self, img_filename, bboxes=None):
        self.img_filename = img_filename
        self.case_id = get_case_id_from_filename(None,self.img_filename) 
        self.img_sitk= sitk.ReadImage(str(self.img_filename))
        self.img_np_orgres=sitk.GetArrayFromImage(self.img_sitk)
        self.size_source = self.img_sitk.GetSize()
        self.sz_dest = get_sitk_target_size_from_spacings(self.img_sitk,self.resample_spacings)
        self.bboxes =bboxes if bboxes else self.set_patchsized_bbox()

    def run(self) :
        '''
        Runs predictions. Then backsamples predictions to img size (DxHxW). Keeps num_channels
        '''

        assert hasattr(self,'img_np_orgres'), "run load_case first."
        self.create_encode_pipeline()
        self.create_dl_tio() 
        self.create_decode_pipeline()
        self.make_prediction()
        self.post_process()



    def set_patchsized_bbox(self):
        shape = self.img_np_orgres.shape
        slc=[]
        for s in shape:
            slc.append(slice(0,s))
        return ([tuple(slc)])

    def create_encode_pipeline(self):
            To=ToTensorI()
            Ch=ChangeDType(torch.float32)
            T = TransposeSITK()
            R = ResampleToStage0(self.img_sitk,self.resample_spacings)
            B = BBoxesToPatchSize(self.patch_size,self.sz_dest,self.expand_bbox)
            C = ClipCenter(clip_range=self.global_properties['intensity_clip_range'],mean=self.global_properties['mean_fg'],std=self.global_properties['std_fg'])
            self.encode_pipeline = Pipeline([To,Ch,T,R,B,C])
    def patches_to_orgres(self):
        x = self.img_transformed, self.bboxes_transformed
        pred_neo = torch.zeros(x[0].shape)
        for bbox,pred_patch in zip(x[1],self.pred_patches):
            pred_neo[bbox]=pred_patch[0]
        BM = BacksampleMask(self.img_sitk)
        pred_orgres= BM(pred_neo)
        self.pred= pred_orgres.permute(2,1,0)

    def create_decode_pipeline(self):
        F = FillBBoxPatches(self.img_transformed.shape,self.out_channels)
        self.decode_pipeline = Pipeline([*self.encode_pipeline[2:4],F]) # i.e., TransposeSITK, ResampleToStage0
    

    def create_dl_tio(self):
            self.img_transformed , self.bboxes_transformed= self.encode_pipeline([self.img_np_orgres, self.bboxes])
            self.dls=[]
            for bbox in self.bboxes_transformed:
                img_cropped = self.img_transformed[bbox]
                img_tio= tio.ScalarImage(tensor=np.expand_dims(img_cropped,0))
                subject = tio.Subject(image=img_tio)
                grid_sampler = tio.GridSampler(subject=subject,patch_size=self.patch_size,patch_overlap=self.patch_overlap) 
                aggregator = tio.inference.GridAggregator(grid_sampler,overlap_mode=self.grid_mode) 
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size)
                self.dls.append([patch_loader,aggregator])



    def load_model(self,model, model_id):
        self._model = model
        self._model.eval()
        self._model.to(self.device)
        self._model_id=model_id

        self.output_image_folder= ("segmentations"+"_"+self._model_id)
        print("Default output folder, based on model name is {}.".format(self.output_image_folder))

    def make_prediction(self):
        self.pred_patches=[]
        for dl_a in self.dls:
            pl , agg = dl_a
            with torch.no_grad():
                    for i,patches_batch in enumerate(pl):
                        input_tensor = patches_batch['image'][tio.DATA].float().to(self.device)
                        locations = patches_batch[tio.LOCATION]
                        y = self._model(input_tensor)
                        if isinstance(y,(tuple,list)) :
                            y=y[0]
                        agg.add_batch(y, locations)

            output_tensor = agg.get_output_tensor()
            output_tensor = torch.nan_to_num(output_tensor,0)
            if self.softmax == True:
                output_tensor=F.softmax(output_tensor,dim=0)
            self.pred_patches.append(output_tensor)


    def post_process(self,save=True,retain_k_label=2): # which label should have k-largest components retains
        self.pred= self.decode_pipeline.decode([self.pred_patches,self.bboxes_transformed])
        self.retain_k_largest_components(retain_k_label) # time consuming function
        if save==True:
            self.save_prediction()



    def save_prediction(self,ext=".nii.gz"):
        maybe_makedirs(self._output_image_folder)
        counts = ["","_1","_2","_3","_4"][:len([self.pred_sitk_i,*self.pred_sitk_f])]
        self.pred_fns= [self._output_image_folder/(self.img_filename.name.split(".")[0]+c+ext) for c in counts]
        print("Saving prediction. File name : {}".format(self.pred_fns))
        for pred_sitk,fn in zip([self.pred_sitk_i]+self.pred_sitk_f,self.pred_fns):
            sitk.WriteImage(pred_sitk,fn)

    def score_prediction(self,mask_filename,n_classes):
        mask_sitk = sitk.ReadImage(mask_filename)
        self.scores = compute_dice_fran(self.pred_int,mask_sitk,n_classes)
        return self.scores

    def view_predictions(self, mode:str = "raw",view_channel=1,orientation='A'):
        assert mode in ["raw","softmax","binary"], "Mode has to be one of : 'raw', 'softmax', 'binary'"
        i= torch.tensor(self.img_np_orgres)
        if mode == 'binary':
            j = torch.tensor(self.mask.astype(np.uint8))
        else:
            j = torch.clone(self.backsampled_pred)
            if mode=="softmax":
                j = F.softmax(j,0)
            j = j[view_channel]

        perm={'S':(0,1,2),
            'A':(2,1,0),
            'C':(1,2,0)
        }
        i ,j = i.permute(*perm[orientation]),j.permute(*perm[orientation])

        ImageMaskViewer([i,j])
            
           
    def unload_case(self):
        to_delete = ['_pred_int','_pred_sitk_f','_pred_sitk_i']
        for item in to_delete: delattr(self,item)
       
    def retain_k_largest_components(self,label:int):
        M = MaskToBinary(label,self.out_channels)
        pred_tmp_binary =  M.encodes(self.pred_int)
        pred_tmp_binary= cc3d.dust(pred_tmp_binary,threshold=self.dusting_threshold,connectivity=26,in_place=True)
        pred_k_largest, N= cc3d.largest_k(pred_tmp_binary,k=self.k_components, return_N=True)
        return pred_k_largest


    def get_bbox_from_pred(self):
        stats = cc3d.statistics(self.dusted_stencil(1))
        bboxes = stats['bounding_boxes'][1:] # bbox 0 is the whole image
        bboxes
        if len(bboxes)<1:
            tr()
        return bboxes
    

    def dusted_stencil(self,axis:int):
        assert isinstance(axis,int),"Provide axis / label index which should serve as the foreground of bbox"
        attr_name = "_".join(["dusted_stencil",str(axis)])
        if not hasattr(self,attr_name):
            dusted = self.retain_k_largest_components(label=axis)

            setattr(self,attr_name,dusted)
        return getattr(self,attr_name)


    @property 
    def pred_sitk_f(self): # list of sitk images len =  out_channels
        if not hasattr(self,"_pred_sitk_f"):
            self._pred_sitk_f = ArrayToSITKF(self.img_sitk).encodes(self.pred)
        return self._pred_sitk_f[1:] #' first channel is only bg'

    @property 
    def pred_sitk_i(self): # list of sitk images len =  out_channels
        if not hasattr(self,"_pred_sitk_i"):
            self._pred_sitk_i = ArrayToSITKI(self.img_sitk).encodes(self.pred_int)
        return self._pred_sitk_i

    @property
    def pred_int(self):
        if not hasattr(self,"_pred_int"):
            self._pred_int= torch.argmax(self.pred, 0, keepdim=False)
            self._pred_int=self._pred_int.to(torch.uint8)
        return self._pred_int
    @property
    def output_image_folder(self):
        return self._output_image_folder

    @output_image_folder.setter
    def output_image_folder(self,folder_name: Path):
        self._output_image_folder =Path(self.proj_defaults.predictions_folder)/folder_name

    @property
    def model(self):
        return self._model

                       
class WholeImageBBoxes(ApplyBBox):
        def __init__(self, patch_size):
            bboxes=[tuple([slice(0,p) for p in patch_size])]
            super().__init__(bboxes)

        def encodes(self, x):
            img,_= x
            return img,self.bboxes

        def decodes(self,x) : return x # no processing to do
        

class Unlist(Transform):  
    def decodes(self,x:list): 
        assert len(x)==1, "Only for lists len=1"
        return x[0]

class WholeImagePredictor(PatchPredictor):
    def __init__(self, proj_defaults, out_channels,resample_spacings,  patch_size: list = [128,128,128], device=None,**kwargs):
        super().__init__(proj_defaults,out_channels,resample_spacings,patch_size,patch_overlap=0,batch_size=1,device=device,**kwargs)
 
    def create_encode_pipeline(self):
            To= ToTensorI()
            Ch=ChangeDType(torch.float32)
            T = TransposeSITK()
            Rz = Resize(self.patch_size)
            # P  = PadDeficitImgMask(patch_size=self.patch_size,input_dims=3)
            W = WholeImageBBoxes(self.patch_size)
            C = ClipCenter(clip_range=self.global_properties['intensity_clip_range'],mean=self.global_properties['mean_fg'],
                           std=self.global_properties['std_fg'])
            self.encode_pipeline = Pipeline([To,Ch,T,Rz,W,C])

    def create_decode_pipeline(self): 
            U = Unlist()
            self.decode_pipeline = Pipeline([*self.encode_pipeline[2:4],U])

    def post_process(self):
        self.pred= self.decode_pipeline.decode(self.pred_patches)


class EndToEndPredictor(object):
    def __init__(self, proj_defaults,run_name_w,run_name_p,use_neptune=False,patch_overlap=0.5,device='cuda', save_localiser=False):
        print("Loading model checkpoints for whole image predictor")
        if use_neptune==True:
            Nep = NeptuneManager(proj_defaults)
            model_w, patch_size_w,resample_spacings_w,out_channels_w= self.load_model_neptune(Nep,run_name_w,device=device)
            model_p, patch_size_p, resample_spacings_p,out_channels_p= self.load_model_neptune(Nep,run_name_p,device=device)
        else:
            print("Not fully implemented")
            P = ModelFromTuneTrial(proj_defaults, trial_name=run_name_w,out_channels= 2)
            model_w = P.model
            patch_size_w= make_patch_size(P.params_dict['dataset_params']['patch_dim0'], P.params_dict['dataset_params']['patch_dim1'])


        self.predictor_w = WholeImagePredictor(proj_defaults=proj_defaults,out_channels= out_channels_w,resample_spacings=resample_spacings_w, patch_size=patch_size_w,device=device)
        self.predictor_w.load_model(model_w,model_id=run_name_w)
        self.save_localiser = save_localiser
        print("\nLoading model checkpoints for patch-based predictor")


        patch_overlap = [int(x*patch_overlap) for x in patch_size_p]
        self.predictor_p= PatchPredictor(proj_defaults=proj_defaults, out_channels= out_channels_p,resample_spacings=resample_spacings_p,patch_size=patch_size_p,patch_overlap=patch_overlap, 
                                            stride = [1,1,1], batch_size=4,device=device)
        self.predictor_p.load_model(model_p,model_id=run_name_p)
        self.n_classes=out_channels_p

        print("---- You can set alternative save folders by setting properties: output_localiser_folder and output_image_folder for localiser and final predictions respectively.----")

     
    def load_model_neptune(self, NepMan, run_name, device= 'cuda'):
            NepMan.load_run(run_name=run_name,param_names = 'default',nep_mode="read-only")
            metadata= NepMan.run_dict['metadata']
            model_params = NepMan.run_dict['model_params']
            dataset_params = NepMan.run_dict['dataset_params']
            resample_spacings = ast.literal_eval(NepMan.run_dict['dataset_params']['spacings'])
            if not   'out_channels' in model_params:
                oc = {'out_channels':  out_channels_from_dict_or_cell(NepMan.run_dict['metadata']['src_dest_labels'])}
                model_params['out_channels']  = out_channels_from_dict_or_cell(metadata['src_dest_labels'])
            out_channels = model_params['out_channels']
            patch_size=NepMan.run_dict['dataset_params']['patch_size']
            model= create_model_from_conf(model_params,dataset_params,metadata,deep_supervision=False)
            load_checkpoint(metadata['model_dir'],model,device)
            return model, patch_size , resample_spacings, out_channels

    def predict(self,img_fn,mask_fn = None,save_localiser=None):
        if save_localiser: self.save_localiser=save_localiser
        self.get_localiser_bbox(img_fn,mask_fn)
        self.run_patch_prediction(img_fn)

    def get_localiser_bbox(self,img_fn,mask_fn=None):
        print("Running predictions. Whole image predictor is on device {0}. Patch-based predictor is on device {1}".format(self.predictor_w.device,self.predictor_p.device))
        self.predictor_w.load_case(img_filename=img_fn)
        self.predictor_w.run()
        self.bboxes = self.predictor_w.get_bbox_from_pred()
        if self.save_localiser==True: self.predictor_w.save_prediction()
    def run_patch_prediction(self,img_fn):
        self.predictor_p.load_case(img_filename=img_fn,bboxes=self.bboxes)
        self.predictor_p.run()
    def unload_case(self):
        self.predictor_w.unload_case()
        self.predictor_p.unload_case()

    def score_prediction(self,mask_fn):
        self.scores = self.predictor_p.score_prediction(mask_fn,self.n_classes)



    @property
    def case_id(self):
        """The case_id property."""
        return self.predictor_p.case_id
    @case_id.setter
    def case_id(self, value):
        self._case_id = value
    @property
    def output_image_folder(self):
        return self.predictor_p.output_image_folder

    @output_image_folder.setter
    def output_image_folder(self,folder_name: Path):
        self.predictor_p.output_image_folder= folder_name

    @property
    def output_localiser_folder(self):
        return self.predictor_w.output_image_folder

    @output_localiser_folder.setter
    def output_localiser_folder(self,folder_name: Path):
        self.predictor_w.output_image_folder= folder_name

    @property
    def save_localiser(self):
        """The save_localiser property."""
        return self._save_localiser
    @save_localiser.setter
    def save_localiser(self, value=None):
        assert not value or type(value)==bool , "Illegal value for bool parameter"
        if not value or value == False:
            print("Localizer image will not be saved")
            value = False
        else:
            print("Localizer image will be saved to {}".format(self.output_localiser_folder))
        self._save_localiser=value

    @property
    def pred(self): return self.predictor_p.pred

    @property
    def pred_sitk(self): return self.predictor_p.pred_sitk_i
# %%
if __name__ =="__main__":
    from fran.utils.common import *
    common_paths_filename=os.environ['FRAN_COMMON_PATHS']
    P = Project(project_title="lits"); proj_defaults= P.proj_summary

    mo_df = pd.read_csv(Path("/media/ub/datasets_bkp/litq/complete_cases/cases_metadata.csv"))
    n= 0
    img_fn =Path(mo_df.image_filenames[n])
    mask_fn =Path(mo_df.mask_filenames[n] )
    patch_size = [160,160,160]
    resample_spacings = [1,1,2]
    run_name_w= "LITS-118" # best trial
    runs_ensemble=["LITS-265","LITS-255","LITS-270","LITS-271","LITS-272"]
    run_name_p = runs_ensemble[0]
    device='cuda'
# %%
    E = EndToEndPredictor(proj_defaults,run_name_w,runs_ensemble[0],use_neptune=True,device=device,save_localiser=True)
    
# %%
    run_name_p='LITS-265'
    bboxes = [(slice(359, 548, None), slice(173, 361, None), slice(111, 277, None))]
    resample_spacings= [1,1,2]
    patch_size = [192, 192, 96]
    img_fn = Path('/media/ub/datasets_bkp/litq/complete_cases/images/litq_0564527_20191116.nii')
# %%
    P = PatchPredictor(proj_defaults,3,resample_spacings,patch_size,device=device)
    P.load_model(E.predictor_p.model,run_name_p)
    P.load_case(img_fn,bboxes=bboxes)
    P.run()
# %%
    score = P.score_prediction(img_fn,3)
# %%
# %%
    w = E.predictor_w
    w.img_np_orgres.shape
    w = WholeImagePredictor(proj_defaults,2,resample_spacings,[160,160,160],'cuda')
    w.load_model(E.predictor_w.model,'chap')
    w.load_case(img_fn)
    w.run()
# %%
    P = E.predictor_p
    bboxes = E.bboxes
    sl = [P.img_np_orgres,P.bboxes]
    a = P.encode_pipeline[0].encodes(sl)
    a = P.encode_pipeline[1].encodes(a)
    a = P.encode_pipeline[2].encodes(a)
    b = P.encode_pipeline[3].encodes(a)


    Rz = Resize(w.patch_size)
# %%
    P.img_np_orgres.shape
    P.img_transformed.shape

# %%

