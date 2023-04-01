
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

import ast
from fastcore.foundation import L
from fastcore.all import GetAttr
from fran.inference.scoring import compute_dice_fran
from fran.transforms.intensitytransforms import ClipCenter
from monai.transforms.post.array import VoteEnsemble
from fran.managers.base import load_checkpoint
import os

from fran.transforms.totensor import ToTensorI, ToTensorT

import sys

from fran.inference.helpers import *
sys.path+=["/home/ub/Dropbox/code/fran/"]
from fran.managers.trainer import *
from fran.transforms.inferencetransforms import *
from fastcore.transform import Transform
from fran.utils.imageviewers import *
import functools as fl
from fran.transforms.inferencetransforms import *
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
def pred_mean(preds):
    '''
    preds are supplied as raw model output
    '''

    pred_avg = torch.stack(preds)
    pred_avg = torch.mean(pred_avg,dim=0)
    return pred_avg
def pred_voted(preds_int:list):
    V = VoteEnsemble(num_classes=3)
    preds_int_vote = [pred.unsqueeze(0) for pred in preds_int]
    out = V(preds_int_vote)
    out.squeeze_(0)
    return out
def get_k_organs(mask_labels):
            k_components = [tissue['k_largest'] for tissue in mask_labels if tissue['label']==1][0]
            return k_components




def voxvol(spacings): return fl.reduce(operator.mul,spacings)  # returnc volume of voxel  in mm3
def volvox(vol_tot,spacings)->int: # gives number of voxels which cover said volume
       '''
       vol_tot must be in cc
       '''
       vol_tot_mm3= vol_tot*1e3
       voxelvol = voxvol(spacings) # mm^3
       return int(vol_tot_mm3/voxelvol)
        
def dust_and_k_largest(self,pred_int): 
    self.dusted_stencils = self.create_dusted_stencils_k_largest(pred_int)
    return self.mask_fill_incremental()

def mask_fill_incremental(self):
    pred_int_dusted= self.dusted_stencils[0] # zero mask
    for n in range(1,len(self.dusted_stencils)):
        pred_int_dusted[self.dusted_stencils[n]==1]= n
    return pred_int_dusted

def create_dusted_stencils_k_largest(self,pred_int):
    dusted_stencils= [torch.zeros(pred_int.shape,dtype=torch.uint8)]
    for axis in range(1,self.out_channels):
        dusted_stencils.append(self.create_dusted_stencil_k_largest(pred_int,axis))
    return dusted_stencils

def create_dusted_stencil_k_largest(self,pred_int,axis:int): 
        assert isinstance(axis,int),"Provide axis / label index which should serve as the foreground of bbox"
        M = MaskToBinary(axis,self.out_channels)
        pred_tmp_binary =  M.encodes(pred_int)
        pred_tmp_binary= cc3d.dust(pred_tmp_binary,threshold=self.dusting_threshold,connectivity=26,in_place=True)
        dusted = self.retain_k_largest_components(pred_tmp_binary)
        return dusted

   
def retain_k_largest_components(self,tnsr_binary):  # works with pred_int form
    pred_k_largest, N= cc3d.largest_k(tnsr_binary,k=self.k_components, return_N=True)
    pred_k_largest = pred_k_largest.astype(np.uint8)
    pred_k_largest = torch.tensor(pred_k_largest)
    return pred_k_largest

class PredFlToInt(Transform):
    def encodes(self,pred_fl:Tensor):
        pred_int= torch.argmax(pred_fl ,0, keepdim=False)
        pred_int=pred_int.to(torch.uint8)
        return pred_int

class ToNumpy(Transform):
    def __init__(self,encode_dtype=np.uint8): 
        if encode_dtype == np.uint8: self.decode_dtype=torch.uint8
        self.encode_dtype=encode_dtype


def encodes(self,tnsrs):
        return [np.array(tnsr,dtype=self.encode_dtype) for tnsr in tnsrs]

class Stencil(Transform):
    def __init__(self,n_classes, merge_labels=[]): 
        '''
        param merge_labels: list of lists, with one list for each of label 1 onwards, containing the label to be merged into the target label
        e.g., for a 0,1,2 label mask (1 is organ, 2 is tumour) [2],[] maps 2->1 when creating organ stencil.
        Note: No stencil is needed/created for background label 0.
        '''
        
        # if not merge_labels: merge_labels = [2],[1]
        fg_classes = n_classes-1
        assert len(merge_labels) == fg_classes, "Give a list (even if empty) for each channel from 1 onwards"
        self.ms = [MaskToBinary(axis,n_classes,merge_labels=ml) for ml,axis in zip(merge_labels,range(1,n_classes))]
    def encodes(self,x):
        pred_int = x
        stencils = [m.encodes(pred_int) for m in self.ms]
        return stencils

class DustKLargest(Transform):
    def __init__(self,mask_labels,spacings):store_attr()
    def encodes(self,stencils):
        stencils_out=[]
        for indx, stencil in enumerate(stencils):
            label = indx+1
            info = self.mask_labels[str(label)]
            dusting_threshold = volvox(info['dusting_threshold'],self.spacings)
            stencil_dusted= cc3d.dust(stencil,threshold=dusting_threshold,connectivity=26,in_place=True)

            stencil_k= cc3d.largest_k(stencil_dusted,k=info['k_largest'], return_N=False)
            stencils_out.append(stencil_k)
        return stencils_out

class MergeStencils(Transform):
    def __init__(self,img_shape):store_attr()
    def encodes(self,stencils):
        pred_int= torch.zeros(self.img_shape,dtype=torch.uint8)
        for n,stencil in enumerate(stencils):
            pred_int[stencil==1]= n+1 # +1 because there is no zero stencil
        return pred_int

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

class _Predictor(GetAttr):

    def save_prediction(self,ext=".nii.gz"):
        maybe_makedirs(self._output_image_folder)
        if self.debug==True:
            counts = ["","_1","_2","_3","_4"][:len([self.pred_sitk_i,*self.pred_sitk_f])]
            self.pred_fns= [self._output_image_folder/(self.img_filename.name.split(".")[0]+c+ext) for c in counts]
            print("Saving prediction. File name : {}".format(self.pred_fns))
            for pred_sitk,fn in zip([self.pred_sitk_i]+self.pred_sitk_f,self.pred_fns):
                sitk.WriteImage(pred_sitk,fn)
        else:
            fn  = self._output_image_folder/(self.img_filename.name.split(".")[0]+ext)
            print("Saving prediction. File name : {}".format(fn))
            sitk.WriteImage(self.pred_sitk_i,fn)
    def score_prediction(self,mask_filename,n_classes):
        mask_sitk = sitk.ReadImage(mask_filename)
        self.scores = compute_dice_fran(self.pred_int,mask_sitk,n_classes)
        return self.scores

    def postprocess(self,cc3d:bool): # starts : pred->dust->k-largest->pred_int
        if cc3d == True:
            self.pred_int = self.postprocess_pipeline(self.pred)
        else:
            pred_int= torch.argmax(self.pred, 0, keepdim=False)
            self.pred_int=pred_int.to(torch.uint8)

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
        for item in to_delete: 
            if hasattr(self,item):
                delattr(self,item)

    @property 
    def pred_sitk_f(self): # list of sitk images len =  out_channels
        self._pred_sitk_f = ArrayToSITKF(sitk_props=self.sitk_props).encodes(self.pred)
        return self._pred_sitk_f[1:] #' first channel is only bg'

    @property 
    def pred_sitk_i(self): # list of sitk images len =  out_channels
        self._pred_sitk_i = ArrayToSITKI(sitk_props=self.sitk_props).encodes(self.pred_int)
        return self._pred_sitk_i
    @property
    def output_image_folder(self):
        return self._output_image_folder

    @output_image_folder.setter
    def output_image_folder(self,folder_name: Path):
        self._output_image_folder =Path(self.proj_defaults.predictions_folder)/folder_name


    @property
    def case_id(self):
        """The case_id property."""
        return self.predictor_p.case_id
    @case_id.setter
    def case_id(self, value):
        self._case_id = value


class PatchPredictor(_Predictor):
    def __init__(self, proj_defaults,out_channels, resample_spacings,  patch_size: list = [128,128,128] ,patch_overlap:Union[list,float,int]=0.5, grid_mode="crop",expand_bbox=0.1,
                     batch_size=8,stride=None,softmax=True,device=None,merge_labels = [[2],[]] ,postprocess_label=2, cc3d=True, debug=False):

        '''
        params:
        cc3d: If True, dusting and k-largest components are extracted based on mask-labels.json (corrected from mm^3 to voxels)
        patch_overlap: float : [0,1] percent overlap of patch_size
                        int : single number -> [int,int,int]
                        list : e.g., [1,256, 256]
        '''
        store_attr(but='new_subfolder')
        # used by stencil transform. WHen creating label1 stencil, label2 will be counted in label1 to avoid holes after dusting

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

   

    def set_sitk_props(self):
        origin = self.img_sitk.GetOrigin()
        spacing = self.img_sitk.GetSpacing()
        direction = self.img_sitk.GetDirection()
        if direction != (1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0): 
            self.img_sitk =sitk.DICOMOrient(self.img_sitk,"LPS")
        self.sz_dest ,self.scale_factor= get_scale_factor_from_spacings(self.img_sitk.GetSize(),spacing,self.resample_spacings)
        self.sitk_props = origin,spacing, direction


    def load_case(self, img_filename, bboxes=None): # tip put this inside a transform which saves these attrs to parent like callbacks do in learner
        self.img_filename = img_filename
        self.case_id = get_case_id_from_filename(None,self.img_filename) 
        self.img_sitk= sitk.ReadImage(str(self.img_filename))
        self.set_sitk_props()
        self.img_np_orgres=sitk.GetArrayFromImage(self.img_sitk)
        self.bboxes =bboxes if bboxes else self.set_patchsized_bbox()

    def run(self,img_filename,bboxes=None,save=True) :
        '''
        Runs predictions. Then backsamples predictions to img size (DxHxW). Keeps num_channels
        '''
        self.load_case(img_filename,bboxes)
        self.create_encode_pipeline()
        self.create_dl_tio() 
        self.create_decode_pipeline()
        self.create_postprocess_pipeline()
        self.make_prediction()
        self.backsample()
        self.postprocess(self.cc3d)
        if save==True:
            self.save_prediction()



    def set_patchsized_bbox(self):
        shape = self.img_np_orgres.shape
        slc=[]
        for s in shape:
            slc.append(slice(0,s))
        return ([tuple(slc)])

    def create_encode_pipeline(self):

        self.encode_tfms=L(
            ToTensorI(),
            ChangeDType(torch.float32),
            TransposeSITK(),
            ResampleToStage0(self.img_sitk,self.resample_spacings),
            BBoxesToPatchSize(self.patch_size,self.sz_dest,self.expand_bbox),
            ClipCenter(clip_range=self.global_properties['intensity_clip_range'],mean=self.global_properties['mean_fg'],std=self.global_properties['std_fg']),
            )
        self.encode_tfms.map(self.add_tfm)
        self.encode_pipeline = Pipeline(self.encode_tfms)

    def create_postprocess_pipeline(self):
        self.postprocess_tfms=L(
            PredFlToInt,
            Stencil(self.out_channels,self.merge_labels),
            ToNumpy,
            DustKLargest(self.proj_defaults.mask_labels,self.sitk_props[1]),
            ToTensorT,
            MergeStencils(self.img_np_orgres.shape),
        )

        self.postprocess_tfms.map(self.add_tfm)
        self.postprocess_pipeline = Pipeline(self.postprocess_tfms)
    def add_tfm(self,tfm):
        if isinstance(tfm,type): tfm= tfm()
        tfm.predictor= self
        setattr(self,tfm.name,tfm)
        return self
        
    # def patches_to_orgres(self):
    #     x = self.img_transformed, self.bboxes_transformed
    #     pred_neo = torch.zeros(x[0].shape)
    #     for bbox,pred_patch in zip(x[1],self.pred_patches):
    #         pred_neo[bbox]=pred_patch[0]
    #     BM = BacksampleMask(self.img_sitk)
    #     pred_orgres= BM(pred_neo)
    #     self.pred= pred_orgres.permute(2,1,0)
    #
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
        self.output_image_folder= (self._model_id)
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


    def backsample(self):
        self.pred= self.decode_pipeline.decode([self.pred_patches,self.bboxes_transformed])





           
       
    def retain_k_largest_components(self,tnsr_binary):  # works with pred_int form
        pred_k_largest, N= cc3d.largest_k(tnsr_binary,k=self.k_components, return_N=True)
        pred_k_largest = pred_k_largest.astype(np.uint8)
        pred_k_largest = torch.tensor(pred_k_largest)
        return pred_k_largest

    def dust_and_k_largest(self,pred_int): 
        self.dusted_stencils = self.create_dusted_stencils_k_largest(pred_int)
        return self.mask_fill_incremental()

    def mask_fill_incremental(self):
        pred_int_dusted= self.dusted_stencils[0] # zero mask
        for n in range(1,len(self.dusted_stencils)):
            pred_int_dusted[self.dusted_stencils[n]==1]= n
        return pred_int_dusted

    def create_dusted_stencils_k_largest(self,pred_int):
        dusted_stencils= [torch.zeros(pred_int.shape,dtype=torch.uint8)]
        for axis in range(1,self.out_channels):
            dusted_stencils.append(self.create_dusted_stencil_k_largest(pred_int,axis))
        return dusted_stencils

    def create_dusted_stencil_k_largest(self,pred_int,axis:int): 
        assert isinstance(axis,int),"Provide axis / label index which should serve as the foreground of bbox"
        M = MaskToBinary(axis,self.out_channels)
        pred_tmp_binary =  M.encodes(pred_int)
        pred_tmp_binary= cc3d.dust(pred_tmp_binary,threshold=self.dusting_threshold,connectivity=26,in_place=True)
        dusted = self.retain_k_largest_components(pred_tmp_binary)
        return dusted


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
    def __init__(self, proj_defaults, out_channels,resample_spacings,  patch_size: list = [128,128,128], device=None,merge_labels=[[]],postprocess_label=1,**kwargs):

        super().__init__(proj_defaults=proj_defaults,out_channels=out_channels,resample_spacings=resample_spacings,patch_size=patch_size,patch_overlap=0.2,
                         batch_size=1,device=device,merge_labels=merge_labels,postprocess_label=postprocess_label, **kwargs)
 

    def get_bbox_from_pred(self,label):
        pred_int_np = np.array(self.pred_int)
        stats = cc3d.statistics(pred_int_np)
        bboxes = stats['bounding_boxes'][1:] # bbox 0 is the whole image
        bboxes
        if len(bboxes)<1:
            tr()
        return bboxes

    def create_encode_pipeline(self):
            To = ToTensorI()
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
            self.decode_pipeline = Pipeline([*self.encode_pipeline[2:4],U]) # Transpose, Resize

    def backsample(self):
        self.pred= self.decode_pipeline.decode(self.pred_patches)

class EndToEndPredictor(_Predictor):
    def __init__(self, proj_defaults,run_name_w,run_name_p,use_neptune=False,patch_overlap=0.5,device:int=None, save_localiser=False):
        print("Loading model checkpoints for whole image predictor")
        if not device: device = get_available_device()
        if use_neptune==True:
            Nep = NeptuneManager(proj_defaults)
            model_w, patch_size_w,resample_spacings_w,out_channels_w= self.load_model_neptune(Nep,run_name_w,device=device)
            model_p, patch_size_p, resample_spacings_p,out_channels_p= self.load_model_neptune(Nep,run_name_p,device=device)


        self.w = WholeImagePredictor(proj_defaults=proj_defaults,out_channels= out_channels_w,resample_spacings=resample_spacings_w, patch_size=patch_size_w,device=device)
        self.w.load_model(model_w,model_id=run_name_w)
        self.save_localiser = save_localiser
        print("\nLoading model checkpoints for patch-based predictor")


        patch_overlap = [int(x*patch_overlap) for x in patch_size_p]
        self.p= PatchPredictor(proj_defaults=proj_defaults, out_channels= out_channels_p,resample_spacings=resample_spacings_p,patch_size=patch_size_p,patch_overlap=patch_overlap, 
                                            stride = [1,1,1], batch_size=4,device=device)
        self.p.load_model(model_p,model_id=run_name_p)
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

    def predict(self,img_fn,save_localiser=None):
        if save_localiser: self.save_localiser=save_localiser
        self.localiser_bbox(img_fn)
        self.run_patch_prediction(img_fn)

    def localiser_bbox(self,img_fn):
        print("Running predictions. Whole image predictor is on device {0}".format(self.w.device))
        self.w.run(img_filename=img_fn,bboxes=None,save=self.save_localiser)
        self.bboxes = self.w.get_bbox_from_pred(1)

    def unload_localizer_model(self):
        delattr(self,'w')
        print("BBoxes obtained. Deleting localiser and freeing ram")
        torch.cuda.empty_cache()

    def run_patch_prediction(self,img_fn):
        self.p.run(img_filename=img_fn,bboxes=self.bboxes,save=True)
    def unload_cases(self):
        self.unload_localizer_model()
        self.w.unload_case()
        self.p.unload_case()

    def score_prediction(self,mask_fn):
        self.scores = self.p.score_prediction(mask_fn,self.n_classes)


    @property
    def output_localiser_folder(self):
        return self.w.output_image_folder

    @output_localiser_folder.setter
    def output_localiser_folder(self,folder_name: Path):
        self.w.output_image_folder= folder_name

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


class EnsemblePredictor(EndToEndPredictor):
    def __init__(self,proj_defaults,run_name_w,runs_p,device,debug=False,cc3d=False):
        '''
        param  debug: When true, prediction heatmaps are stored as numbered sitk files, each number representing the prob of that label versus all others
        '''
        
        store_attr()
        self.Nep = NeptuneManager(proj_defaults)
        self.patch_overlap=0.25
        self.output_image_folder = "ensemble_"+"_".join(self.runs_p)

    def load_localiser_model(self,run_name_w):
            model_w, patch_size_w,resample_spacings_w,out_channels_w= self.load_model_neptune(self.Nep,run_name_w,device='cpu')
            self.w = WholeImagePredictor(proj_defaults=self.proj_defaults,out_channels= out_channels_w,resample_spacings=resample_spacings_w, patch_size=patch_size_w,device=self.device)
            self.w.load_model(model_w,model_id=run_name_w)
            self.save_localiser=True
    def load_patch_model(self,n):
        run_name_p = self.runs_p[n]
        model_p, patch_size_p, resample_spacings_p,out_channels_p= self.load_model_neptune(self.Nep,run_name_p,device='cpu')
        if n ==0:
            self.p= PatchPredictor(proj_defaults=self.proj_defaults, out_channels= out_channels_p,resample_spacings=resample_spacings_p,patch_size=patch_size_p,patch_overlap=self.patch_overlap, 
                                            stride = [1,1,1], batch_size=4,device=self.device,debug=self.debug)
            self.n_classes=out_channels_p

        self.p.load_model(model_p,model_id=run_name_p)

    def run_patch_prediction(self,img_fn,n):
        if n ==0:
            self.p.load_case(img_fn,self.bboxes)
            self.p.create_encode_pipeline()
            self.p.create_dl_tio() 
            self.p.create_decode_pipeline()
            self.p.create_postprocess_pipeline()
        self.p.make_prediction()
        
        self.p.backsample()
        self.p.postprocess(self.cc3d)
        self.p.save_prediction()
        print("Patch predictions done. Deleting current model in the ensemble and loading next")
        self.p.unload_case()
        torch.cuda.empty_cache()

    def postprocess(self,cc3d:bool): 
        if cc3d==True:
            self.pred_int=self.p.postprocess_pipeline(self.pred)
        else:
            super().postprocess(cc3d=False)

    def save_prediction(self):
        self.sitk_props = self.p.sitk_props
        super().save_prediction()

    def run(self,img_fn):
        self.img_filename = img_fn
        self.load_localiser_model(self.run_name_w)
        self.localiser_bbox(img_fn)
        self.preds=[]
        for n in range(len(self.runs_p)):
            self.load_patch_model(n)
            self.run_patch_prediction(img_fn,n)
            self.preds.append(self.p.pred)

        self.pred = pred_mean(self.preds)
        self.postprocess(self.cc3d)
        self.save_prediction()
        self.unload_case()
# %%

if __name__ =="__main__":
    from fran.utils.common import *
    common_paths_filename=os.environ['FRAN_COMMON_PATHS']
    P = Project(project_title="lits"); proj_defaults= P.proj_summary

    mo_df = pd.read_csv(Path("/media/ub/datasets_bkp/litq/complete_cases/cases_metadata.csv"))
    patch_size = [160,160,160]
    resample_spacings = [1,1,2]
    run_name_w= "LITS-276" # best trial
    # runs_ensemble=["LITS-265","LITS-255","LITS-270","LITS-271","LITS-272"]
    runs_ensemble=["LITS-408","LITS-385","LITS-383","LITS-357"]
    run_name_p = runs_ensemble[0]
    device=1
    En = EnsemblePredictor(proj_defaults,run_name_w,runs_ensemble,device,debug=True)

# %%
    img_fn = Path("/media/ub/datasets_bkp/litq/sitk/images/litq_00060_20190815.nrrd")
    img_fn=Path(img_fn)
    En.run(img_fn)
# %%
    ImageMaskViewer([En.w.img_np_orgres,En.w.pred[1]])
# %%
    E = EndToEndPredictor(proj_defaults,run_name_w,runs_ensemble[0],use_neptune=True,device=device,save_localiser=True)
    E.localiser_bbox(img_fn)
    E.run_patch_prediction(img_fn)
    bboxes = E.bboxes
    ImageMaskViewer([E.predictor_w.img_np_orgres,E.predictor_w.pred_int])
    ImageMaskViewer([E.predictor_w.img_np_orgres[bboxes[0]],E.predictor_w.pred_int[bboxes[0]]])
# %%
    P = E.predictor_p
    ImageMaskViewer([P.img_np_orgres,P.pred_int])
# %%
    dl = P.dls[0][0]
    iteri = iter(dl)
    b = next(iteri)
    ImageMaskViewer([b['image']['data'][0,0],b['image']['data'][0,0]])
# %%
    run_name_p='LITS-265'
    resample_spacings= [1,1,2]
    patch_size = [192, 192, 96]
    img_fn =  Path('/media/ub/datasets_bkp/litq/complete_cases/images/litq_0014389_20190925.nii')
# %%
    P = PatchPredictor(proj_defaults,3,resample_spacings,patch_size,device=device)
    P.load_model(E.predictor_p.model,run_name_p)
    P.run(img_fn,bboxes=bboxes)
# %%
    score = P.score_prediction(img_fn,3)
# %%
# %%
    w = En.w
    w.img_np_orgres.shape
    w = WholeImagePredictor(proj_defaults,2,resample_spacings,[160,160,160],'cuda')
    w.load_model(E.predictor_w.model,'chap')
    w.load_case(img_fn)
    w.run()


    E.predictor_p.make_prediction()
    E.predictor_p.postprocess()
    E.predictor_p.save_prediction()
    delattr(E.predictor_p,'_model')
    torch.cuda.empty_cache()
    E.preds.append(E.predictor_p.pred)
# %%

    E.preds_ensemble = pred_mean(E.preds)
    E.output_image_folder

    E.output_image_folder = "ensemble_"+"_".join(E.runs_p)
# %%
    [p.shape for p in E.preds]
    E.preds[0].shape
# %%
    E.pred_int

    pred_int= torch.argmax(En.p.pred, 0, keepdim=False)
    pred_int=pred_int.to(torch.uint8)
    En.p.pred_int = En.p.retain_k_largest_components(pred_int,En.p.postprocess_label)

# %%







    n = 0
    En.img_filename = img_fn
    En.load_localiser_model(En.run_name_w)
    En.localiser_bbox(img_fn)

    En.load_patch_model(1)
    En.run_patch_prediction(img_fn,n)
    En.preds.append(En.p.pred)
# %%
    p = En.p

    p.backsample()
    p.postprocess()
# %%
    ImageMaskViewer([w.img_np_orgres,pred[0]])
# %%
    pred_int= torch.argmax(w.pred, 0, keepdim=False)
    pred_int=pred_int.to(torch.uint8)
    w = En.w
    w.postprocess_pipeline
    w.Stencil.merge_labels
# %%
    pp(w.postprocess_pipeline)
    pred = w.postprocess_pipeline[0].encodes(w.pred)
    [[a.dtype, a.max()] for a in pred]
    pred = w.postprocess_pipeline[1].encodes(pred)
    pred = w.postprocess_pipeline[2].encodes(pred)
    pred = w.postprocess_pipeline[3].encodes(pred)

    pred = w.postprocess_pipeline[4].encodes(pred)
# %%
    pred = w.postprocess_pipeline[5].encodes(pred)

    En.bboxes = En.w.get_bbox_from_pred(1)
    w.pred_int = w.postprocess_pipeline(w.pred)
    bboxes = w.get_bbox_from_pred(1)
# %%
    pred_int = pred
    stencils = [m.encodes(pred_int) for m in w.Stencil.ms]
# %%
    indx=0
    stencils =   pred 
    stencil = stencils[indx]
    label = indx+1
    info = w.DustKLargest.mask_labels[str(label)]
    st = cc3d.statistics(stencil)
    dusting_threshold = volvox(info['dusting_threshold'],w.DustKLargest.spacings)
    stencil_dusted= cc3d.dust(stencil,threshold=dusting_threshold,connectivity=26,in_place=True)

    stencil_k= cc3d.largest_k(stencil_dusted,k=info['k_largest'], return_N=False)
# %%
    ext='.nii.gz'
    counts = ["","_1","_2","_3","_4"][:len([En._pred_sitk_i,*En.pred_sitk_f])]
    En.pred_fns= [En._output_image_folder/(En.img_filename.name.split(".")[0]+c+ext) for c in counts]
    print("Saving prediction. File name : {}".format(En.pred_fns))
# %%
    En._pred_sitk_i = ArrayToSITKI(sitk_props=En.sitk_props).encodes(En.pred_int)
    for pred_sitk,fn in zip([En._pred_sitk_i]+En.pred_sitk_f,En.pred_fns):
        sitk.WriteImage(pred_sitk,fn)
# %%

        En.postprocess(En.cc3d)
        En.sitk_props = En.p.sitk_props
        En.super().save_prediction()
