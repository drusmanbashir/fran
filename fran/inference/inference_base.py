
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
# %%

if 'get_ipython' in globals():

    from IPython import get_ipython
    ipython = get_ipython()
    # ipython.run_line_magic('load_ext', 'autoreload')
    # ipython.run_line_magic('autoreload', '2') 
    ipython.run_line_magic("load_ext","autoreload")
    ipython.run_line_magic("autoreload",2)
import sys

from fran.inference.helpers import *
sys.path+=["/home/ub/Dropbox/code/fran/"]
from fran.managers.trainer import *
from fran.managers.tune import ModelFromTuneTrial
from fran.transforms.inferencetransforms import *
from fastcore.transform import NoneType, Transform
from fran.utils.imageviewers import *
from skimage.transform import resize

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

class ArrayToSITK(Transform):
    def __init__(self,img_sitk): store_attr()
    def encodes(self,pred_pt):
            pred = sitk.GetImageFromArray(pred_pt)
            pred.SetOrigin(self.img_sitk.GetOrigin())
            pred.SetSpacing(self.img_sitk.GetSpacing())
            return pred

def get_k_organs(mask_labels):
            k_components = [tissue['k_largest'] for tissue in mask_labels if tissue['label']==1][0]
            return k_components


class PatchPredictor(object):
    def __init__(self, proj_defaults,resample_spacings,  patch_size: list = [128,128,128] ,patch_overlap:Union[list,float,int]=0.5, grid_mode="crop",expand_bbox=0.1,k_components=None, batch_size=8,stride=None,device=None,softmax=True,dusting_threshold=10):

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
            patch_overlap = [int(x*patch_overlap) for x in patch_size]
        elif isinstance(patch_overlap,int):
            patch_overlap = [patch_overlap]*3
        self.global_properties = load_dict(global_properties_fname)
        if device is None:
            self.device = get_available_device()

    def load_case(self, img_filename,mask_filename=None, case_id=None, bboxes=None):
        self.unload_previous()
        self.gt_fn = mask_filename
        self.img_filename = img_filename
        self.case_id = get_case_id_from_filename(None,self.img_filename) if not case_id else case_id
        self.img_sitk= sitk.ReadImage(str(self.img_filename))
        self.img_np_orgres=sitk.GetArrayFromImage(self.img_sitk)
        self.size_source = self.img_sitk.GetSize()
        self.sz_dest = get_sitk_target_size_from_spacings(self.img_sitk,self.resample_spacings)
        self.bboxes =bboxes if bboxes else self.set_patchsized_bbox()
        self.create_encode_pipeline()
        self.create_dl_tio()


    def set_patchsized_bbox(self):
        shape = self.img_np_orgres.shape
        slc=[]
        for s in shape:
            slc.append(slice(0,s))
        return ([tuple(slc)])

    def create_encode_pipeline(self):
            T = TransposeSITKToNp()
            R = ResampleToStage0(self.img_sitk,self.resample_spacings)
            B = BBoxesToPatchSize(self.patch_size,self.sz_dest,self.expand_bbox)
            C = ClipCenter(clip_range=self.global_properties['intensity_clip_range'],mean=self.global_properties['mean_fg'],std=self.global_properties['std_fg'])
            self.encode_pipeline = Pipeline([T,R,B,C])

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

    def create_grid_sampler_aggregator(self):
            self.grid_sampler = tio.GridSampler(subject=self.subject,patch_size=self.patch_size,patch_overlap=self.patch_overlap) 
            self.aggregator = tio.inference.GridAggregator(self.grid_sampler,overlap_mode=self.grid_mode) 
            self.patch_loader = torch.utils.data.DataLoader(self.grid_sampler, batch_size=self.batch_size)


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
            # self.patches_to_orgres()


    def post_process(self,save=True):
        self.retain_k_largest_components() # time consuming function
        if save==True:
            self.save_prediction()

    def patches_to_orgres(self):
        x = self.img_transformed, self.bboxes_transformed
        pred_neo = torch.zeros(x[0].shape)
        for bbox,pred_patch in zip(x[1],self.pred_patches):
            pred_neo[bbox]=pred_patch[0]
        BM = BacksampleMask(self.img_sitk)
        pred_orgres= BM(pred_neo)
        self.pred= pred_orgres.permute(2,1,0)

    def save_prediction(self):
        maybe_makedirs(self._output_image_folder)
        self.pred_fn=str(self._output_image_folder/(self.case_id+".nii.gz"))
        print("Saving prediction. File name : {}".format(self.pred_fn))
        sitk.WriteImage(self.pred_sitk,self.pred_fn)

    def score_prediction(self,mask_filename,n_classes):
        if self.gt_fn is None: self.gt_fn = mask_filename
        if self.gt_fn is not None:
            self.gt_fn=str(self.gt_fn)
        else:
            raise TypeError
        mask_sitk = sitk.ReadImage(mask_filename)
        # self.scores = compute_metrics_for_case(self.pred_fn,self.gt_fn)
        self.scores = compute_dice_fran(mask_sitk,self.pred_sitk,n_classes)




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
            
           
    def dump(self):
        # legacy function. destroy after one run of prediction storage
        print ("Returning img_org , image_np, normalized_np, backsampled_pred(predictions).\nLoad model again after this")
        del self._model
        torch.cuda.empty_cache()
        return  self.img_np_orgres, self.image_np, self.backsampled_pred
       
    def retain_k_largest_components(self,axis:int):
        pred_tmp_binary = np.array(self.pred_int[0],dtype=np.uint8)
        if axis==1:
            pred_tmp_binary [pred_tmp_binary >1]= 1
        if axis==2:
            pred_tmp_binary [pred_tmp_binary <2]= 0
            pred_tmp_binary [pred_tmp_binary <1]= 1
        pred_tmp_binary= cc3d.dust(pred_tmp_binary,threshold=self.dusting_threshold,connectivity=26,in_place=True)
        pred_k_largest, N= cc3d.largest_k(pred_tmp_binary,k=self.k_components, return_N=True)
        return pred_k_largest

        

    def save_np_prediction(self,original_res=True, overwrite=False):
        self.prediction_filename_nonbinary=str(self._output_image_folder/(self.proj_defaults.project_title+"_"+self.case_id+".npy"))
        pred = self.backsampled_pred if original_res==True else self.mask_cc
        save_np(pred,self.prediction_filename_nonbinary)
        return self.prediction_filename_nonbinary
 
    def save_binary_prediction(self, prefix=None):
        if not hasattr(self,"pred_binary"): self.create_binary_pred()
        if prefix is None: 
            self.prediction_filename_binary= self._output_image_folder/(self.proj_defaults.project_title+"_"+self.case_id+".nii.gz")
        save_to_nii(self.pred_binary,self.prediction_filename_binary,verbose=True)
        return self.prediction_filename_binary

    def unload_previous(self):
        attributes = ["backsampled_pred", "mask_cc","mask_sitk", "subject"]
        try:
            for att in attributes:
                delattr(self,att)
        except:
            pass


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
            dusted = self.retain_k_largest_components(axis=axis)
            setattr(self,attr_name,dusted)
        return getattr(self,attr_name)


    @property 
    def pred_sitk(self):
        if not hasattr(self,"_pred_sitk"):
            self._pred_sitk = ArrayToSITK(self.img_sitk).encodes(self.pred)
        return self._pred_sitk
    @property
    def pred_int(self):
        if not hasattr(self,"_pred_int"):
            self._pred_int= torch.argmax(self.pred, 0, keepdim=True)
        return self._pred_int
    @property
    def output_image_folder(self):
        return self._output_image_folder

    @output_image_folder.setter
    def output_image_folder(self,folder_name: Path):
        self._output_image_folder =Path(self.proj_defaults.predictions_folder)/folder_name

    @property
    def case(self):
        return self.subject
    @property
    def model(self):
        return self._model


    @property
    def image_np(self):
        return self.case['image']['data']


    @property
    def pred_fn(self):
        return self._pred_fn

    @pred_fn.setter
    def pred_fn(self, value):
        self._pred_fn= value

class WholeImageBBoxes(ApplyBBox):
        def __init__(self, patch_size):
            bboxes=[tuple([slice(0,p) for p in patch_size])]
            super().__init__(bboxes)

        def encodes(self, x):
            img,bboxes = x
            return img,bboxes

        def decodes(self,x) : return x # no processing to do
        


                       
# %%


class WholeImagePredictor(PatchPredictor):
    def __init__(self, proj_defaults, resample_spacings,  patch_size: list = [128,128,128], device=None,**kwargs):
        super().__init__(proj_defaults,resample_spacings,patch_size,patch_overlap=0,batch_size=1,device=device,**kwargs)
 
    def create_encode_pipeline(self):

            T = TransposeSITKToNp()
            Rz = Resize(self.patch_size)
            P  = PadNpArray(patch_size=self.patch_size)
            W = WholeImageBBoxes(self.patch_size)
            C = ClipCenter(clip_range=self.global_properties['intensity_clip_range'],mean=self.global_properties['mean_fg'],
                           std=self.global_properties['std_fg'])
            self.encode_pipeline = Pipeline([T,Rz,P,W,C])

    @property 
    def pred(self): return self.pred_patches[0]

class EndToEndPredictor(object):
    def __init__(self, proj_defaults,run_name_w,run_name_p,use_neptune=False,patch_overlap=0.5,device='cuda', save_localiser=False):
        print("Loading model checkpoints for whole image predictor")
        if use_neptune==True:
            Nep = NeptuneManager(proj_defaults)
            model_w, patch_size_w,resample_spacings_w,_= self.load_model_neptune(Nep,run_name_w,device=device)
            model_p, patch_size_p, resample_spacings_p,self.n_classes= self.load_model_neptune(Nep,run_name_p,device=device)
        else:
            print("Not fully implemented")
            P = ModelFromTuneTrial(proj_defaults, trial_name=run_name_w,out_channels= 2)
            model_w = P.model
            patch_size_w= make_patch_size(P.params_dict['dataset_params']['patch_dim0'], P.params_dict['dataset_params']['patch_dim1'])


        self.predictor_w = WholeImagePredictor(proj_defaults=proj_defaults,resample_spacings=resample_spacings_w, patch_size=patch_size_w,device=device)
        self.predictor_w.load_model(model_w,model_id=run_name_w)
        self.save_localiser = save_localiser
        print("\nLoading model checkpoints for patch-based predictor")


        patch_overlap = [int(x*patch_overlap) for x in patch_size_p]
        self.predictor_p= PatchPredictor(proj_defaults=proj_defaults, resample_spacings=resample_spacings_p,patch_size=patch_size_p,patch_overlap=patch_overlap, 
                                            stride = [1,1,1], batch_size=4,device=device)
        self.predictor_p.load_model(model_p,model_id=run_name_p)

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
        self.run_patch_prediction()

    def get_localiser_bbox(self,img_fn,mask_fn=None):
        print("Running predictions. Whole image predictor is on device {0}. Patch-based predictor is on device {1}".format(self.predictor_w.device,self.predictor_p.device))
        self.predictor_w.load_case(img_filename=img_fn,mask_filename= mask_fn)
        self.predictor_w.make_prediction()
        if self.save_localiser==True: self.predictor_w.save_prediction()
        self.bboxes = self.predictor_w.get_bbox_from_pred()

    def score_prediction(self,mask_fn):
        self.scores = self.predictor_p.score_prediction(mask_fn,self.n_classes)


    def run_patch_prediction(self):
        self.predictor_p.load_case(img_filename=img_fn,mask_filename=mask_fn,bboxes=self.bboxes)
        self.predictor_p.make_prediction()


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
    def pred_sitk(self): return self.predictor_p.pred_sitk

class SubsampleInferenceVersion(Transform):  # TRAINING TRANSFORM
    # Use for any number of dims as long as sample factor matches
    def __init__(self, sample_factor=[1, 2, 2],dim=3):
        store_attr()
    def encodes(self, x):
        slices = []
        for dim,stride in zip(x.shape,self.sample_factor):
            slices.append(slice(0,dim,stride))
        return x[tuple(slices)]
    def decodes(self,x):
        upsample_factor = self.sample_factor[-self.dim:]
        x = F.interpolate(x,scale_factor=upsample_factor,mode='trilinear')
        return x

def create_prediction_from_imagefilename(proj_defaults,predictor, filename_nii,save=True):
       project_title = proj_defaults.project_title
       mask_= filename_nii
       
       if project_title=="kits19":
           img_=filename_nii.replace(".nii","_0000.nii")
           proj_defaults=  proj_defaults
       else:
           img_= filename_nii
           proj_defaults=  proj_defaults
       img_filename = proj_defaults.raw_data_folder/("images"+"/"+img_)
       mask_filename = proj_defaults.raw_data_folder/("masks"+"/"+mask_)
       print("Files exist {},{}".format(img_filename.exists(), mask_filename.exists()))
       predictor.load_case(img_filename,mask_filename=mask_filename)
       predictor.make_prediction(save=save)
       predictor.create_binary_pred(save=save)

# %%
# gui2([input_tensor[0].cpu().detach().numpy(),output_tensor[0].cpu().numpy()],shared_slider=True)
if __name__ =="__main__":

    common_paths_filename=os.environ['FRAN_COMMON_PATHS']
    P = Project(project_title="lits"); proj_defaults= P.proj_summary

