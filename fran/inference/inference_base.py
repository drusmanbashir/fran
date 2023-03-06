
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
    def __init__(self, proj_defaults,resample_spacings,  patch_size: list = [128,128,128] ,patch_overlap:Union[list,float,int]=0.5, grid_mode="crop",expand_bbox=0.1,k_components=None, batch_size=8,stride=None,device=None,softmax=False):

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
        self.case_id = get_case_id_from_filename(self.proj_defaults.project_title,self.img_filename) if not case_id else case_id
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

    def make_prediction(self,save=True):
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
            max_idx = torch.argmax(output_tensor, 0, keepdim=True)
            self.pred_patches.append(max_idx)
        self.patches_to_orgres()
        self.retain_k_largest_components() # time consuming function
        if save==True:
            self.pred_sitk = ArrayToSITK(self.img_sitk).encodes(self.pred_final)
            self.save_prediction()

    def patches_to_orgres(self):
        x = self.img_transformed, self.bboxes_transformed
        pred_neo = torch.zeros(x[0].shape)
        for bbox,pred_patch in zip(x[1],self.pred_patches):
            pred_neo[bbox]=pred_patch[0]
        BM = BacksampleMask(self.img_sitk)
        pred_orgres= BM(pred_neo)
        pred_orgres = pred_orgres.permute(2,1,0)
        self.pred_orgres= pred_orgres.numpy()
    #
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
       
    def retain_k_largest_components(self):
        self.pred_final = self.pred_orgres.copy()
        pred_tmp_binary = self.pred_orgres.copy()
        pred_tmp_binary [pred_tmp_binary >1]= 1
        pred_tmp_binary = pred_tmp_binary.astype(int)
        self.pred_k_largest, N= cc3d.largest_k(pred_tmp_binary,k=self.k_components, return_N=True)
        self.pred_final[self.pred_k_largest==0]=0
        
        

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
        stats = cc3d.statistics(self.pred_k_largest.astype(np.uint8))
        bboxes = stats['bounding_boxes'][1:] # bbox 0 is the whole image
        bboxes
        if len(bboxes)<1:
            tr()
        return bboxes


    @property
    def mask(self):
        if not hasattr(self,"mask_cc"):
            self.create_binary_pred(save=False)
        return self.mask_cc
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
class WholeImageBBoxes(ItemTransform):

        def __init__(self, patch_size):
            store_attr()

        def encodes(self, x):
            img,bboxes = x
            bboxes=[tuple([slice(0,p) for p in self.patch_size])]
            return img,bboxes
# %%
                       
# %%


class WholeImagePredictor(PatchPredictor):
    def __init__(self, proj_defaults, resample_spacings,  patch_size: list = [128,128,128], device=None,**kwargs):
        super().__init__(proj_defaults,resample_spacings,patch_size,patch_overlap=0,batch_size=1,device=device,**kwargs)
 
    def create_encode_pipeline(self):

            T = TransposeSITKToNp()
            Rz = ResizeNP(self.patch_size)
            P  =  PadNpArray(patch_size=self.patch_size)
            W = WholeImageBBoxes(self.patch_size)
            C = ClipCenter(clip_range=self.global_properties['intensity_clip_range'],mean=self.global_properties['mean_fg'],
                           std=self.global_properties['std_fg'])
            self.encode_pipeline = Pipeline([T,Rz,P,W,C])


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
        self.predictor_w.make_prediction(save=self.save_localiser)
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
    def pred_final(self): return self.predictor_p.pred_final

    @property
    def pred_sitk(self): return self.predictor_p.pred_sitk_

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

    train_list,valid_list,test_list = get_train_valid_test_lists_from_json(project_title=proj_defaults.project_title,fold=0, json_fname=proj_defaults.validation_folds_filename,image_folder=proj_defaults.raw_data_folder/"images",ext=".nii")

    preprocessed_dataset_properties_fname = list(proj_defaults.stage0_folder.glob("*json"))[0]

    preprocessed_dataset_properties_21 = load_dict(proj_defaults.raw_dataset_properties_filename)
    mask_files=[ proj_defaults.raw_data_folder/"masks"/test_file.name.replace(".npy",".nii.gz") for test_file in test_list]
    img_files =  [proj_defaults.raw_data_folder/"images"/mask_file.name for mask_file in mask_files]


    case_id = "203"
    mask_filename= [case_ for case_ in mask_files if case_id in str(case_id)][0]
# %%
    ######################################################################################
# %% [markdown]
## All predictions from valid_list
# %%
    # 3d
    model_id = "KITS-1672"
    M =NeptuneManager(proj_defaults)
    M.load_run(run_name=model_id,nep_mode='read-only',param_names='default')
    df = M.config_dict
# %%
    model = create_model_from_conf  (df['model_params'])
    # M._model = UNet3D proj_defaults.raw_data_folder/"masks"/valid_file.name.replace(chs= (1, 32, 64, 128, 256), num_classes = 2)

    load_checkpoint(M.checkpoints_folder,model)
# %%
    new_subfolder=True
    aa = df['dataset_params']['patch_size']
    patch_size= df["dataset_params"]["patch_size"]
    patch_overlap = [int(x/2) for x in patch_size]
    # output_image_subfolder = "tumour_segmentation"
# %%
    p= WholeImagePredictor(proj_defaults=proj_defaults, patch_size=patch_size)
    p.load_model(model,model_id=M.run_name)
# %%


    n=35
    n=30 # hard
    p.load_case(img_filename=img_files[n])


# %%
    p.make_prediction()
    p.create_binary_pred(threshold=0.2)
# %%

    mask_filename= [case_ for case_ in mask_files if p.case_id in str(case_)][0]
    p.mask_sitk = sitk.ReadImage(str(mask_filename))
    p.score_prediction()
# %%
    threshold = 0.2
    dusting_threshold=100
    j = torch.clone(p.backsampled_pred)
    j = F.softmax(j,0)
    j = j[1]
    mask_binary = np.zeros(j.shape, dtype=bool)
    mask_binary[torch.where(j>threshold)]=1
    mask_binary= cc3d.dust(mask_binary,threshold=dusting_threshold,connectivity=26,in_place=True)
    p.mask_cc, N= cc3d.largest_k(mask_binary,k=2, return_N=True)
    p.mask_cc[p.mask_cc>0]=1

# %%
    i = torch.tensor(p.img_org).permute(2,1,0)
    j = torch.tensor(p.mask_cc.astype(np.uint8)).permute(2,1,0)
# %%

    ImageMaskViewer([i,j])
# %%
    p.view_predictions('binary',view_channel=1,orientation='A')
    p.view_predictions('softmax',view_channel=1,orientation='A')
    p.view_predictions('raw',view_channel=1,orientation='A')

# %%
######################################################################################
# %% [markdown]
## Trying a different run
    
# %%
# %%
    model_id = "KITS-383"
    M =NeptuneManager(run_name=model_id,mode="read-only")
    M.new_run()
    df = M.run_dict
    df['model_params']['layer_channels']= list(ast.literal_eval(df['model_params']['layer_channels']))
# %%
    M.create_model(arch = df['model_params']['architecture'])
    # M._model = UNet3D proj_defaults.raw_data_folder/"masks"/valid_file.name.replace(chs= (1, 32, 64, 128, 256), num_classes = 2)

    M.load_checkpoint()
# %%
    new_subfolder=True
    patch_size= ast.literal_eval(df["dataset_params"]["patch_size"])
    stride =ast.literal_eval(df['dataset_params']['stride'])
    patch_overlap = [int(x/2) for x in patch_size]
    output_image_subfolder = "tumour_segmentation"
    num_slices =ast.literal_eval(df['model_params']['layer_channels'])[0]
# %%
    p= WholeImagePredictor(proj_defaults=proj_defaults, patch_size=patch_size)
    p.load_model(M.model,model_id=M.run_name)
# %%



    n=35
    n=30 # hard
    p.load_case(img_filename=img_files[n])


# %%
    p.make_prediction()
    p.create_binary_pred(threshold=0.2)
# %%

    mask_filename= [case_ for case_ in mask_files if p.case_id in str(case_)][0]
    p.mask_sitk = sitk.ReadImage(str(mask_filename))
    p.score_prediction()


# %%
    ######################################################################################
    # %% [markdown]
    ## Patch-based predictors
    # %%
    
    model_id = "KITS-327"  # latest kidney model
    df1 = M.nep_run.fetch()
    M1 =NeptuneManager(run_name=model_id,mode="read-only")
    df1 = M1.nep_run.fetch()
# %%
    M1.create_model(arch = df1['model_params']['architecture'])
    # M1._model = UNet3D proj_defaults.raw_data_folder/"masks"/valid_file.name.replace(chs= (1, 32, 64, 128, 256), num_classes = 2)

    M1.load_checkpoint()

# %%
    M1.create_model(arch = df1['model_params']['architecture'])
    # M1._model = UNet3D proj_defaults.raw_data_folder/"masks"/valid_file.name.replace(chs= (1, 32, 64, 128, 256), num_classes = 2)

    M1.load_checkpoint()
# %%
    new_subfolder=True
    patch_size= ast.literal_eval(df["dataset_params"]["patch_size"])
    stride =ast.literal_eval(df['dataset_params']['stride'])
    patch_overlap = [int(x/2) for x in patch_size]
    output_image_subfolder = "tumour_segmentation"
    num_slices =ast.literal_eval(df['model_params']['layer_channels'])[0]

    p1= PatchBasedPredictor(proj_defaults=proj_defaults, patch_size=patch_size,patch_overlap=patch_overlap, 
                                       bboxes = None ,stride = stride, batch_size=4)
# %%
    p1.load_model(M.model,model_id=M.run_name)
    p1.load_case(img_filename=img_files[n])

# %%
    p1.make_prediction(save=False)
    p1.create_binary_pred(threshold=0.2)

# %%
# %%
    mask_filename= [case_ for case_ in mask_files if p1.case_id in str(case_)][0]
    p1.mask_sitk = sitk.ReadImage(str(mask_filename))
    p1.score_prediction()

# %%

    p1.view_predictions('softmax')
# %%
######################################################################################
# %% [markdown]
##  Comparing WholeImagePredictor with PatchBasedPredictor

# %%

    p.view_predictions(False)
    bbox = p.get_bbox_from_binarymask()[0]

    im = p.img_org
    im2 = im.transpose(2,0,1)

    ImageMaskViewer([im2,im2[bbox]])
# %%

    bbox1 = p1.get_bbox_from_binarymask()
# %%

    # p1.case_id = get_case_id_from_filename(p1.proj_defaults.project_title,p1.img_filename)
# %%
    mask_filename = str(mask_filename).replace("203",p1.case_id)
    p1.mask_sitk = sitk.ReadImage(str(mask_filename))

    p1.score_prediction()
    p1.view_predictions(True)
# %%
    p1.size_source = p1.img_sitk.GetSize()
    sz_source, spacing_source = p1.img_sitk.GetSize(), p1.img_sitk.GetSpacing()
    np_array=sitk.GetArrayFromImage(p1.img_sitk).transpose(2,0,1) # this transpose reverses the transpose done by sitk->np to apply correct interpolation factors
    p1.sz_source = np_array.shape
    sz_dest, scale_factor = get_scale_factor_from_spacings(np_array.shape,spacing_source,p1.resample_spacings)
    np_resized = resize(np_array, sz_dest,order=None).transpose(0,2,1) # this transpose makes images upright for viewing

# %%

# %%
    image_normed= clip_image(np_resized,clip_range=p1.global_properties['intensity_clip_range'])
    image_normed = standardize(img,mn=p1.global_properties['mean_fg'], std=p1.global_properties['std_fg'])
    #,mean=p1.global_properties['mean_fg'],std=p1.global_properties['std_dataset_clipped']).astype(np.float32)
# %%
    p1.P = PadDeficitNpArray(patch_size=p1.patch_size,return_padding=True)
    image_normed_padded,_,p1.padding= p1.P.encodes([image_normed,image_normed])
    image_normed_padded = np.expand_dims(image_normed_padded,0)
    normalized_tio = tio.ScalarImage(tensor=image_normed_padded)
    p1.subject = tio.Subject(image=normalized_tio)
# %%
    load_dict()

# # %%
#         patch_loader = torch.utils.data.DataLoader(p1.grid_sampler, batch_size=p1.batch_size)
#
#         with torch.no_grad():
#             for i,patches_batch in enumerate(patch_loader):
#                 input_tensor = patches_batch['image'][tio.DATA].to(p1.device)
#                 locations = patches_batch[tio.LOCATION]
#                 input_tensor ,_= p1.apply_transforms('encode', [input_tensor,torch.zeros(input_tensor.shape)])
#                 y =p1._model(input_tensor.float())
#                 y ,_= p1.apply_transforms('decode',[y,torch.zeros(y.shape)])
#                 p1.aggregator.add_batch(y, locations)
#         p1.output_tensor = p1.aggregator.get_output_tensor()
#         p1.output_tensor = torch.nan_to_num(p1.output_tensor,0)
#
#         tensor_bboxed = torch.zeros((len(tmp_tensors),)+p1.size_source)
#         tmp_tensors = []
#         for tensr in p1.output_tensor:
#             unpadded,_ = p1.P.decodes([tensr,tensr],p1.padding)
#             tmp_tensors.append(unpadded)
#         p1.output_tensor_unpadded=torch.stack(tmp_tensors)
#
#         p1.output_tensor_unpadded= p1.output_tensor_unpadded.permute(0,1,3,2)
#
#         p1.backsampled_pred = F.interpolate(input= p1.output_tensor_unpadded.unsqueeze(0),size=p1.sz_source)
#         p1.backsampled_pred = p1.backsampled_pred[0] # no need for bach dim in single image
#
#
#
# # %%
# # %%
#     ImageMaskViewer([p1.mask_cc,p1.mask_cc])
#     ImageMaskViewer([image_normed_padded,image_normed_padded])
# # %%
#     p1.P = PadDeficitNpArray(patch_size=p1.patch_size,return_padding=True)
#     image_normed_padded,_,p1.padding= p1.Ax encapsulates both the surrogateP.encodes([image_normed,image_normed])
#     image_normed_padded = np.expand_dims(image_normed_padded,0)
#     normalized_tio = tio.ScalarImage(tensor=image_normed_padded)
#     p1.subject = tio.Subject(image=normalized_tio)
# # %%
#     p1.create_grid_sampler_aggregator()
#
# # %%
#     p1.make_prediction(save=False)
# # %%
#     p1.view_predictions(binary =False)
# # %%
#     p1.create_binary_pred()
# #
#
#         p1.output_tensor = p1.aggregator.get_output_tensor()
#         p1.output_tensor = torch.nan_to_num(p1.output_tensor,0)
#
#         tmp_tensors = []
#         for tensr in p1.output_tensor:
#             unpadded,_ = p1.P.decodes([tensr,tensr],p1.padding)
#             tmp_tensors.append(unpadded)
#         p1.output_tensor_unpadded=torch.stack(tmp_tensors)
#         p1.backsampled_pred = F.interpolate(input= p1.output_tensor_unpadded.unsqueeze(0),size=p1.size_source)
#         p1.backsampled_pred = p1.backsampled_pred[0] # no need for bach dim in single image
#
# # %%
#
#
# # %%
#     pbar = get_pbar()
#     valid_list_small = valid_list[:20]
#     for n in pbar(range(len(valid_list_small))):
#         filename_npy_= valid_list_small[n].name.replace(".npy",".nii.gz")
#         create_prediction_from_imagefilename(proj_defaults,p1,str(filename_npy_))
# # %%
#         def score_maskfile(proj_defaults,prediction_folder, mask_filename: str):
#            '''
#            mask_filename is the filename alone
#            '''
#            mask_filename = proj_defaults.raw_data_folder/("masks"+"/"+mask_filename)
#            prediction_filename = prediction_folder+"/" + mask_filename.name.lower()
#            if Path(prediction_filename).exists():
#                score = compute_metrics_for_case(str(prediction_filename),str(mask_filename)) # %%
#                return {"prediction_filename": prediction_filename, "score":score}
#            else:
#                 print("Prediction file {0} corresponding to gt file{1} does not exist. Skipping..".format(prediction_filename,mask_filename))
#                 return []
#
#
# # %%
#     prediction_folder = '/home/ub/datasets/predictions/kits21/segmentations_KITS-289_150422_2204'
#     args = [[proj_defaults,prediction_folder,mask.name] for mask in mask_files]
#     scores = multiprocess_multiarg(score_maskfile,args, debug=False)
#     s = np.array([sc['score'] for sc in scores if isinstance(sc,dict)])
#     np.median(s,0)
#     s.mean(0)
#     s[:,0,0]
#     s[:,0,0].min()
#     P_19 = SingleCasePredictor(proj_defaults=proj_defaults, patch_size=patch_size, patch_overlap=.5,
#                                batch_size=4)
#
#     P_19.load_model(M.model, model_id=M.run_name)
#     P_19.load_case(img_filename=img_files[n], mask_filename=mask_files[n])
#     # %%
#     P_19.make_prediction()
# # %%
#     P_21= SingleCasePredictor(proj_defaults=proj_defaults, patch_size=patch_size,patch_overlap=patch_overlap,  batch_size=4)
#
#     n=15
# # %%
#     P_21.load_model(M.model,model_id=M.run_name)
#     P_21.load_case(img_filename=img_files[n],mask_filename=mask_files[n])
# # %%
#
#     model_id = "most_recent"  # latest kidney model
#     M =NeptuneManager(run_name=model_id,mode="read-only")
#     df = M.nep_run.fetch()
# # %%
#     M.create_model(arch = df['model_params']['architecture'])
#     # M._model = UNet3D proj_defaults.raw_data_folder/"masks"/valid_file.name.replace(chs= (1, 32, 64, 128, 256), num_classes = 2)
#
#     M.load_checkpoint()
# # %%
#     patch_size= ast.literal_eval(df["dataset_params"]["patch_size"])
#     patch_overlap = [0]+[int(x/2) for x in patch_size]
#     output_image_subfolder = "tumour_segmentation"
#     num_slices =ast.literal_eval(df['model_params']['layer_channels'])[0]

 # %%
