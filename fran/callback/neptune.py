# %%
from fastai.callback.tracker import TrackerCallback

from fran.utils.common import *
from paramiko import SSHClient
import torch.nn.functional as F
import os
from pathlib import Path
from fastai.callback.core import Callback
from fastcore.basics import store_attr
from neptune.types import File
from torchvision.utils import make_grid
import torch
from fran.transforms.spatialtransforms import one_hot
from fran.managers.base import make_patch_size
from fran.utils.fileio import load_json, load_yaml, maybe_makedirs
import neptune
from neptune.utils import stringify_unsupported
import ast
from fastai.learner import *
from fran.utils.config_parsers import *
# from fran.managers.learner_plus import *
from fran.utils.helpers import *
from fran.callback.neptune import *
from fran.callback.tune import *
from fran.utils.config_parsers import *
from torchinfo import summary

try:
    hpc_settings_fn = os.environ['HPC_SETTINGS']
except: pass


_ast_keys= ['dataset_params,patch_size','metadata,src_dest_labels' ]
_immutable_keys =['fold'] # once set in a particular runs these will not be changed without corrupting the run
str_to_key = lambda string: string.split(',')

def dictionary_fix_ast(dictionary:dict):
    for keys in map(str_to_key,_ast_keys):
        val = dictionary[keys[0]][keys[1]]
        dictionary[keys[0]][keys[1]]= ast.literal_eval(val)
    return dictionary


def get_neptune_config(proj_defaults):
    '''
    Returns particular project workspace
    '''
    project_title =proj_defaults.project_title
    commons= load_yaml(common_paths_filename)
    config_json= Path(commons['neptune_folder'])/("config.json")
    neptune_project_info = load_json(config_json)
    project_name = f"{neptune_project_info['workspace-name']}/{project_title}"
    api_token = neptune_project_info['api_token']
    return project_name, api_token


def get_neptune_project(proj_defaults, mode):
    '''
    Returns project instance based on project title
    '''
   
    project_name, api_token = get_neptune_config(proj_defaults)
    return neptune.init_project(project=project_name, api_token=api_token, mode=mode)


class NeptuneManager():
    def __init__(self, proj_defaults):
        '''
        '''
        store_attr()
        project_name, api_token = get_neptune_config(proj_defaults)
        os.environ['NEPTUNE_API_TOKEN']= api_token
        os.environ['NEPTUNE_PROJECT']= project_name
        self.df = self.fetch_project_df()

        #start a run or not?

    def new_run(self, config_dict,run_name=None):
        self.config_dict = config_dict
        self.nep_run = neptune.init_run( mode="async") # Dont set a name here . It is bugged
        self._upload_config_dict(config_dict)
        self._additional_init_settings(run_name)

        print("This run id is: {}".format(self.run_name))
 
    def load_run(self, run_name, param_names='default', nep_mode="async",update_nep_run_from_config:dict=None):

            '''

            :param run_name: 
                If a legit name is passed it will be loaded. 
                If an illegal run-name is passed, throws an exception
                If most_recent is passed, most recent run  is loaded.
                   
            :param update_nep_run_from_config: This is a dictionary which can be uploaded on Neptune to alter the parameters of the existing model and track new parameters
            '''
            run_id,msg = self.get_run_id(run_name)
            print ("{}. Loading".format(msg))
            self.nep_run = neptune.init_run(with_id=run_id,
                                            mode=nep_mode)

            if isinstance(update_nep_run_from_config,dict):
                self.update_run_from_config(update_nep_run_from_config)

    def update_run_from_config (self,update_nep_run_from_config):
        for category, dict in update_nep_run_from_config.items():
            for key,value in dict.items():
                if all ([key !=k for k in _immutable_keys]):
                    old_val = self.nep_run[category][key].fetch()
                    if str(old_val) != str(value):
                        print("Updating nep-run {0}/{1} from config provided. Previous value: {2}. New value {3}".format(category,key,old_val,value))
                        self.nep_run[category][key]=stringify_unsupported(value)

       
    def get_run_id(self,run_name):
            if run_name == "most_recent":
                run_id, run_name = self.id_most_recent()
                msg = "Most recent run"
            elif run_name is any(['',None]): 
                raise Exception("Illegal run name: {}. No ids exist with this name".format(run_name))
            else:
                run_id = self.id_from_name(run_name)
                msg = "Run id matching {}".format(run_name)
            return run_id, msg


    def id_from_name(self,run_name ):
        row =self.df.loc[self.df['metadata/run_name']==run_name]
        try:

            print("Existing Run found. Run id {}".format(row['sys/id'].item()))
            return row['sys/id'].item()
        except Exception as e:
               print("No run with that name exists .. {}".format(e)) 

    def id_most_recent(self):
        self.df = self.df.sort_values(by="sys/creation_time", ascending=False)
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            if self._has_checkpoints(row):
                print("Loading most recent run. Run id {}".format(row['sys/id']))
                return row['sys/id'], row['metadata/run_name']

    def _upload_config_dict(self, config_dict):
        for key, value in config_dict.items():
            self.set_run_value(key,value)
            setattr(self, key, value)

    def _additional_init_settings(self,run_name):
        '''
        optionally give a personalized run_name to the new run
        '''
        
        if run_name==None:
            self.nep_run['metadata/run_name']=self.fetch('sys/id')
        else:
            self.nep_run['metadata/run_name'] = run_name
        self.nep_run["model_params/epoch"] = 0
        self.nep_run["metadata/best_loss"] = 0e20

    def download_run_params(self, param_names='default'):
            config_dict = {}
            if param_names == 'default':
                param_names = 'metadata', 'model_params', 'dataset_params', 'after_item_intensity', 'after_item_spatial', 'after_batch_affine', 'loss_params'
            # else:
            #     raise NotImplementedError("Only 'default' params are supported for now.")
            nep_df = self.nep_run.get_structure()
            for param in param_names:
                if param in nep_df.keys():
                    neptune_dict = self.nep_run[param].fetch()
                    config_dict.update({param: parse_neptune_dict(neptune_dict)})

            return config_dict
    def set_run_value(self,key,value):
        value = stringify_unsupported(value)
        print("Setting Neptune field {0} value: {1}".format(key,value))
        self.nep_run[key]= value
        self.nep_run.wait()



    def fetch_project_df(self,columns=None):
        print("Downloading runs history as dataframe")
        project_tmp = get_neptune_project(self.proj_defaults, 'read-only')
        df = project_tmp.fetch_runs_table(columns = columns).to_pandas()
        return df

    

    def update_logs(self):
        # self.model_params['summary']= str(self.summary())
        self.nep_run["model_params"] = self.learn.model_params
        self.nep_run["dataset_params"] = self.learn.dataset_params

    def fetch(self,key:str):
        val = self.nep_run[key].fetch()
        self.nep_run.wait()
        return val

    def _has_checkpoints(self, row):
        if isinstance(row['metadata/model_dir'], str):
            foldr_str = Path(row['metadata/model_dir'])
            fnames = list(foldr_str.glob("*model*"))
            if len(fnames) > -1: return True
        return False

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def run_dict(self):
        _run_dict = self.nep_run.fetch()
        self.nep_run.wait()
        
        _run_dict = dictionary_fix_ast(_run_dict)
        return _run_dict

    @property
    def run_name(self):
        if not hasattr(self, "_run_name"):
            self._run_name = self.run_dict['sys']['id']
        return self._run_name

    @run_name.setter
    def run_name(self, run_name):
        self._run_name = run_name

    def stop(self):
        self.nep_run.stop()


def normalize(tensr,intensity_percentiles=[0.,1.]):
        tensr = (tensr-tensr.min())/(tensr.max()-tensr.min())
        tensr = tensr.to('cpu',dtype=torch.float32)
        qtiles = torch.quantile(tensr, q=torch.tensor(intensity_percentiles))
        
        vmin = qtiles[0]
        vmax = qtiles[1]
        tensr[tensr<vmin]=vmin
        tensr[tensr>vmax] = vmax
        return tensr

class NeptuneCallback(NeptuneManager, Callback):

    order = TrackerCallback.order+1
    def __init__(self,proj_defaults,config_dict,run_name=None,nep_run=None, freq=2,metrics=None,hyperparameters=None,tmp_folder="/tmp"):

        super().__init__(proj_defaults)
        if nep_run == None:
            super().new_run(config_dict, run_name)
        else: self.nep_run = nep_run
        self.freq=freq
        self.tmp_folder =tmp_folder

        if hyperparameters==None:
            self.hyperparameters = ['lr','mom','wd']

        self.ignore_losses =['loss_dice_batch']
    def partial_str_match(self,x:str,y:list): 
        if any ([a in x for a in y]): return True
        return False
    def after_create(self):
        # if self.run_name is not None:
        #         self.run_id = self.id_from_name()
        #         self.nep_run = self.load_run()
        #         
        # else:
        #     self.nep_run = self._init_run()
        self.learn.nep_run = self.nep_run
        self.learn.epoch_running_total=self.learn.nep_run["model_params/epoch"].fetch()
        # self.learn.nep_run['metadata/run_name']=self.run_name  # not sure wh/besty but metadata/run_name changes to default by itself around this stage
        self.learn.run_name = self.run_name
        if not self.nep_run.exists('model_params/summary'):
            self.nep_run['model_params/summary'].upload(self.create_summary())

    def after_batch(self):
        if self.iter%self.freq==0:
            if self.training:
                for hp in self.hyperparameters:
                    self.learn.nep_run['hyperparameters/{}'.format(hp)].log( self.opt.hypers[0][hp])
                for key, val in self.learn.loss_func.loss_dict.items():
                    if not self.partial_str_match(key,self.ignore_losses):
                        self.learn.nep_run['metrics/train_loss/'+key].log(val)
            else:
                for key, val in self.learn.loss_func.loss_dict.items():
                    if not self.partial_str_match(key,self.ignore_losses):
                        self.learn.nep_run['metrics/valid_loss/'+key].log(val)

    def after_epoch(self):

        self.learn.epoch_running_total+=1
        self.learn.nep_run['model_params/epoch']=self.learn.epoch_running_total
        self.learn.nep_run['model_params/lr'] = self.learn.opt.hypers[0]['lr']

    def after_fit(self):
        self.learn.nep_run.stop()

    def create_summary(self):
        try:
             print("Creating model summary. Please wait ..")
             patch_size=make_patch_size(self.config_dict['dataset_params']['patch_dim0'],self.config_dict['dataset_params']['patch_dim1'])
             summ = summary(self.learn.model, input_size=tuple([1,1]+patch_size),col_names=["input_size","output_size","kernel_size"],depth=4, verbose=0,device='cpu')
             tmp_filename = self.tmp_folder+"/summary.txt"
             with open (tmp_filename,"w") as f:
                    f.write(str(summ))
             return tmp_filename
             print("Done.")
        except:
                print ("Error encountered in creating model summary")
                return 'emtpy.txt'


    @classmethod

    def from_existing_run(cls,proj_defaults, config_dict, run_name, nep_run,**kwargs):
         cls = cls(proj_defaults,config_dict,run_name=run_name,nep_run=nep_run,**kwargs)
         return cls


class NeptuneCheckpointCallback(TrackerCallback, NeptuneManager):
    "A `TrackerCallback` that saves the model's best during training and loads it at the end."
    order = NeptuneCallback.order+1
    def __init__(self, checkpoints_parent_folder, fname='model',monitor='valid_loss', comp=None, min_delta=0.,   every_epoch=False, at_end=False,
                 with_opt=True, reset_on_fit=False,resume_epoch:int=None,raytune_trial_name=None,keep_last=5):
        store_attr()
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
        assert not (every_epoch and at_end), "every_epoch and at_end cannot both be set to True"
        self.last_saved_path = None

    def download_remote_folder(self, hpc, remote_dir):

        print("\nSSH to remote folder {}".format(remote_dir))
        client = SSHClient()
        client.load_system_host_keys()
        client.connect(hpc['host'], username=hpc['username'], password=hpc['password'])
        ftp_client = client.open_sftp()


        local_dir =self.checkpoints_parent_folder/self.run_name
        fnames = ftp_client.listdir(remote_dir)
        remote_fnames =[os.path.join(remote_dir,f) for f in fnames]
        local_fnames =[os.path.join(local_dir,f) for f in fnames]
        maybe_makedirs(local_dir)
        for rem,loc in zip(remote_fnames,local_fnames):
            print("Copying file {0} to local folder {1}".format(rem,local_dir))
            ftp_client.get(rem, loc)
        self.set_run_value('metadata/model_dir',local_dir)



    def set_model_dir(self,model_dir:str):
    # model_dir = '/data/scratch/mpx588/fran_storage/checkpoints/lits/LITS-413'


        hpc = load_yaml(hpc_settings_fn)
        if hpc['hpc_storage'] in model_dir:
                self.download_remote_folder(hpc,model_dir)

        self.learn.model_dir=Path(self.nep_run['metadata/model_dir'].fetch())

    def after_create(self):
        # keep track of file path for loggers
        # if self.learn.nep_run.exists('metadata/best_loss'):
        #     self.best = float(self.learn.nep_run['metadata/best_loss'].fetch())
        # else:
        #     self.learn.nep_run['metadata/best_loss']=self.best
        #     self.learn.nep_run.wait()
        if self.nep_run.exists('metadata/model_dir'):
            model_dir = self.nep_run['metadata/model_dir'].fetch()
            self.nep_run.wait()
            self.set_model_dir(model_dir)
            self._load_model()
        else:
            if not hasattr(self.learn,'tune_checkpoint'):
                self.learn.model_dir = self.checkpoints_parent_folder / self.learn.run_name
            self.learn.nep_run['metadata/model_dir']=str(self.learn.model_dir)
        maybe_makedirs(self.learn.model_dir)

    def after_epoch(self):
        "Compare the value monitored to its best score and save if best."
        if self.every_epoch:
            if (self.epoch%self.every_epoch) == 0: self._save()
        else: #every improvement
            super().after_epoch()
            if self.new_best:
                print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.')
                self._save()
        self.delete_old_checkpoints()

    def after_fit(self, **kwargs):
        "Load the best model."
        if self.at_end: self._save()
        elif not self.every_epoch: # if a resume_epoch number has been set
            print("Loading suitable model at end of fit.")
            self._load_model()

    

    def _load_model(self):
        if self.resume_epoch:
            assert(isinstance(self.resume_epoch,int)), "resume_epoch should be int type but was provided as {}".format(type(self.resume_epoch))
            fname = self._get_epoch_checkpoint()
        else:
            fname = self._get_latest_checkpoint()

        if fname:
            fname = fname.parent/fname.name.replace(".pth","")
            self.learn.load(fname)
            print("Successfully loaded model from checkpoint {} .".format(fname.name))

        else:
                print("Training with new initialization.")

    def _get_latest_checkpoint(self):
        fname = self.learn.model_dir.glob("*")
        try:
            return max(fname,key=os.path.getctime)
        except:
            return None
    def _get_epoch_checkpoint(self):
        fname = "_".join([self.fname,str(self.resume_epoch)])
        if Path(fname).exists(): return fname
        else: 
            print("{0} does not exist in folder {1}".format(fname,self.learn.model_dir/("model_checkpoints")))
            return None

    def _save(self):
        name = f'{self.fname}_{self.learn.epoch_running_total}'
        self.learn.nep_run['metadata/best_loss'].assign(self.best)
        self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)

    def delete_old_checkpoints(self):
        all_files = list(self.learn.model_dir.glob("*"))
        all_files.sort(key=os.path.getctime)
        discard = all_files[:-self.keep_last]
        [fn.unlink() for fn in discard]



class NeptuneImageGridCallback(Callback):
    order = NeptuneCallback.order+1
    def __init__(self, classes, patch_size,freq=10,imgs_per_grid=32, imgs_per_batch=4, publish_deep_preds=False, apply_activation=True):
        store_attr('freq')
        if not isinstance(patch_size, torch.Size): patch_size = torch.Size(patch_size)
        self.iter_num_train = int(imgs_per_grid/imgs_per_batch) -1 # minus 1 because 1 valid iter batch will be saved too
        self.stride=int(patch_size[0]/ imgs_per_batch)

        store_attr()

    def before_epoch (self):
        self.grid_imgs = []
        self.grid_preds = []
        self.grid_masks = []

    def after_batch(self):
        if self.training:
            if self.iter < self.iter_num_train:
                self.populate_grid()
        else:
            if self.iter == 0:
                self.populate_grid()

    def after_epoch(self):
        if self.epoch%self.freq==0:
            grd_final=[]
            for grd,category in zip([self.grid_imgs,self.grid_preds,self.grid_masks], ["imgs","preds","masks"]):
                grd = torch.cat(grd)
                if category=="imgs":
                        grd = normalize(grd)
                grd_final.append(grd)
            grd= torch.stack(grd_final)
            grd2 = grd.permute(1,0,2,3,4).contiguous().view(-1,3,grd.shape[-2],grd.shape[-1])
            grd3 = make_grid(grd2,nrow=self.imgs_per_batch*3)
            grd4 = grd3.permute(1,2,0)
            self.learn.nep_run["images"].log(File.as_image(grd4))

    def img_to_grd(self,batch):
                    imgs = batch[0,:,::self.stride,:,:].clone()
                    imgs = imgs[:,:self.imgs_per_batch]
                    imgs = imgs.permute(1,0,2,3) # BxCxHxW
                    return imgs

    def populate_grid(self):
        for batch,category,grd in zip([self.learn.x,self.learn.pred, self.learn.y],['imgs','preds','masks'] ,[self.grid_imgs,self.grid_preds,self.grid_masks]):
                    if isinstance(batch,(list,tuple))  and self.publish_deep_preds==False:
                        batch = [x for x in batch if x.size()[2:] == self.patch_size][0] # gets that pred which has same shape as imgs
                    elif isinstance(batch,(list,tuple))  and self.publish_deep_preds==True:
                        batch_tmp = [F.interpolate(b, size=batch[-1].shape[2:],mode="trilinear") for b in batch[:-1]]
                        batch = batch_tmp+batch[-1]
                    batch=batch.cpu()

                    if self.apply_activation==True and category=="preds":
                        batch = F.softmax(batch,dim=1)

                    imgs =self.img_to_grd(batch)
                    if category=="masks" :
                        imgs = imgs.squeeze(1)
                        imgs = one_hot(imgs,self.classes,axis=1)
                    if category!="imgs" and imgs.shape[1]!=3:
                        imgs=imgs[:,1:,:,:]
                    if imgs.shape[1]==1:
                        imgs = imgs.repeat(1,3,1,1)

                    grd.append(imgs)

#
# class NeptuneCallback(Callback):
#
#     order = TrackerCallback.order+1
#     def __init__(self,proj_defaults,config_dict,run_name=None,freq=2,metrics=None,hyperparameters=None,tmp_folder="/tmp",update_nep_run_from_config=False):
#     
#         self.project=get_neptune_project(proj_defaults,'async')
#         self.freq=freq
#         self.tmp_folder =tmp_folder
#         self.update_nep_run_from_config=update_nep_run_from_config
#
#         self.config_dict = config_dict
#         if hyperparameters==None:
#             self.hyperparameters = ['lr','mom','wd']
#
#
#         self.neptune_settings['project']=self.neptune_settings['name']
#         self.neptune_settings={'project':self.neptune_settings['name'], 'api_token':self.neptune_settings['api_token']}
#         self.run_name=run_name
#
#
#      
#     def after_create(self):
#         if self.run_name is not None:
#                 self.run_id = self.get_run_id_from_name()
#                 self.nep_run = self.load_run()
#                 
#         else:
#             self.nep_run = self._init_run()
#         self.nep_run['model_params/summary'].upload(self.summary())
#         self.learn.nep_run = self.nep_run
#         self.learn.epoch_running_total=self.learn.nep_run["model_params/epoch"].fetch()
#         self.learn.nep_run['metadata/run_name']=self.run_name  # not sure wh/besty but metadata/run_name changes to default by itself around this stage
#         self.learn.run_name = self.run_name
#
#     def after_batch(self):
#         if self.iter%self.freq==0:
#             if self.training:
#                 for hp in self.hyperparameters:
#                     self.learn.nep_run['hyperparameters/{}'.format(hp)].log( self.opt.hypers[0][hp])
#                 for key, val in self.learn.loss_func.loss_dict.items():
#                     self.learn.nep_run['metrics/train_loss/'+key].log(val)
#             else:
#                 for key, val in self.learn.loss_func.loss_dict.items():
#                     self.learn.nep_run['metrics/valid_loss/'+key].log(val)
#
#     def after_epoch(self):
#
#         self.learn.epoch_running_total+=1
#         self.learn.nep_run['model_params/epoch']=self.learn.epoch_running_total
#
#     def after_fit(self):
#         self.learn.nep_run.stop()
#
#     def summary(self):
#          try:
#              patch_size=make_patch_size(self.config_dict['dataset_params']['patch_dim0'],self.config_dict['dataset_params']['patch_dim1'])
#              summ = summary(self.learn.model, input_size=tuple([1,1]+patch_size),col_names=["input_size","output_size","kernel_size"],depth=4, verbose=0)
#              tmp_filename = self.tmp_folder+"/summary.txt"
#              with open (tmp_filename,"w") as f:
#                     f.write(str(summ))
#              return tmp_filename
#          except: return 'empty.txt'
#
#     def get_run_id_from_name(self):
#         self.df = self.project.fetch_runs_table(
#             owner="drusmanbashir",
#         ).to_pandas()
#         row =self.df.loc[self.df['metadata/run_name']==self.run_name]
#         if len(row)==0 : return None# i.e., first time
#         else: return row['sys/id'].item()
#
#     def _init_run(self):
#         nep_run = neptune.init_run(run=None,mode="async",**self.ttings)
#         nep_run['model_params/epoch']= 0
#         nep_run = self.populate_nep_run(nep_run,self.config_dict)
#         if self.run_name is None:
#             self.run_name = nep_run['metadata/run_name']=nep_run['sys/id'].fetch()
#         else:
#             nep_run['metadata/run_name']=self.run_name
#         nep_run.wait()
#         print("Initializing new run. Run name {0} ".format(self.run_name))
#         return nep_run
#
#     def load_run(self,param_names=None,nep_mode="async"):
#         if self.run_id is not None:
#             nep_run = neptune.init(run=self.run_id,mode=nep_mode,**self.neptune_settings)
#             print("Run loaded. Run id {}".format(self.run_name))
#         else:
#             nep_run = self._init_run()
#             # run_name_old = self.run_name
#             # self.run_name= self.df.sort_values("sys/creation_time")["sys/id"].iloc[-1]
#             # print("Run id {} does not exist. Loading most recent".format(run_name_old, self.run_name))
#             # nep_run = neptune.init(run=self.run_name,mode=nep_mode,  **self.neptune_settings)
#         if self.update_nep_run_from_config==True:
#             self.populate_nep_run(nep_run,self.config_dict)
#         return nep_run
#
#     def populate_nep_run(self,nep_run,config_dict):
#             for key, value in config_dict.items():
#                 nep_run[key] = value
#                 setattr(self, key, value)
#             return nep_run
#
# %%
if __name__ == "__main__":
    P = Project(project_title="lits"); proj_defaults= P.proj_summary
    config = ConfigMaker(proj_defaults.configuration_filename, raytune=False).config
# %%
    def process_html(fname= 'case_id_dices_valid.html'):
        df = pd.read_html(fname)[0]
        cols = df.columns
        df = df.drop([col for col in cols if 'Unnamed' in col], axis= 1)
        df = df.drop(['loss_dice','loss_dice_label1'],axis = 1)
        df.dropna(inplace=True)
        return df

# %%
    df = process_html()
    df.to_csv("dice_loss2.csv",index=False)
    uniques = (pd.concat([df[val] for val in df.columns if 'case_file' in val])).unique()
# %%
    file_cols = [c for c in df.columns if 'case_file' in c]
    case_ids = list(set([get_case_id_from_filename('lits',Path(f)) for f in uniques]))
    case_id = case_ids[0]
    case_files = [fn for fn in uniques if get_case_id_from_filename('lits',Path(fn)) == case_id]
    fn= case_files[0]
# %%
    df_file_sp=[]
    for file_col in file_cols:
        df_file_sp.append(df[df[file_col].str.contains(fn)].loss_dice_label2)
    df_file_sp = pd.concat(df_file_sp,axis=0)
# %%
# %%

# %%
# %%

    df = pd.read_csv("~/Downloads/legislators-historical.csv")
    df2 = df.groupby("state")

# %%
    run_name = "most_recent"
    M = NeptuneManager(proj_defaults=proj_defaults)
    M.load_run('most_recent', nep_mode = 'read-only', param_names=config.keys())
    M.nep_run['metrics/case_id_dices_valid'].download()
# %%
    run_name = "KITS-507"
    nep_mode = "read-only"
    config_json = "experiments/.neptune/config.json"
    neptune_config = load_json(config_json)
    neptune_config['project'] = neptune_config['name']
    del neptune_config['name']
    nep_run = neptune.init(run=run_name, mode=nep_mode, **neptune_config)
    run = neptune.init(run_name=run_name)
# %%
# %%
    learn = M.new_run(config)
