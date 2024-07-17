# %%
from fastai.callback.schedule import CancelFitException
from fastai.callback.tracker import TrackerCallback

from torchinfo import summary

from fastai.callback.core import Callback
from fastai.callback.fp16 import MixedPrecision
import torch.nn.functional as F
import configparser
import os
from batchgenerators.utilities.file_and_folder_operations import load_json
import ray
from pathlib import Path
from fastai.callback.core import Callback
from fastcore.basics import listify, store_attr
from neptune.types import File
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchvision.utils import make_grid
import torchvision
import torch
import matplotlib.pyplot as plt
from fran.utils.fileio import maybe_makedirs
from ray import tune
import json
import neptune.new as neptune
import ipdb
from fran.utils.helpers import  make_patch_size
tr = ipdb.set_trace

# %%

def make_grid_5d_input(a:torch.Tensor,batch_size_to_plot=16):
    '''
    this function takes in a 5d tensor (BxCxDxWXH) e.g., shape 4,1,64,128,128)
    and creates a grid image for tensorboard
    '''
    middle_point= int(a.shape[2]/2)
    middle = slice(int(middle_point-batch_size_to_plot/2), int(middle_point+batch_size_to_plot/2))
    slc = [0,slice(None), middle,slice(None),slice(None)]
    img_to_save= a [slc]
# BxCxHxW
    img_to_save2= img_to_save.permute(1,0,2,3)  # re-arrange so that CxBxHxW (D is now minibatch)
    img_grid = torchvision.utils.make_grid(img_to_save2,nrow=int(batch_size_to_plot))
    return img_grid


def make_grid_5d_input_numpy_version(a:torch.Tensor,batch_size_to_plot=16):
    img_grid = make_grid_5d_input(a)
    img_grid_np = img_grid.cpu().detach().permute(1,2,0).numpy()
    plt.imshow(img_grid_np)

class FixPredNan(Callback):
    "A `Callback` that terminates training if loss is NaN."
    order = -9

    def after_pred(self):
        self.learn.pred = torch.nan_to_num(self.learn.pred, nan=0.5)
        "Test if `last_loss` is NaN and interrupts training."


class TensorboardCallback(Callback):
    #     def __init__(self,config, logdir,frequency,metrics=None):
    def __init__(self, tb_dir, frequency, metrics=None, grid_maker_func=make_grid_5d_input):
        self.grid_maker_func= grid_maker_func
        self.writer = SummaryWriter(tb_dir)
        self.logfile = Path(tb_dir + '/log.ini')
        self.log = configparser.ConfigParser()
        if self.logfile.exists():
            self.log.read(self.logfile)
            self.train_step, self.valid_step = int(self.log['TENSORBOARD_STEP']['train_step']), int(
                self.log['TENSORBOARD_STEP']['valid_step'])
        else:
            self.log['TENSORBOARD_STEP'] = {'train_step': 0, 'valid_step': 0}
            self.train_step = self.valid_step = 0

        if metrics: self.metrics = listify(metrics)
        self.frequency = frequency

    def after_batch(self):
        if self.iter % self.frequency == (self.frequency - 1):
            if self.training:
                self.train_step += self.x.shape[0]
                self.writer.add_scalar("Loss/Train", self.loss.item(), self.train_step)
                if self.metrics:
                    [
                        self.writer.add_scalar(metric.name,
                                               metric(self.pred, self.yb).item(), self.train_step)
                        for metric in self.metrics
                    ]
                for i, h in enumerate(self.opt.hypers):
                    for k, v in h.items():
                        self.writer.add_scalar(f'{k}_{i}', v, self.train_step)

            else:
                self.valid_step += self.x.shape[0]
                self.writer.add_scalar("Loss/Valid", self.loss.item(), self.valid_step)
                for img,img_name in zip([self.x,self.y,self.pred], ['input_images','input_labels','predictions']):
                    img_grid = self.grid_maker_func(img)
                    self.writer.add_image(img_name,img_grid,self.valid_step)
    #
    #             pred_imgs = self.pred[0, :, :, :, self.inds].permute(3, 0, 1, 2).cpu().float()
    #             # y_imgs = self.y[0, :, :, self.inds].permute(2, 0, 1).cpu().float().unsqueeze(1)
    #             y_imgs = self.y[0, :, :, :, self.inds].permute(3, 0, 1, 2).cpu()
    #             x_imgs = self.x[0, :, :, :, self.inds].permute(3, 0, 1, 2).cpu()
    #             img_grid0 = torchvision.utils.make_grid(x_imgs)  #tensor shape needed : [C,H,W]
    #             img_grid1 = torchvision.utils.make_grid(pred_imgs)  #tensor shape needed : [C,H,W]
    #             img_grid2 = torchvision.utils.make_grid(torch.argmax(pred_imgs, 1, keepdim=True))
    #             img_grid3 = torchvision.utils.make_grid(y_imgs)
    #
    #             self.writer.add_image('16_actual iamges', img_grid0, self.valid_step)
    #             self.writer.add_image('16_predictions', img_grid1, self.valid_step)
    #             self.writer.add_image('16_arg_max', img_grid2, self.valid_step)
    #             self.writer.add_image('16_true_labels', img_grid3, self.valid_step)
    #
    #             # kits_log = get_val_metrics(self.pred,self.y,self.config)
    #             # for key_name, value in kits_log.items(): self.writer.add_scalar(key_name,value,self.valid_step)
    #
    # def after_epoch(self):
        self.log['TENSORBOARD_STEP']['train_step'] = str(self.train_step)
        self.log['TENSORBOARD_STEP']['valid_step'] = str(self.valid_step)
        with open(self.logfile, 'w') as lf:
            self.log.write(lf)

    def after_fit(self):
        print("Closing tensorboard")
        self.writer.add_graph(self.model, self.x)
        self.writer.close()

    def after_cancel_train(self):
        print(self.n_iter)
        print("Closing tensorboard ... fit cancelled prematurely")
        self.writer.add_graph(self.model, self.x)
        self.writer.close()
#
# class TrackerCallback(Callback):
#     "A `Callback` that keeps track of the best value in `monitor`."
#     order,remove_on_fetch = 60,True
#     def __init__(self, monitor='valid_loss', comp=None, min_delta=0., reset_on_fit=True, best=None):
#         if comp is None: comp = np.less if 'loss' in monitor or 'error' in monitor else np.greater
#         if comp == np.less: min_delta *= -1
#         self.monitor,self.comp,self.min_delta,self.reset_on_fit,self.best= monitor,comp,min_delta,reset_on_fit,best
#
#     def before_fit(self):
#         "Prepare the monitored value"
#         self.run = not hasattr(self, "lr_finder") and not hasattr(self, "gather_preds")
#         if self.reset_on_fit or self.best is None: self.best = float('inf') if self.comp == np.less else -float('inf')
#         assert self.monitor in self.recorder.metric_names[1:]
#         self.idx = list(self.recorder.metric_names[1:]).index(self.monitor)
#
#     def after_epoch(self):
#         "Compare the last value to the best up to now"
#         val = self.recorder.values[-1][self.idx]
#         if self.comp(val - self.min_delta, self.best): self.best,self.new_best = val,True
#         else: self.new_best = False
#
#     def after_fit(self): self.run=True


# class NeptuneCallback(Callback):
#
#     order = TrackerCallback.order+1
#     def __init__(self,proj_defaults,config_dict,run_name=None,freq=2,metrics=None,hyperparameters=None,tmp_folder="/tmp"):
#         self.neptune_settings=load_json(Path(proj_defaults.neptune_folder)/"config.json")
#         self.project=neptune.init_project(**self.neptune_settings)
#         self.tmp_folder =tmp_folder
#         self.freq=freq
#         self.run_name=run_name
#
#         self.config_dict = config_dict
#         if hyperparameters==None:
#             self.hyperparameters = ['lr','mom','wd']
#
#     def after_create(self):
#         # self.checkpoints_folder=Path(tune.get_trial_dir())
#
#         self.df = self.project.fetch_runs_table(
#             owner="drusmanbashir",
#         ).to_pandas()
#
#         self.neptune_settings['project']=self.neptune_settings['name']
#         self.neptune_run_settings={'project':self.neptune_settings['name'], 'api_token':self.neptune_settings['api_token']}
#
#         if self.run_name == None and self.config_dict:
#             self.learn.nep_run = self._init_run()
#
#
#         else:
#             self.learn.nep_run = self.load_run()
#             if self.config_dict:
#                 self.learn.nep_run = self.populate_nep_run(self.learn.nep_run,self.config_dict)
#         try:
#             self.learn.epoch_running_total=self.learn.nep_run["model_params/epoch"].fetch()
#         except:
#             self.learn.epoch_running_total = 0
#         self.learn.run_name = self.run_name
#         self.learn.nep_run['model_params/summary'].upload(self.summary()) 
#
#     def after_batch(self):
#         if self.iter%self.freq==0:
#             if self.training:
#                 for hp in self.hyperparameters:
#                     self.learn.nep_run['hyperparameters/{}'.format(hp)].log( self.opt.hypers[0][hp])
#                 for loss_item in self.loss_dict.items():
#                     self.learn.nep_run['metrics/train_loss/'+loss_item[0]].log(loss_item[1].item())
#             else:
#
#                 for loss_item in self.loss_dict.items():
#                     self.learn.nep_run['metrics/valid_loss/'+loss_item[0]].log(loss_item[1])
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
#          patch_size=make_patch_size(self.config_dict['dataset_params']['patch_dim0'],self.config_dict['dataset_params']['patch_dim1'])
#          summ = summary(self.learn.model, input_size=tuple([1,1]+patch_size),col_names=["input_size","output_size","kernel_size"],depth=4, verbose=0)
#          tmp_filename = self.tmp_folder+"/summary.txt"
#          with open (tmp_filename,"w") as f:
#                 f.write(str(summ))
#          return tmp_filename
#
#     def _init_run(self):
#             nep_run = neptune.init_run(run=None,mode="async",**self.neptune_run_settings)
#             nep_run['model_params/epoch']= self.learn.epoch_running_total =0
#             nep_run = self.populate_nep_run(nep_run,self.config_dict)
#             self.run_name = nep_run['sys/id'].fetch()
#             return nep_run
#
#     def load_run(self,param_names=None,nep_mode="async"):
#         try:
#             nep_run = neptune.init(run=self.run_name,mode=nep_mode,**self.neptune_settings)
#             print("Existing Run loaded. Run id {}".format(self.run_name))
#         except:
#             run_name_old = self.run_name
#             self.run_name= self.df.sort_values("sys/creation_time")["sys/id"].iloc[-1]
#             print("Run id {} does not exist. Loading most recent".format(run_name_old, self.run_name))
#             nep_run = neptune.init(run=self.run_name,mode=nep_mode,  **self.neptune_settings)
#         return nep_run
#
#     def populate_nep_run(self,nep_run,config_dict):
#             for key, value in config_dict.items():
#                 nep_run[key] = value
#                 setattr(self, key, value)
#             return nep_run
#
#  
# class NeptuneCheckpointCallback(TrackerCallback):
#     "A `TrackerCallback` that saves the model's best during training and loads it at the end."
#     order = NeptuneCallback.order+1
#     def __init__(self, monitor='valid_loss', comp=None, min_delta=0., fname='model', every_epoch=False, at_end=False,
#                  with_opt=True, reset_on_fit=True,resume=True,raytune_trial_name=None):
#         super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
#         assert not (every_epoch and at_end), "every_epoch and at_end cannot both be set to True"
#         # keep track of file path for logg`ers
#         self.last_saved_path = None
#         store_attr('fname,every_epoch,at_end,with_opt,resume,raytune_trial_name')
#
#     def after_create(self):
#         try:
#             self.best=self.learn.nep_run["model_params/best_loss"].fetch()
#             self.learn.nep_run.wait()
#         except: pass
#         # keep track of file path for loggers
#         self.learn.nep_run['meta/model_dir']=self.learn.model_dir
#         if not self.raytune_trial_name:
#             self.learn.nep_run['model_params/best_loss']=self.best
#             # self.learn.model_dir = self.learn.model_dir/self.learn.run_name
#             maybe_makedirs(self.learn.model_dir)
#         else:
#             self.learn.nep_run["meta/run_name"]=self.raytune_trial_name
#         if self.resume==True:
#             print("Loading model state from {}".format(self.learn.model_dir))
#             try:
#                 self.learn.load(self.fname,device=self.device)
#                 print("Successfully loaded model.")
#             except:
#                 print("{0} does not exist in folder {1}".format(self.fname,self.learn.model_dir))
#                 print("Training with new initialization.")
#     def _save(self, name):
#
#         self.learn.nep_run['model_params/best_loss']=self.best
#         self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)
#
#     def after_epoch(self):
#         "Compare the value monitored to its best score and save if best."
#
#         if self.every_epoch:
#             if (self.epoch%self.every_epoch) == 0: self._save(f'{self.fname}_{self.epoch}')
#         else: #every improvement
#             super().after_epoch()
#             if self.new_best:
#                 print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.')
#                 self._save(f'{self.fname}')
#
#     def after_fit(self, **kwargs):
#         "Load the best model."
#         if self.at_end: self._save(f'{self.fname}')
#         elif not self.every_epoch: self.learn.load(f'{self.fname}', with_opt=self.with_opt)      
#
#
#  #
# class NeptuneCheckpointCallback_old(TrackerCallback):
#     "A `TrackerCallback` that saves the model's best during training and loads it at the end."
#     _only_train_loop,order = True,NeptuneCallback.order+1
#     def __init__(self, monitor='valid_loss', comp=None, min_delta=0.,  freq=None, at_end=False,
#                  with_opt=True, reset_on_fit=False, upload_model=False,delete_previous_checkpoints=True,one_file_only=True):
#         # ray.util.pdb.set_trace()
#         super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
#         store_attr()
#
#     def after_create(self):
#         try:
#             self.best=self.learn.nep_run["model_params/best_loss"].fetch()
#             self.learn.nep_run.wait()
#         except: pass
#         # keep track of file path for loggers
#         self.learn.model_dir = self.learn.model_dir/self.learn.run_name
#         maybe_makedirs(self.learn.model_dir)
#         # store_attr('manager,every_epoch,at_end,with_opt, upload_model')
#
#     def save(self): 
#         if self.one_file_only == True:
#             name = "model"
#         else:
#             name ="epoch_" + str(self.learn.epoch_running_total)
#         self.learn.nep_run['model_params/best_loss']=self.best
#         self.lastsaved_path = self.learn.save(name, with_opt=self.with_opt)
#         if self.upload_model == True:
#             file_to_upload = str(self.lastsaved_path)
#             self.learn.nep_run["checkpoints"][name].upload(file_to_upload)
#
#     def after_epoch(self):
#         "Compare the value monitored to its best score and save if best."
#
#         if self.freq:
#             if (self.epoch%self.freq) == 0: self.save()
#         else: #every improvement
#              if hasattr(self,"new_best"):
#                 print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.')
#                 self.save()
#
#     def after_fit(self, **kwargs):
#         "Load the best model."
#         if self.at_end: self.save()
#         elif not self.freq: 
#                 self.load()
#
#     def load(self):
#             list_of_files= self.learn.model_dir.glob('*.pth')     
#             try:
#                 latest_chckpnt= max(list_of_files, key=lambda p: p.stat().st_ctime)
#                 name =latest_chckpnt.name.split(".")[0]
#                 self.learn.load(name)
#                 # self.learn.load(f'{name}', with_opt=self.with_opt)
#             except:
#                 print("no checkpoints to load yet")
#     def __del__(self):
#         print("Cleaning up..")
#         self.nep_run.stop()
#
# class NeptuneCallback_old(Callback):
#     def __init__(self,manager, frequency, metrics=None):
#
#         self.manager= manager
#         if metrics: self.metrics = listify(metrics)
#         self.frequency = frequency
#
#     def before_fit(self):
#         # self.manager.update_logs()
#         self.learn.epoch_running_total=self.manager.nep_run["model_params/epoch"].fetch()
#
#     def after_batch(self):
#         if self.training:
#             self.manager.nep_run['hyperparameters/lr'].log( self.opt.hypers[0]["lr"])
#             for loss_item in self.loss_dict.items():
#                 self.manager.nep_run['metrics/train_loss/'+loss_item[0]].log(loss_item[1].item())
#         else:
#
#             for loss_item in self.loss_dict.items():
#                 self.manager.nep_run['metrics/valid_loss/'+loss_item[0]].log(loss_item[1])
#
#     def after_epoch(self):
#         self.learn.epoch_running_total+=1
#         self.manager.nep_run['model_params/epoch']=self.learn.epoch_running_total
#
#     def after_fit(self):
#         self.manager.nep_run.stop()
#
#
#
# class NeptuneCallback_old(Callback):
#     def __init__(self,config_dict,experiment_name, neptune_dir= "/home/ub/Dropbox/code/fran/experiments/.neptune",run_name=None, freq=2,metrics=None):
#
#         if metrics: self.metrics = listify(metrics)
#         self.neptune_settings=load_json(Path(neptune_dir)/"config.json")
#         self.project=neptune.init_project(**self.neptune_settings)
#         self.config_dict = config_dict
#         self.config_dict.update({'experiment_name':experiment_name})
#         self.freq=freq
#         self.run_name = run_name
#
#     def after_create(self):
#         # self.checkpoints_folder=Path(tune.get_trial_dir())
#
#         self.df = self.project.fetch_runs_table(
#             owner="drusmanbashir",
#         ).to_pandas()
#
#         self.neptune_settings['project']=self.neptune_settings['name']
#         self.neptune_run_settings={'project':self.neptune_settings['name'], 'api_token':self.neptune_settings['api_token']}
#         if self.run_name == None:
#             self.learn.nep_run = self._init_run(self.config_dict)
#         else:
#             self.learn.nep_run = self.load_run(self.run_name)
#
#     def _init_run(self,config_dict):
#             nep_run = neptune.init_run(run=None,mode="async",**self.neptune_run_settings)
#             nep_run['model_params/epoch']= self.learn.epoch_running_total =0
#             for key, value in config_dict.items():
#                 nep_run[key] = value
#                 setattr(self, key, value)
#             return nep_run
#
#
#     def load_run(self,run_name,param_names=None,nep_mode="async"):
#         try:
#             self.learn.nep_run = neptune.init(run=run_name,mode=nep_mode,**self.neptune_settings)
#             print("Existing Run loaded. Run id {}".format(run_name))
#         except:
#             run_name_old = self.run_name
#             self.run_name= self.df.sort_values("sys/creation_time")["sys/id"].iloc[-1]
#             print("Run id {} does not exist. Loading most recent".format(run_name_old, self.run_name))
#             self.learn.nep_run = neptune.init(run=run_name,mode=nep_mode,  **self.neptune_settings)
#         # config_dict = {}
#         # if not param_names:
#         #     pass # not impl
#         # for param in param_names:
#         #     neptune_dict = self.nep_run[param].fetch()
#         #     config_dict.update({param: parse_neptune_dict(neptune_dict)})
#         #
#         # self.assimilate_attribs(config_dict)
#         # self.config_dict = config_dict
#         # learn = self.create_learner()
#         #     self.learn.nep_run = self.load_run(run_name)
#  
#     def after_batch(self):
#         if self.iter%self.freq==0:
#             if self.training:
#                 self.learn.nep_run['hyperparameters/lr'].log( self.opt.hypers[0]["lr"])
#                 for loss_item in self.loss_dict.items():
#                     self.learn.nep_run['metrics/train_loss/'+loss_item[0]].log(loss_item[1].item())
#             else:
#
#                 for loss_item in self.loss_dict.items():
#                     self.learn.nep_run['metrics/valid_loss/'+loss_item[0]].log(loss_item[1])
#
#     def after_epoch(self):
#         self.learn.epoch_running_total+=1
#         self.learn.nep_run['model_params/epoch']=self.learn.epoch_running_total
#
#     def after_fit(self):
#         self.learn.nep_run.stop()
#
#
class NeptuneCallback_Ray(NeptuneCallback):
    def _init_run(self):
        if not hasattr(self,"trial_name"):
            self.trial_name = tune.get_trial_name()
        row =self.df.loc[self.df['meta/run_name']==self.trial_name]
        if len(row)==0 : # i.e., first time
            nep_run = neptune.init_run(run=None,mode="async",name=self.trial_name,**self.neptune_run_settings)
            nep_run['model_params/epoch']= self.learn.epoch_running_total =0
        else:
            run_name = row['sys/id'].item()
            nep_run = neptune.init_run(run=run_name,mode="async",**self.neptune_run_settings)
            self.learn.epoch_running_total=nep_run['model_params/epoch'].fetch()

        if hasattr(self,"config_dict"):
            for key, value in self.config_dict.items():
                nep_run[key] = value
                setattr(self, key, value)

        return nep_run

    @classmethod
    def from_running_trial(self,proj_defaults, config_dict,experiment_name,**kwargs):
        config_dict.update({'experiment_name':experiment_name})
        self = self(proj_defaults=proj_defaults,config_dict=config_dict, **kwargs)
        return self

    @classmethod
    def from_trial_name(self,trial_name,proj_defaults,config_dict=None,**kwargs):
        self = self(proj_defaults=proj_defaults,config_dict=config_dict, **kwargs)
        self.trial_name= trial_name
        return self
class NeptuneImageGridCallback(Callback):
    order = NeptuneCallback_Ray.order+1
    def __init__(self,classes ,patch_size,imgs_per_grid=32,imgs_per_batch=4, publish_deep_preds=False, apply_activation=True):
        self.iter_num_train = int(imgs_per_grid/imgs_per_batch) -1 # minus 1 because 1 valid iter batch will be saved too
        # self.stride=int( patch_size/imgs_per_batch)
        self.stride=2
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
        grd_final=[]
        for grd,category in zip([self.grid_imgs,self.grid_preds,self.grid_masks], ["imgs","preds","lms"]):
            grd = torch.cat(grd)
            if category=="imgs":
                    grd = normalize(grd)
            grd_final.append(grd.cpu())
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
        for batch,category,grd in zip([self.learn.x,self.learn.pred, self.learn.y],['imgs','preds','lms'] ,[self.grid_imgs,self.grid_preds,self.grid_masks]):
                    if isinstance(batch,list)  and self.publish_deep_preds==False:
                            batch = batch[-1]
                    elif isinstance(batch,list)  and self.publish_deep_preds==True:
                        batch_tmp = [F.interpolate(b, size=batch[-1].shape[2:],mode="trilinear") for b in batch[:-1]]
                        batch = batch_tmp+batch[-1]

                    if self.apply_activation==True and category=="preds":
                        batch = F.softmax(batch,dim=1)

                    imgs =self.img_to_grd(batch)
                    if category=="lms" :
                        imgs = imgs.squeeze(1)
                        imgs = one_hot(imgs,self.classes)
                    if category!="imgs" and imgs.shape[1]!=3:
                        imgs=imgs[:,1:,:,:]
                    if imgs.shape[1]==1:
                        imgs = imgs.repeat(1,3,1,1)

                    grd.append(imgs)

class TuneCheckpointCallback_old(Callback):
    def __init__(self, checkpoint_freq=2, with_opt=True,**kwargs):
        store_attr()
        self.trial_dir = tune.get_trial_dir()
        super().__init__(**kwargs)
    def before_fit(self):
        ray.util.pdb.set_trace()
        checkpoint_folders = list(Path(self.trial_dir).glob("*checkpoint*"))
        if len(checkpoint_folders)>0:
            checkpoint_folders.sort(key=os.path.getmtime,reverse=True)
            latest_chckpnt_file =checkpoint_folders[0]/"checkpoint"
            checkpoint = torch.load(latest_chckpnt_file)
            print("Loading checkpoint from filename: {}..".format(latest_chckpnt_file))
            self.learn.model.load_state_dict(checkpoint[0])
            self.learn.opt.load_state_dict(checkpoint[1])
            self.learn.model.load_state_dict(torch.load(latest_chckpnt_file))
            self.learn.load(f'{latest_chckpnt_file}', with_opt=self.with_opt)
            print("Checkpoint loaded successfully.")
        else:
            print("No checkpoints found on Tune.")

    def after_epoch(self):
        if self.epoch%self.checkpoint_freq==0:
            with tune.checkpoint_dir(self.learn.epoch) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        print("Saving model to {}".format(path))
                        torch.save((self.learn.model.state_dict(), self.learn.opt.state_dict()), path)

class TuneCheckpointCallback(TrackerCallback):
    order=51 # (Recorder callback order +1)
    def __init__(self, monitor='valid_loss', comp=None, min_delta=0., checkpoint_dir=None, fname='model', freq= 6, at_end=True,
                 with_opt=True, reset_on_fit=False):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
        store_attr('checkpoint_dir,fname,freq,at_end,with_opt')
        # keep track of file path for loggers
        self.last_saved_path = None
    def after_create(self):
        if self.checkpoint_dir:
            with open(os.path.join(self.checkpoint_dir, "checkpoint")) as f:
                state = json.loads(f.read())
                state["step"] + 1
        self.learn.model_dir = Path(tune.get_trial_dir()) / "model_checkpoints"
        # self.learn.model_dir = Path(self.checkpoint_dir)/ "model_checkpoints"
        self.fname_prefix = self.learn.model_dir/self.fname
        self._load_latest_checkpoint()

    def after_epoch(self):
        step=self.epoch
        with tune.checkpoint_dir(step=step) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            with open(path, "w") as f:
                f.write(json.dumps({"step": step}))
        "Compare the value monitored to its best score and save if best."
        if (self.epoch%self.freq) == 0:
            # self._save(f'{self.fname_prefix}_{self.epoch}')
            self._save(f'{self.fname_prefix}')
    def after_fit(self, **kwargs):
        "Load the best model."
        if self.at_end: self._save(f'{self.fname_prefix}')
        # elif not self.every_epoch: self.learn.load_model(f'{self.fname_prefix}', with_opt=self.with_opt)

    def _save(self, name):
        self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)

    def _load_latest_checkpoint(self):
        checkpoints = list(self.learn.model_dir.glob("*"))
        if len(checkpoints) > 0:
            checkpoints.sort(key=os.path.getmtime, reverse=True)
            latest_chckpnt_file = checkpoints[-1].name[:-4]
            print("Loading checkpoint from {}".format(latest_chckpnt_file))
            self.learn.load(latest_chckpnt_file, device='cuda', with_opt=self.with_opt)
        else:
            print("No checkpoints on Tune. Initializing..")

class TuneTrackerCallback(Callback):
    order = TuneCheckpointCallback.order+1
    def __init__(self,freq=2, **kwargs):
        store_attr()
        super().__init__(**kwargs)
    def before_epoch(self):
        self.running_losses_valid = []
    def after_batch(self):
        if not self.training and self.iter%self.freq==0:
            self.running_losses_valid.append(([v.item() for v in self.loss_dict.values()]))
    def after_epoch(self):
        mean_loss =torch.tensor(self.running_losses_valid).mean(0)
        reporting = {key:val for key,val in zip(self.loss_dict.keys(),mean_loss)}
        tune.report(**reporting)

class TerminateOnNaNCallback_ub(Callback):
    "A `Callback` that terminates training if loss is NaN."

    order = -9
    def after_batch(self):
        "Test if `last_loss` is NaN and interrupts training."
        if torch.isinf(self.loss) or torch.isnan(self.loss):
            print("NaNs !!")
            raise CancelFitException

# Cell
class GradientClip(Callback):
    "Clip norm of gradients"
    order = MixedPrecision.order + 1

    def __init__(self, max_norm: float = 1., norm_type: float = 2.0):
        store_attr()

    def before_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), self.max_norm, self.norm_type, error_if_nonfinite=True)



def normalize(tensr,intensity_percentiles=[0.,1.]): 
        tensr = (tensr-tensr.min())/(tensr.max()-tensr.min())
        tensr = tensr.to('cpu')
        qtiles = torch.quantile(tensr, q=torch.tensor(intensity_percentiles))
        
        vmin = qtiles[0]
        vmax = qtiles[1]
        tensr[tensr<vmin]=vmin
        tensr[tensr>vmax] = vmax
        return tensr


