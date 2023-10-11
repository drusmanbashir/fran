
# %%
import time
from monai.transforms.croppad.dictionary import ResizeWithPadOrCropd
from monai.transforms.intensity.dictionary import RandAdjustContrastd, RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
from torch.profiler import profile, record_function, ProfilerActivity
from neptune.types import File
from torchvision.utils import make_grid
from lightning.pytorch.profilers import AdvancedProfiler
import warnings
from typing import Any
# from fastcore.basics import GetAttr
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers.neptune import NeptuneLogger
from monai.transforms.spatial.dictionary import RandAffined, RandFlipd
from torchvision.transforms import Compose
from monai.data import DataLoader
from monai.transforms import RandAffined
from lightning.pytorch import LightningDataModule, LightningModule, Trainer

import torch
import operator
from fran.data.dataset import ImageMaskBBoxDatasetd, MaskLabelRemap2, NormaliseClipd
from fran.transforms.spatialtransforms import one_hot
from fran.utils.helpers import folder_name_from_list
from fran.data.dataloader import  img_mask_bbox_collated
import itertools as il
from fran.utils.helpers import *
from fran.utils.fileio import *
from fran.utils.imageviewers import *

# from fastai.learner import *
from fran.evaluation.losses import *
from fran.architectures.create_network import create_model_from_conf, nnUNet, pool_op_kernels_nnunet
# from fran.managers.base import *
from fran.utils.helpers import *

def get_neptune_config(proj_defaults):
        """
        Returns particular project workspace
        """
        project_title = proj_defaults.project_title
        commons = load_yaml(common_vars_filename)
        project_name = "/".join([commons["neptune_workspace_name"], project_title])
        api_token = commons["neptune_api_token"]
        return project_name, api_token




def normalize(tensr, intensity_percentiles=[0.0, 1.0]):
    tensr = (tensr - tensr.min()) / (tensr.max() - tensr.min())
    tensr = tensr.to("cpu", dtype=torch.float32)
    qtiles = torch.quantile(tensr, q=torch.tensor(intensity_percentiles))

    vmin = qtiles[0]
    vmax = qtiles[1]
    tensr[tensr < vmin] = vmin
    tensr[tensr > vmax] = vmax
    return tensr


class NeptuneImageGridCallback(Callback):
    def __init__(
        self,
        classes,
        patch_size,
        freq=20,
        grid_rows=6,
        imgs_per_batch=4,
        publish_deep_preds=False,
        apply_activation=True,
    ):
        if not isinstance(patch_size, torch.Size):
            patch_size = torch.Size(patch_size)
        self.stride = int(patch_size[0] / imgs_per_batch)
        store_attr()

    def on_train_start(self,trainer,pl_module):
        len_dl= int(len(trainer.train_dataloader)/trainer.accumulate_grad_batches)
        self.freq = int(len_dl/self.grid_rows)
    def on_train_epoch_start(self,trainer,pl_module):
        self.grid_imgs = []
        self.grid_preds = []
        self.grid_labels= []

    def on_validation_epoch_start(self,trainer,pl_module):
        self.validation_grid_created=False

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        if trainer.global_step% self.freq==0:
                self.populate_grid(pl_module, batch)

    def on_validation_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        if self.validation_grid_created==False:
                self.populate_grid(pl_module, batch)
                self.validation_grid_created=True
    #
    def on_train_epoch_end(self,trainer,pl_module):
        if trainer.current_epoch % self.freq == 0:
            grd_final = []
            for grd, category in zip(
                [self.grid_imgs, self.grid_preds, self.grid_labels],
                ["imgs", "preds", "labels"],
            ):
                grd = torch.cat(grd)
                if category == "imgs":
                    grd = normalize(grd)
                grd_final.append(grd)
            grd = torch.stack(grd_final)
            grd2 = (
                grd.permute(1, 0, 2, 3, 4)
                .contiguous()
                .view(-1, 3, grd.shape[-2], grd.shape[-1])
            )
            grd3 = make_grid(grd2, nrow=self.imgs_per_batch * 3)
            grd4 = grd3.permute(1, 2, 0)
            grd4 = np.array(grd4)
            trainer.logger.experiment["images"].append(File.as_image(grd4))

    def img_to_grd(self, batch):
        imgs = batch[0, :, :: self.stride, :, :].clone()
        imgs = imgs[:, : self.imgs_per_batch]
        imgs = imgs.permute(1, 0, 2, 3)  # BxCxHxW
        return imgs

    def fix_channels(self,tnsr):
            if tnsr.shape[1] == 2:
                tnsr = tnsr[:, 1:, :, :]
            if tnsr.shape[1] == 1:
                tnsr = tnsr.repeat(1, 3, 1, 1)
            return tnsr


    def populate_grid(self,pl_module,batch):
        img = batch['image'].cpu()

        label = batch['label'].cpu()
        label = label.squeeze(1)
        label =one_hot(label,self.classes,axis=1)
        pred = pl_module.pred
        if isinstance(pred,tuple): 
            pred = pred[0] 
        pred = pred.cpu()

        if self.apply_activation==True:
            pred = F.softmax(pred.to(torch.float32),dim=1) 

        img,label,pred = self.img_to_grd(img),self.img_to_grd(label),self.img_to_grd(pred)
        img, label,pred = self.fix_channels(img), self.fix_channels(label),self.fix_channels(pred)

        self.grid_imgs.append(img)
        self.grid_preds.append(pred)
        self.grid_labels.append(label)



class NepMan(NeptuneLogger,Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_batch_end(self,trn,plm,outputs,batch,batch_idx):
        ld = plm.loss_fnc.loss_dict
        plm.logger.log_metrics(ld)

class NepImages(Callback):
    def __init__(self,freq): store_attr()
    def on_train_start(self, trainer, pl_module):
        pass

    def populate_grid(self):
        for batch, category, grd in zip(
            [self.learn.x, self.learn.pred, self.learn.y],
            ["imgs", "preds", "masks"],
            [self.grid_imgs, self.grid_preds, self.grid_masks],
        ):
            if isinstance(batch, (list, tuple)) and self.publish_deep_preds == False:
                batch = [x for x in batch if x.size()[2:] == self.patch_size][
                    0
                ]  # gets that pred which has same shape as imgs
            elif isinstance(batch, (list, tuple)) and self.publish_deep_preds == True:
                batch_tmp = [
                    F.interpolate(b, size=batch[-1].shape[2:], mode="trilinear")
                    for b in batch[:-1]
                ]
                batch = batch_tmp + batch[-1]
            batch = batch.cpu()

            if self.apply_activation == True and category == "preds":
                batch = F.softmax(batch, dim=1)

            imgs = self.img_to_grd(batch)
            if category == "masks":
                imgs = imgs.squeeze(1)
                imgs = one_hot(imgs, self.classes, axis=1)
            if category != "imgs" and imgs.shape[1] != 3:
                imgs = imgs[:, 1:, :, :]
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)

            grd.append(imgs)


class DataManager(LightningDataModule):
    def __init__(
            self,project,   dataset_params:dict,transform_factors:dict,affine3d:dict,batch_size=8, ):
        '''

        '''
        super().__init__()
        self.save_hyperparameters()
        store_attr(but='transform_factors')
        global_properties=load_dict(project.global_properties_filename)
        self.dataset_params['intensity_clip_range']=global_properties["intensity_clip_range"]
        self.dataset_params['mean_fg']=global_properties["mean_fg"]
        self.dataset_params['std_fg']=global_properties["std_fg"]
        self.assimilate_tfm_factors(transform_factors)

    # 
    # def state_dict(self):
    #     state={'batch_size':'j'}
    #     return state
    #     # return self.dataset_params
    #
    # def load_state_dict(self,state_dict):
    #     self.batch_size= state_dict['batch_size']
    # #
    def assimilate_tfm_factors(self,transform_factors):
        for key, value in transform_factors.items():
            dici = {'value':value[0], 'prob':value[1]}
            setattr(self,key,dici)

    def prepare_data(self):
        #getting the right folders
        prefixes, value_lists = ["spc", "dim"], [
            self.dataset_params["spacings"],
            self.dataset_params["src_dims"],
        ]
        if bool(self.dataset_params['patch_based']) == True:
            parent_folder = self.project.patches_folder
        else:
            parent_folder = self.project.whole_images_folder
            for listi in prefixes, value_lists:
                del listi[0]

        for prefix, value_list in zip(prefixes, value_lists):
            parent_folder = folder_name_from_list(prefix, parent_folder, value_list)
        self.dataset_folder = parent_folder
        assert self.dataset_folder.exists(), "Dataset folder {} does not exists".format(
            self.dataset_folder
        )

    def create_transforms(self):
        all_after_item=[
            MaskLabelRemap2(keys=['label'],src_dest_labels=self.dataset_params['src_dest_labels']),
            EnsureChannelFirstd(keys=['image','label'],channel_dim='no_channel'),
            NormaliseClipd(keys=['image'],clip_range= self.dataset_params['intensity_clip_range'],mean=self.dataset_params['mean_fg'],std=self.dataset_params['std_fg']),
            ResizeWithPadOrCropd(keys=['image','label'],source_key='image',spatial_size=self.dataset_params['src_dims']),
    ]


        t2 =  [

            EnsureTyped(keys=["image", "label"], device='cuda',track_meta=False),
                RandFlipd(keys=["image", "label"], prob=self.flip['prob'], spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=self.flip['prob'], spatial_axis=1),
                RandScaleIntensityd(keys="image", factors=self.scale['value'], prob=self.scale['prob']),
                RandGaussianNoised(keys=['image'],std = self.noise['value'],prob=self.noise['prob']),
                RandShiftIntensityd(keys="image",offsets=self.shift['value'], prob=self.shift['prob']),
                RandAdjustContrastd(['image'],gamma=self.contrast['value'],prob=self.contrast['prob']),
                self.create_affine_tfm(),
                   ]
        t3 = [ResizeWithPadOrCropd(keys=['image','label'],source_key='image',spatial_size=self.dataset_params['patch_size'])]
        self.tfms_train = Compose(all_after_item+t3)
        self.tfms_valid = Compose(all_after_item+t3)



    def create_affine_tfm(self):

        affine = RandAffined(
            keys = ['image','label'],
            mode= ['bilinear','nearest'],
            prob= self.affine3d['p'],
            # spatial_size=self.dataset_params['src_dims'],
            rotate_range = self.affine3d['rotate_range'],
            scale_range= self.affine3d['scale_range'],
        )
        return affine
        

    def setup(
        self,
        stage:str=None
    ):

        self.train_list, self.valid_list =project.get_train_val_files(self.dataset_params['fold'])
        self.create_transforms()
        bboxes_fname = self.dataset_folder / "bboxes_info"
        self.train_ds = ImageMaskBBoxDatasetd(
            self.train_list,
            bboxes_fname,
            self.dataset_params["class_ratios"],
            transform=self.tfms_train
        )
        self.valid_ds = ImageMaskBBoxDatasetd(
            self.valid_list,
            bboxes_fname,
            transform=self.tfms_valid
        )

    def train_dataloader(self,  num_workers=24, **kwargs):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=num_workers,
            collate_fn=img_mask_bbox_collated,
            persistent_workers=True,
            pin_memory=True
        )
        return train_dl


    def val_dataloader(self,num_workers=24,**kwargs):
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=num_workers,
            collate_fn=img_mask_bbox_collated,
            persistent_workers=True,
            pin_memory=True
        )
        return valid_dl
    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        # if dataloader_idx==0:
        #     batch=self.after_batch(batch)
        # else:
            # tr()
        return batch

    def forward(self,inputs,target):
        return self.model(inputs)


class nnUNetTrainer(LightningModule):

    def __init__(self,project,dataset_params,model_params,loss_params,compiled=False):
        super().__init__()
        store_attr('project,dataset_params,model_params,loss_params,compiled')
        self.save_hyperparameters('model_params','loss_params')
        self.model,self.loss_fnc= self.create_model()

    def training_step(self, batch, batch_idx):
        inputs, target,bbox = batch['image'],batch['label'],batch['bbox']
        self.pred = self.forward(inputs )
        target_listed = []
        for s in self.deep_supervision_scales:
            if all([i == 1 for i in s]):
                target_listed.append(target)
            else:
                size = [int(np.round(ss*aa)) for ss,aa in zip(s,target.shape[2:])]
                target_downsampled = F.interpolate(target,size=size,mode="nearest")
                target_listed.append(target_downsampled)
        loss = self.loss_fnc(self.pred, target_listed)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch,batch_idx)

    def configure_optimizers(self):
            return torch.optim.SGD(self.model.parameters(), lr=0.1)

    def forward(self,inputs):
        return self.model(inputs)


    def create_model(self):

        # self.device = device
        # creates learner from configs. Loads checkpoint if any exists in self.checkpoints_folder

        model = create_model_from_conf(self.model_params, self.dataset_params)
        # if self.checkpoints_folder:
        #     load_checkpoint(self.checkpoints_folder, model)
        self.batch_size=8

        if (
            self.model_params["arch"] == "DynUNet"
            or self.model_params["arch"] == "nnUNet"
        ):
            if self.model_params["arch"] == "DynUNet":
                num_pool = 4  # this is a hack i am not sure if that's the number of pools . this is just to equalize len(mask) and len(pred)
                ds_factors = list(
                    il.accumulate(
                        [1]
                        + [
                            2,
                        ]
                        * (num_pool - 1),
                        operator.truediv,
                    )
                )
                ds = [1, 1, 1]
                self.deep_supervision_scales = list(
                    map(
                        lambda list1, y: [x * y for x in list1],
                        [
                            ds,
                        ]
                        * num_pool,
                        ds_factors,
                    )
                )

            else:
                num_pool = 5

                self.net_num_pool_op_kernel_sizes =pool_op_kernels_nnunet(self.dataset_params['patch_size'])
                # self.net_num_pool_op_kernel_sizes = [
                #     [2, 2, 2],
                # ] * num_pool
                self.deep_supervision_scales = [[1, 1, 1]] + list(
                    list(i)
                    for i in 1
                    / np.cumprod(np.vstack(self.net_num_pool_op_kernel_sizes), axis=0)
                )[:-1]

            loss_func = DeepSupervisionLoss(
                levels=num_pool, bs=self.batch_size,fg_classes=self.model_params['out_channels']-1
            )
            # cbs += [DownsampleMaskForDS(self.deep_supervision_scales)]

        else:
            loss_func = CombinedLoss(**self.loss_params,bs=self.batch_size,fg_classes=self.model_params['out_channels']-1)
        # if distributed==True:
        #     cbs+=[DistributedTrainer]
        if self.compiled==True:
            model=torch.compile(model)
        return model,loss_func

    @classmethod
    def fromNeptuneRun(
        self,
        project,
        resume_epoch=None,
        run_name=None,
        cbs=[],
        update_nep_run_from_config: dict=None,
        **kwargs
    ):
        '''
        params: resume. If resume is True, load the run by run_name, if run_name is None,'', or most_recent, then the last run is loaded
        params: resume_epoch loads the checkpoint at the given epoch (if on disc) or the the one immediately before.
        params: update_nep_run_from_config. Allows you to tweak existing run, e.g., change batch-size or arch etc..

        '''
        
        # this takes run_name from NeptuneManager and passes it to NeptuneCallback.  I think the callback can itself do all this at init
        Nep = NeptuneManager(project)

        Nep.load_run(run_name=run_name, param_names='default', update_nep_run_from_config=update_nep_run_from_config)
        config_dict = Nep.download_run_params()
        # dest_labels = config_dict["metadata"]["src_dest_labels"]
        # out_channels = out_channels_from_dict_or_cell(dest_labels)
        run_name = Nep.run_name
        # Nep.stop()
        cbs += [
            ReduceLROnPlateau(patience=50),
            NeptuneCallback.from_existing_run(project=project, config_dict  = config_dict, run_name = run_name, nep_run= Nep.nep_run),
            NeptuneCheckpointCallback(
                checkpoints_parent_folder=project.checkpoints_parent_folder,
                resume_epoch=resume_epoch,
            ),
            NeptuneImageGridCallback(
                classes=config_dict['model_params']['out_channels'],
                patch_size= config_dict['dataset_params']['patch_size'],
            ),
        ]

        self = self(
            project=project, config_dict=config_dict, cbs=cbs, **kwargs
        )
        # Nep.nep_run.stop()
        return self

    @classmethod
    def fromExcel(self, project, **kwargs):

        config_dict= ConfigMaker(project,raytune=False).config
        self = self(project=project, config_dict=config_dict, **kwargs)
        return self


def update_nep_run_from_config(nep_run, config):
    for key, value in config.items():
        nep_run[key] = value
    return nep_run




# %%



if __name__ == "__main__":
    # warnings.filterwarnings("ignore", "TypedStorage is deprecated.*")

    torch.set_float32_matmul_precision('medium')
    from fran.utils.common import *
    project_title = "lits32"
    project = Project(project_title=project_title)

    configs= ConfigMaker(project, raytune=False).config
    
    global_props = load_dict(project.global_properties_filename)
# %%
    cpk="/home/ub/code/fran/fran/managers/.neptune/Untitled/LIT-42/checkpoints/epoch=249-step=3500.ckpt"
    D = DataManager(project,dataset_params=configs['dataset_params'],transform_factors=configs['transform_factors'],affine3d=configs['affine3d'])
    D = DataManager.load_from_checkpoint(cpk)
    D.prepare_data()
# %%
    project_name, api_token = get_neptune_config(project)
    nl= NeptuneLogger(
        api_key=api_token,
        project=project_name,
        tags=["simple", "notebook"],
        log_model_checkpoints=True,  # Update to True to log model checkpoints

    )


# %%
    N = nnUNetTrainer(project,configs['dataset_params'],configs['model_params'],configs['loss_params'])
    N = nnUNetTrainer.load_from_checkpoint(cpk,project=project,dataset_params=D.dataset_params)

# %%
    nep = NepMan()
    mcp = ModelCheckpoint()
    NepImg  = NeptuneImageGridCallback(3,patch_size=configs['dataset_params']['patch_size'])
# %%
    # strategy=DDPStrategy(find_unused_parameters=True)
    cbs = [TQDMProgressBar(refresh_rate=3),mcp, nep,NepImg]
    trainer=Trainer(callbacks=cbs,accelerator="gpu",devices=2,precision='16-mixed',logger=nl,
         max_epochs=500, log_every_n_steps=1,num_sanity_val_steps=0,enable_checkpointing=True,default_root_dir=project.checkpoints_parent_folder, 
                    strategy='ddp_find_unused_parameters_true')
# %%
    tm = time.time()
    trainer.fit(model = N,datamodule=D,ckpt_path=cpk)
    # trainer.fit(model = N,train_dataloaders=D.train_dataloader(),val_dataloaders=D.val_dataloader(),ckpt_path='/home/ub/code/fran/fran/.neptune/Untitled/LITS-567/checkpoints/epoch=53-step=2484.ckpt')
    tm_end =time.time()
    diff= tm_end-tm
    print(diff)
# %%

