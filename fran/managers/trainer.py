# %%
import time
from SimpleITK import ImageViewer_SetProcessDelay
from fastai.data.core import DataLoaders
from fastai.distributed import DistributedTrainer
from fastai.torch_core import delegates

import torch
import operator
from fran.preprocessing.stage0_preprocessors import folder_name_from_list
from functools import partial
from fastai.callback.tracker import (
    ReduceLROnPlateau,
)
from fran.data.dataloader import TfmdDLKeepBBox
from fran.data.dataset import *

import itertools as il
from fran.utils.helpers import *
from fran.utils.fileio import *
from fran.utils.imageviewers import *
from fran.transforms.spatialtransforms import *
# from fastai.learner import *
from fastai.learner import Learner
from fran.data.dataset import *
from fran.evaluation.losses import *
from fran.architectures.create_network import create_model_from_conf, pool_op_kernels_nnunet
from fran.callback.neptune import NeptuneManager
from fran.managers.base import *
import fran.transforms.intensitytransforms as intensity
import fran.transforms.spatialtransforms as spatial
from fran.transforms.misc_transforms import  FilenameFromBBox
from fran.utils.helpers import *
from fran.callback.neptune import *
from fran.callback.tune import *
from fran.callback.case_recorder import CaseIDRecorder

def compute_bs(project,bs=2,distributed=False,step=1):
        '''
        bs = starting bs
        
        '''
    
    
        print("Computing optimal batch-size for available vram")
        if distributed==True:
            step =step*2 
        while True:
            La = Trainer.fromExcel(
                project,
                bs=bs,
                dummy_ds=bs*2,

            )
            learn = La.create_learner(cbs=[], compile=False,distributed=distributed)
            try:
                print("Trial bs: {}".format(bs))
                learn.fit(1)
            except RuntimeError:
                print("Final broken bs: {}\n-----------------".format(bs))
                bs  = bs-step*2
                print("\n----- Accepted bs: {}".format(bs))
                break
            bs+=step
            del learn
            del La
            gc.collect()
            torch.cuda.empty_cache()
        return bs



def load_model_from_raytune_trial(folder_name,out_channels):
    #requires params.json inside raytune trial
    params_dict = load_json(Path(folder_name)/"params.json")
    model =create_model_from_conf(params_dict,out_channels)
    
    folder_name/("model_checkpoints")
    list((folder_name/("model_checkpoints")).glob("model*"))[0]
    load_checkpoint(folder_name / ("model_checkpoints"), model)
    # state_dict= torch.load(chkpoint_filename)
    # model.load_state_dict(state_dict['model'])
    return  model
class Trainer:
    def __init__(
            self, project, config_dict, cbs=[], bs=2, max_workers=0, pin_memory=True, device='cuda', dummy_ds:int=0
    ):
        '''
        dummy_ds if >0, creates a short ds=dummy_ds. Used to run quick fits (to estimate vram needs)

        '''
        store_attr('device,project,dummy_ds')
        self.assimilate_config(config_dict)

        self.train_list, self.valid_list =project.get_train_val_files(config_dict['metadata']['fold'])

        self.cbs = cbs + [
            TerminateOnNaNCallback_ub,
            # GradientClip(max_norm=12.0),

            CaseIDRecorder,
        ]  # 12.0 following nnUNet

        self.set_dataset_folders()
        self.create_datasets()
        self.create_transforms()
        self.create_dataloaders( max_workers=max_workers, bs=bs, pin_memory=pin_memory,device=self.device)

    def assimilate_config(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

        global_properties = load_dict(self.project.global_properties_filename)
        self.dataset_params['clip_range']=global_properties["intensity_clip_range"]
        self.dataset_params['mean_fg']=global_properties["mean_fg"]
        self.dataset_params['std_fg']=global_properties["std_fg"]

    def generate_config(self):
        config_dict = {}
        for key in [
            "after_batch_affine",
            "after_item_intensity",
            "after_item_spatial",
            "dataset_params",
            "loss_params",
            "model_params",
        ]:
            config_dict[key] = getattr(self, key)
        return config_dict

    @classmethod
    @delegates(__init__)
    def from_tune_trial(self, trial_name, **kwargs):
        super().__init__(**kwargs)
        folder_name = get_raytune_folder_from_trialname(project, trial_name)
        self.model = load_model_from_raytune_trial(folder_name)

    def _create_augmentations(self):
        self.intensity_augs = []
        self.spatial_augs = []
        self.probabilities_intensity = []
        self.probabilities_spatial = []
        for key, value in self.after_item_intensity.items():
            func = getattr(intensity, key)
            out_fnc = partial(func, factor_range=value[0])
            self.intensity_augs.append(out_fnc)
            self.probabilities_intensity.append(value[1])

        for key, value in self.after_item_spatial.items():
            self.spatial_augs.append(getattr(spatial, key))
            self.probabilities_spatial.append(value)

    def set_dataset_folders(self):
        prefixes, value_lists = ["spc", "dim"], [
            self.dataset_params["spacings"],
            self.dataset_params["src_dims"],
        ]
        if bool(self.metadata['patch_based']) == True:
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
        self._create_augmentations()
        after_item_intensity = TrainingAugmentations(
            augs=self.intensity_augs, p=self.probabilities_intensity
        )
        after_item_spatial = TrainingAugmentations(
            augs=self.spatial_augs, p=self.probabilities_spatial
        )
        self.after_item_train = Pipeline(
            [

                FilenameFromBBox,
                MaskLabelRemap(self.metadata["src_dest_labels"]),
                PermuteImageMask(p=0.3),
                after_item_intensity,
                after_item_spatial,
                CropImgMask(self.dataset_params["src_dims"], 3),
                PadDeficitImgMask(self.dataset_params["src_dims"], 3),
                Unsqueeze,
            ]
        )
        self.after_item_valid = Pipeline(
            [

                FilenameFromBBox,
                MaskLabelRemap(self.metadata["src_dest_labels"]),
                Unsqueeze,
            ],
        )
        self.after_batch_train = Pipeline(
            [
                AffineTrainingTransform3D(**self.after_batch_affine),
                CropImgMask(self.dataset_params["patch_size"], 5),
                PadDeficitImgMask(self.dataset_params["patch_size"], 5),
            ]
        )
        self.before_batch = [],
        self.after_batch_valid = Pipeline(
            [
                CropImgMask(self.dataset_params["patch_size"], 5),
                PadDeficitImgMask(self.dataset_params["patch_size"], 5),
            ]
        )

    def create_datasets(
        self,
    ):
        bboxes_fname = self.dataset_folder / "bboxes_info"
        if self.dummy_ds> 0:
            self.train_list = self.train_list[:self.dummy_ds]
            self.valid_list= self.train_list[:1]
        self.train_ds = ImageMaskBBoxDataset(
            self.train_list,
            bboxes_fname,
            self.dataset_params["class_ratios"],
        )
        self.valid_ds = ImageMaskBBoxDataset(
            self.valid_list,
            bboxes_fname,
        )

    def create_dataloaders(self, bs, max_workers=4,device=None, **kwargs):
        if not device:
            device= 'cuda'
        train_dl = TfmdDLKeepBBox(
            self.train_ds,
            shuffle=True,
            bs=bs,
            num_workers=np.minimum(max_workers, bs * 2),
            after_item=self.after_item_train,
            before_batch=self.before_batch,
            after_batch=self.after_batch_train,
            **kwargs
        )
        valid_dl = TfmdDLKeepBBox(
            self.valid_ds,
            shuffle=False,
            bs=bs,
            num_workers=np.minimum(max_workers, bs * 4),
            after_item=self.after_item_valid,
            before_batch=self.before_batch,
            after_batch=self.after_batch_valid,
            **kwargs
        )

        self.dls = DataLoaders(train_dl, valid_dl,device=device)

    def create_learner(self, cbs=[], distributed=False,compile=False, **kwargs):
        # self.device = device
        # creates learner from configs. Loads checkpoint if any exists in self.checkpoints_folder

        model = create_model_from_conf(self.model_params, self.dataset_params)
        # if self.checkpoints_folder:
        #     load_checkpoint(self.checkpoints_folder, model)

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
                levels=num_pool, bs=self.dls.bs,fg_classes=self.model_params['out_channels']-1,device=self.device
            )
            if not any([c.name=='downsample_mask_for_ds' for c in self.cbs]):
                cbs += [DownsampleMaskForDS(self.deep_supervision_scales)]

        else:
            loss_func = CombinedLoss(**self.loss_params,bs=self.dls.bs,fg_classes=self.model_params['out_channels']-1)
        if distributed==True:
            cbs+=[DistributedTrainer]
      
        self.cbs += cbs
        learn = Learner(
            dls=self.dls,
            model=model,
            loss_func=loss_func,
            cbs=self.cbs,
            model_dir=None,
            **kwargs
        )
        # learn.to(device)
        if compile==True:
            print("Compiling model")
            learn.model = torch.compile(learn.model)
             # learn.model  = torch.nn.DataParallel(learn.model)

        else:
            print("Training will be done on cuda: {}".format(self.device))
            # learn.dls = learn.dls.to(torch.device(self.device))
            # torch.cuda.set_device(self.device)

       
        learn.dls.cuda()
        learn.to_non_native_fp16()
        return learn
    #
    # @property
    # def device(self):
    #     """The device property."""
    #     return self._device
    # @device.setter
    # def device(self, value):
    #     assert value in [None]+list(range(100)), "Print device can only be None or int "
    #     if isinstance(value,int):
    #         self._device = value
    #     # elif not hasattr(self,'_device'):
    #     #     self._device= get_available_device()


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

    from fran.utils.common import *
    project_title = "lits"
    project = Project(project_title=project_title)
    from fran.managers.tune import get_raytune_folder_from_trialname

    # trial_name = "kits_675_080"
    # folder_name = get_raytune_folder_from_trialname(project, trial_name)
    # checkpoints_folder = folder_name / ("model_checkpoints")
    # ray_conf_fn = folder_name / "params.json"
    # config_dict_ray_trial = load_dict(ray_conf_fn)
    # chkpoint_filename = list((folder_name/("model_checkpoints")).glob("model*"))[0]
    #

    configs= ConfigMaker(project, raytune=False).config
    

# %%
    cbs = [
            ReduceLROnPlateau(patience=50),
            NeptuneCallback(project, configs, run_name=None),
            NeptuneCheckpointCallback(project.checkpoints_parent_folder),
            NeptuneImageGridCallback(
                classes=out_channels_from_dict_or_cell(
                    configs["metadata"]["src_dest_labels"]
                ),
                patch_size=make_patch_size(
                    configs["dataset_params"]["patch_dim0"],
                    configs["dataset_params"]["patch_dim1"],
                ),
            ),
            #
        ]
# %%
    La = Trainer.fromExcel(project,cbs=cbs)
# %%
#     #     run_name = None
#     run_name = "KITS-2490"
#     La = Trainer.fromNeptuneRun(
#         project,
#         run_name=run_name,
#         update_nep_run_from_config=False,
#     )
#     #



# %%
# %%
    learn = La.create_learner(distributed=True)
#     # learn.model = model
# %%
    learn.fit(n_epoch=1, lr=1e-6)
# # # %%

    #     La.dataset_params['fake_tumours']=True
    #     La.create_transforms()
    #     La.create_dataloaders()
    #     learn = La.create_learner()
# %%
    #     learn.fit(n_epoch=350, lr=configs_excel['model_params']['lr'])

    # for a ,b in enumerate(La.dls.train):
    #     print (b[0].shape,b[1].shape)
    #
    # cbs = [
    #     ReduceLROnPlateau(patience=50),
    #     NeptuneCallback(project, configs_excel, run_name=None),
    #     NeptuneCheckpointCallback(project.checkpoints_parent_folder),
    #     NeptuneImageGridCallback(
    #         classes=out_channels_from_dict_or_cell(
    #             configs_excel["metadata"]["src_dest_labels"]
    #         ),
    #         patch_size=make_patch_size(
    #             configs_excel["dataset_params"]["patch_dim0"],
    #             configs_excel["dataset_params"]["patch_dim1"],
    #         ),
    #     ),
    #
    # ]
# %%



