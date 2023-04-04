# %%
from SimpleITK import ImageViewer_SetProcessDelay
from fastai.data.core import DataLoaders
# from fastai.distributed import *
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
from fastai.learner import *
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
            self, proj_defaults, config_dict, cbs=[], bs=None, max_workers=0, pin_memory=True, device:Union[int,None]=None
    ):
        '''
        device: If None, it is passed as such. Otherwise supply an integer for cuda device or 'auto' for system to determine a free GPU.
        '''
        self.device= device
        self.proj_defaults = proj_defaults
        self.assimilate_config(config_dict)

        self.patch_based = bool(self.metadata["patch_based"])

        
        self.train_list, self.valid_list, _ = get_fold_case_ids(
            fold=self.metadata["fold"],
            json_fname=self.proj_defaults.validation_folds_filename,
        )
        self.cbs = cbs + [
            TerminateOnNaNCallback_ub,
            # GradientClip(max_norm=12.0),

            CaseIDRecorder,
        ]  # 12.0 following nnUNet

        self.create_datasets()
        self.create_transforms()
        self.create_dataloaders( max_workers=max_workers, bs=bs, pin_memory=pin_memory,device=self.device)

    def assimilate_config(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

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
        folder_name = get_raytune_folder_from_trialname(proj_defaults, trial_name)
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
        if self.patch_based == True:
            parent_folder = self.proj_defaults.patches_folder
        else:
            parent_folder = self.proj_defaults.whole_images_folder
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
        self.set_dataset_folders()
        bboxes_fname = self.dataset_folder / "bboxes_info"
        self.train_ds = ImageMaskBBoxDataset(
            self.proj_defaults,
            self.train_list,
            bboxes_fname,
            self.dataset_params["class_ratios"],
        )
        self.valid_ds = ImageMaskBBoxDataset(
            self.proj_defaults,
            self.valid_list,
            bboxes_fname,
        )

    def create_dataloaders(self, bs=None, max_workers=4,device=None, **kwargs):
        if bs == None:
            bs = self.dataset_params["bs"]
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
            cbs += [DownsampleMaskForDS(self.deep_supervision_scales)]

        else:
            loss_func = CombinedLoss(**self.loss_params,bs=self.dls.bs,fg_classes=self.model_params['out_channels']-1)
      
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
        if distributed==True:
             learn.model  = torch.nn.DataParallel(learn.model)
        else:
            print("Training will be done on cuda: {}".format(self.device))
            # learn.dls = learn.dls.to(torch.device(self.device))
            torch.cuda.set_device(self.device)
        if compile==True:

            print("Compiling model")
            learn.model = torch.compile(learn.model)
       
        learn.dls.cuda()
        learn.to_non_native_fp16()
        return learn

    @property
    def device(self):
        """The device property."""
        return self._device
    @device.setter
    def device(self, value):
        assert value in [None]+list(range(100)), "Print device can only be None or int "
        if isinstance(value,int):
            self._device = value
        elif not hasattr(self,'_device'):
            self._device= get_available_device()


    @classmethod
    def fromNeptuneRun(
        self,
        proj_defaults,
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
        Nep = NeptuneManager(proj_defaults)

        Nep.load_run(run_name=run_name, param_names='default', update_nep_run_from_config=update_nep_run_from_config)
        config_dict = Nep.download_run_params()
        dest_labels = config_dict["metadata"]["src_dest_labels"]
        out_channels = out_channels_from_dict_or_cell(dest_labels)
        run_name = Nep.run_name
        # Nep.stop()
        cbs += [
            ReduceLROnPlateau(patience=50),
            NeptuneCallback.from_existing_run(proj_defaults=proj_defaults, config_dict  = config_dict, run_name = run_name, nep_run= Nep.nep_run),
            NeptuneCheckpointCallback(
                checkpoints_parent_folder=proj_defaults.checkpoints_parent_folder,
                resume_epoch=resume_epoch,
            ),
            NeptuneImageGridCallback(
                classes=out_channels,
                patch_size=make_patch_size(
                    config_dict["dataset_params"]["patch_dim0"],
                    config_dict["dataset_params"]["patch_dim1"],
                ),
            ),
        ]

        self = self(
            proj_defaults=proj_defaults, config_dict=config_dict, cbs=cbs, **kwargs
        )
        # Nep.nep_run.stop()
        return self

    @classmethod
    def fromExcel(self, proj_defaults, **kwargs):

        config_dict= ConfigMaker(proj_defaults.configuration_filename, raytune=False).config
        self = self(proj_defaults=proj_defaults, config_dict=config_dict, **kwargs)
        return self


def update_nep_run_from_config(nep_run, config):
    for key, value in config.items():
        nep_run[key] = value
    return nep_run




# %%


if __name__ == "__main__":

    from fran.utils.common import *
    project_title = "lits"
    P = Project(project_title=project_title); proj_defaults= P.proj_summary
    from fran.managers.tune import get_raytune_folder_from_trialname

    # trial_name = "kits_675_080"
    # folder_name = get_raytune_folder_from_trialname(proj_defaults, trial_name)
    # checkpoints_folder = folder_name / ("model_checkpoints")
    # ray_conf_fn = folder_name / "params.json"
    # config_dict_ray_trial = load_dict(ray_conf_fn)
    # chkpoint_filename = list((folder_name/("model_checkpoints")).glob("model*"))[0]
    #

    configs_excel= ConfigMaker(proj_defaults.configuration_filename, raytune=False).config
    

# %%
# # %%
#     #     run_name = None
#     run_name = "KITS-2490"
#     La = Trainer.fromNeptuneRun(
#         proj_defaults,
#         run_name=run_name,
#         update_nep_run_from_config=False,
#     )
#     #
# # %%
#     learn = La.create_learner(gbs=[], device=device)
# #     # learn.model = model
#     learn.fit(n_epoch=500, lr=1e-6)
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
    #     NeptuneCallback(proj_defaults, configs_excel, run_name=None),
    #     NeptuneCheckpointCallback(proj_defaults.checkpoints_parent_folder),
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
    cbs =[]


    La = Trainer.fromExcel(
        proj_defaults,
        bs=2
    )
# %%
#     ds = La.valid_ds
#     bb = [b for b in ds.bboxes_per_id if b[0]['case_id']=='lits-9']
# # %%
#     for x , bb in enumerate(La.dls.valid):
#         a,b,c = bb
#         ids  = [get_case_id_from_filename(None,Path(cc)) for cc in c]
#         if any([i == 'lits-9' for i in ids]):
#               inx = np.where(np.array(ids)=='lits-9')
#               img = a[inx]
#               mask = b[inx]
#               
# # %%
#     ImageMaskViewer([img[0,0].detach().cpu(), mask[0,0].detach().cpu()])
# %%
    learn = La.create_learner(cbs=cbs, compile=False,distributed=False)
# %%
    ImageMaskViewer([a.detach().cpu()[0,0],b.detach().cpu()[0,0]])
# %%

    # model = SwinUNETR(La.dataset_params['patch_size'],1,3)
    # learn.model = model
    learn.fit(n_epoch=30, lr=La.model_params["lr"])
## %%
# %%
# %%
    bboxes_fname = La.dataset_folder / "bboxes_info"
    ds = ImageMaskBBoxDataset(
                La.proj_defaults,
                La.train_list,
                bboxes_fname,
                [0,0 ,100]
            )

# %%
    present =[]
    for x in range(len(ds)):
        a,b,c = ds[x]
        s = c['bbox_stats']
        labs =[a['label'] for a in s]
        present.append(2 in labs)
    sum(present)
# %%
    x = 2
    a,b,c = ds[x]
    ImageMaskViewer([a,b])
# %%
# %%
    a,b,c = La.dls.one_batch()
# %%
# b n m , . zdfghgbuhy
