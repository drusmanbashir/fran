# %%
import torch
import operator
from fastai.callback.fp16 import FP16TestCallback, ModelToHalf, NonNativeMixedPrecision
from fastai.test_utils import synth_learner
from fran.preprocessing.stage0_preprocessors import dec_to_str, folder_name_from_list
from functools import partial
from fastai.callback.schedule import ParamScheduler, combined_cos, one_hot_decode
from fastai.callback.tracker import (
    ReduceLROnPlateau,
)
from torch.cuda.amp.autocast_mode import autocast
from fran.data.dataloader import TfmdDLKeepBBox
from fran.data.dataset import *


from fran.utils.helpers import *
from fran.utils.fileio import *
from fran.utils.imageviewers import *
from fran.transforms.spatialtransforms import *
from fastai.learner import *
from fastai.learner import Learner
from fastcore.foundation import L
from fran.data.dataset import *
from fran.evaluation.losses import *
from fran.architectures.create_network import create_model_from_conf
from fran.callback.neptune import NeptuneManager
from fran.managers.base import *
import fran.transforms.intensitytransforms as intensity
import fran.transforms.spatialtransforms as spatial
from fran.transforms.misc_transforms import DropBBoxFromDataset, BGToMin, FilenameFromBBox
from fran.utils.helpers import *
from fran.callback.neptune import *
from fran.callback.tune import *
from fran.callback.case_recorder import CaseIDRecorder
from neptune.new.types import File

class Learner_Plus(Learner):

    # This learner allows the loss function to return more  than one output (e.g., for combined loss ,dice is additionally reported separately for plotting) '''
    def __init__(self, device=None, *args, **kwargs):
        self.device = device
        super().__init__(*args, **kwargs)

    def _do_one_batch(self):
        self.pred = self.model(*self.xb)
        self('after_pred')
        if len(self.yb):
            self.loss_grad = self.loss_func(self.pred, *self.yb)
            self.loss = self.loss_grad.clone()
        self('after_loss')
        if not self.training or not len(self.yb): return



    def _do_one_batch(self):
        # with torch.autocast(device_type='cuda',dtype=torch.float16):
        self.pred= self.model(*self.xb)
    # self.pred = self.model(*self.xb)
        self("after_pred")
        losses = self.loss_func(self.pred, *self.yb)
        if isinstance(losses, dict):
            self.loss_grad = losses["loss"]
            self.loss_dict = losses
        else:
            self.loss_grad = losses
        before = self.loss_grad.item()
        if before > 100:
            tr()
        self.loss = self.loss_grad.clone()
        self("after_loss")
        if not self.training or not len(self.yb):
            return
        self._do_grad_opt()


    def is_half_precision(self):
        param_dtypes=[]
        for param in self.model.parameters():
            param_dtypes.append(param.dtype)
        return all([p == torch.float16 for p in param_dtypes])

    def fit_one_cycle(
        self: Learner,
        n_epoch,
        lr_max=None,
        div=25.0,
        div_final=1e5,
        pct_start=0.25,
        wd=None,
        moms=None,
        cbs=None,
        reset_opt=False,
    ):
        "Fit `self.model` for `n_epoch` using the 1cycle policy."
        if self.opt is None:
            self.create_opt()
        self.opt.set_hyper("lr", self.lr if lr_max is None else lr_max)
        lr_max = np.array([h["lr"] for h in self.opt.hypers])
        scheds = {
            "lr": combined_cos(pct_start, lr_max / div, lr_max, lr_max / div_final),
            "mom": combined_cos(pct_start, *(self.moms if moms is None else moms)),
        }
        self.fit(
            n_epoch, cbs=ParamScheduler(scheds) + L(cbs), reset_opt=reset_opt, wd=wd
        )

    @property
    def model(self):
        """The model property."""
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self = self.to(self.device)




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
            PredAsList,

            CaseIDRecorder,
        ]  # 12.0 following nnUNet

        self.create_datasets()
        self.create_transforms()
        self.create_dataloaders( max_workers=max_workers, bs=bs, pin_memory=pin_memory)

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

    def create_dataloaders(self, bs=None, max_workers=4, **kwargs):
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

        self.dls = DataLoaders(train_dl, valid_dl)

    def create_learner(self, cbs=[], device=None, **kwargs):
        self.device = device
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
                    accumulate(
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
                self.net_num_pool_op_kernel_sizes = [
                    [2, 2, 2],
                ] * num_pool
                self.deep_supervision_scales = [[1, 1, 1]] + list(
                    list(i)
                    for i in 1
                    / np.cumprod(np.vstack(self.net_num_pool_op_kernel_sizes), axis=0)
                )[:-1]

            loss_func = setup_multioutputloss_nnunet(
                net_numpool=num_pool, batch_dice=True
            )
            cbs += [DownsampleMaskForDS(self.deep_supervision_scales)]

        else:
            loss_func = CombinedLoss(**self.loss_params)
      
        self.cbs += cbs
        learn = Learner_Plus(
            device=self.device,
            dls=self.dls,
            model=model,
            loss_func=loss_func,
            cbs=self.cbs,
            model_dir=None,
            **kwargs
        )
        # learn.to(device)
        print("Training will be done on cuda: {}".format(self.device))
        torch.cuda.set_device(self.device)
        learn.dls = learn.dls.to(torch.device(self.device))
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
    P = Project(project_title="lits"); proj_defaults= P.proj_summary
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

    cbs = [
        ReduceLROnPlateau(patience=50),
        NeptuneCallback(proj_defaults, configs_excel, run_name=None),
        NeptuneCheckpointCallback(proj_defaults.checkpoints_parent_folder),
        NeptuneImageGridCallback(
            classes=out_channels_from_dict_or_cell(
                configs_excel["metadata"]["src_dest_labels"]
            ),
            patch_size=make_patch_size(
                configs_excel["dataset_params"]["patch_dim0"],
                configs_excel["dataset_params"]["patch_dim1"],
            ),
        ),

    ]
# %%
    # cbs =[]
# %%


    La = Trainer.fromExcel(
        proj_defaults,
        bs=2
    )
# %%
    learn = La.create_learner(cbs=cbs, device=1)
    # learn.dls.device=device
# %%

    # model = SwinUNETR(La.dataset_params['patch_size'],1,3)
    # learn.model = model
    learn.fit(n_epoch=1, lr=La.model_params["lr"])
# %%
    for i ,batch in enumerate(learn.dls.valid):
        print(type(batch[0][0]))
# %%
#     a,b = learn.dls.one_batch()
# # %%
#     c = learn.model(a.cuda())
#     c = [cc.to("cuda") for cc in c]
#     torch.save(b, "tmp/mask.pt")

    #  C =CombinedLossDeepSupervision()

    #  C(pred,targ)
    a = La.dls.one_batch()
    pred = learn.model(a[0].cuda())
# %%
    #
    #        targs =  [F.interpolate(targ, size=a.shape[2:] ,mode="nearest") for a in pred]
    #        targs = [tn.squeeze(1) for tn in targs]
    #        if C.apply_activation == True:
    #                 pred=  [C.activation(pred) for pred in pred]
    #        l1 = C.loss2(pred[-1], targs[-1].type(C.mask_dtype))
    #        deep_losses=[]
    #        for n in range(len(pred)):
    #             targ= targs[n]
    #             pred = pred[n]
    #             l2 = C.dice_loss(pred, targ)
    #             deep_losses.append(l2['loss_dice'])
    #        l2['loss_dice']= torch.tensor(deep_losses).mean(0)
    #        final =C.theta*l1 + (1-C.theta)*l2['loss_dice']
# %%

# %%
    cbs = []
    cbs += [
        ReduceLROnPlateau(patience=10),
        NeptuneCheckpointCallback(),
        NeptuneCallback(proj_defaults, configs_excel, run_name=None),
    ]

    # La = LearnerManager(proj_defaults=proj_defaults,config_dict= configs,cbs=cbs)
    #
    #     configs = load_config_from_workbook(proj_defaults.configuration_filename, raytune=False)
# %%
    La = Trainer(proj_defaults=proj_defaults, config_dict=configs_excel)
    #     La.create_dataloaders(train_list_w,valid_list_w, bs=5,max_workers=5,pin_memory=True)
# %%
    #
    #
# %%
    learn = La.create_learner(cbs=cbs)
    #     learn.load("model",with_opt=True)
# %%
    #     learn.fit(n_epoch=150, lr=config_dict['model_params']['lr'])
    # #     # state_dict= torch.load(chkpoint_filename)
    # #     # model = model_from_config(conf)
    # #     # model_best = load_model_from_raytune_trial(folder_name)
    # #     # model_best.load_state_dict(state_dict['model'])
# %%
    #     run_name = "KITS-1705"
    #     config_dict = configs
    #     nep_mode= 'async'
    #     param_names = config_dict.keys()
    #     param = 'dataset_params'
    #
    #

