# %%
from typing import Any
from neptune.types import File
import random
import lightning as pl
import torch.nn.functional as F
import os
from torchvision.utils import make_grid
import torch
from fran.transforms.spatialtransforms import one_hot
import neptune
import ast
from fran.utils.config_parsers import *
from fran.utils.fileio import load_json, load_yaml

# from fran.managers.learner_plus import *
from fran.utils.helpers import *
from fran.utils.config_parsers import *

from lightning.pytorch.callbacks import Callback
try:
    hpc_settings_fn = os.environ["HPC_SETTINGS"]
except:
    pass

from fran.utils.colour_palette import colour_palette

_ast_keys = ["dataset_params,patch_size", "metadata,src_dest_labels"]
_immutable_keys = [
    "fold"
]  # once set in a particular runs these will not be changed without corrupting the run
str_to_key = lambda string: string.split(",")


def normalize(tensr, intensity_percentiles=[0.0, 1.0]):
    tensr = (tensr - tensr.min()) / (tensr.max() - tensr.min())
    tensr = tensr.to("cpu", dtype=torch.float32)
    qtiles = torch.quantile(tensr, q=torch.tensor(intensity_percentiles))

    vmin = qtiles[0]
    vmax = qtiles[1]
    tensr[tensr < vmin] = vmin
    tensr[tensr > vmax] = vmax
    return tensr
def dictionary_fix_ast(dictionary: dict):
    for keys in map(str_to_key, _ast_keys):
        val = dictionary[keys[0]][keys[1]]
        dictionary[keys[0]][keys[1]] = ast.literal_eval(val)
    return dictionary


def is_remote(model_dir):
    hpc_settings = load_yaml(hpc_settings_fn)
    if hpc_settings["hpc_storage"] in model_dir:
        return True
    else:
        return False


def get_neptune_config():
    """
    Returns particular project workspace
    """
    commons = load_yaml(common_vars_filename)
    project_name = commons['neptune_project']
    api_token = commons["neptune_api_token"]
    return project_name, api_token


def get_neptune_project(proj_defaults, mode):
    """
    Returns project instance based on project title
    """

    project_name, api_token = get_neptune_config()
    return neptune.init_project(project=project_name, api_token=api_token, mode=mode)


class NeptuneImageGridCallback(Callback):
    def __init__(
        self,
        classes,
        patch_size,
        grid_rows=6,
        imgs_per_batch=4,
        publish_deep_preds=False,
        apply_activation=True,
        epoch_freq=5,  # skip how many epochs.
    ):
        if not isinstance(patch_size, torch.Size):
            patch_size = torch.Size(patch_size)
        self.stride = int(patch_size[0] / imgs_per_batch)
        store_attr()

    #
    def on_train_start(self, trainer, pl_module):
        trainer.store_preds = False  # DO NOT SET THIS TO TRUE. IT WILL BUG  
        len_dl = int(len(trainer.train_dataloader) / trainer.accumulate_grad_batches)
        self.freq = np.maximum(2, int(len_dl / self.grid_rows))

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.epoch_freq == 0:
            super().on_train_epoch_start(trainer, pl_module)
            self.grid_imgs = []
            self.grid_preds = []
            self.grid_labels = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self.validation_grid_created = False

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.current_epoch % self.epoch_freq == 0:
            if trainer.global_step % self.freq == 0:
                trainer.store_preds = True
            else:
                trainer.store_preds = False
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.store_preds == True:
            self.populate_grid(pl_module, batch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.store_preds == True:
        # if trainer.current_epoch % self.epoch_freq == 0:
            if self.validation_grid_created == False:
                self.populate_grid(pl_module, batch)
                self.validation_grid_created = True

    #
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.epoch_freq == 0 and len(self.grid_imgs) > 0:
            grd_final = []
            for grd, category in zip(
                [self.grid_imgs, self.grid_preds, self.grid_labels],
                ["imgs", "preds", "lms"],
            ):
                grd = torch.cat(grd)
                # if category == "imgs":
                #     grd = normalize(grd)
                grd_final.append(grd)
            grd = torch.stack(grd_final)
            grd2 = (
                grd.permute(1, 0, 2, 3, 4)
                .contiguous()
                .view(-1, 3, grd.shape[-2], grd.shape[-1])
            )
            grd3 = make_grid(grd2, nrow=self.imgs_per_batch * 3,scale_each=True)
            grd4 = grd3.permute(1, 2, 0)
            grd4 = np.array(grd4)
            trainer.logger.experiment["images"].append(File.as_image(grd4))

    def populate_grid(self, pl_module, batch):
        def _randomize():
            n_slices = img.shape[-1]
            batch_size = img.shape[0]
            self.slices = [
                random.randrange(0, n_slices) for i in range(self.imgs_per_batch)
            ]
            self.batches = [
                random.randrange(0, batch_size) for i in range(self.imgs_per_batch)
            ]

        img = batch["image"].cpu()

        label = batch["lm"].cpu()
        label = label.squeeze(1)
        label = one_hot(label, self.classes, axis=1)
        # pred = torch.rand(img.shape,device='cuda')
        # pred.unsqueeze_(0) # temporary hack)
        # pred = pred.cpu()
        pred = pl_module.pred
        if isinstance(pred, Union[list, tuple]):
            pred = pred[0]
        elif pred.dim() == img.dim() + 1:  # deep supervision
            pred = pred[:, 0, :]  # Bx1xCXHXWxD

        # if self.apply_activation == True:
        pred = F.softmax(pred.to(torch.float32), dim=1)

        _randomize()
        img, label, pred = (
            self.img_to_grd(img),
            self.img_to_grd(label),
            self.img_to_grd(pred),
        )
        img, label, pred = (
            self.scale_tensor(img),
            self.assign_colour(label),
            self.assign_colour(pred),
            # self.fix_channels(label),
            # self.fix_channels(pred),
        )

        self.grid_imgs.append(img)
        self.grid_preds.append(pred)
        self.grid_labels.append(label)

    def img_to_grd(self, batch):
        imgs = batch[self.batches, :, :, :, self.slices].clone()
        return imgs

    def assign_colour(self,tnsr):
        argmax_tensor= torch.argmax(tnsr,dim=1)
        B, H, W = argmax_tensor.shape
        rgb_tensor = torch.zeros((B, 3, H, W), dtype=torch.uint8)

        for key, color in colour_palette.items():
            mask = argmax_tensor == key
            for channel in range(3):
                rgb_tensor[:, channel, :, :][mask] = color[channel]
        return rgb_tensor

    def scale_tensor(self, tnsr):
        min,max = tnsr.min(), tnsr.max()
        rng = max-min
        tnsr= tnsr.repeat(1, 3, 1, 1)
        tnsr2 = tnsr.clone()
        t3 = tnsr2-min
        t4 = t3/rng
        t5 = t4*255
        t6 = torch.clamp(t5,min=0,max=255)
        return t6

# %%

if __name__ == "__main__":
    P = Project(project_title="lits")
    proj_defaults = P
    config = ConfigMaker(proj_defaults.configuration_filename, raytune=False).config

# %%
    def process_html(fname="case_id_dices_valid.html"):
        df = pd.read_html(fname)[0]
        cols = df.columns
        df = df.drop([col for col in cols if "Unnamed" in col], axis=1)
        df = df.drop(["loss_dice", "loss_dice_label1"], axis=1)
        df.dropna(inplace=True)
        return df

    from fran.utils.common import *
    project_title = "lits"
    project = Project(project_title=project_title)

    # trial_name = "kits_675_080"
    # folder_name = get_raytune_folder_from_trialname(project, trial_name)
    # checkpoints_folder = folder_name / ("model_checkpoints")
    # ray_conf_fn = folder_name / "params.json"
    # config_dict_ray_trial = load_dict(ray_conf_fn)
    # chkpoint_filename = list((folder_name/("model_checkpoints")).glob("model*"))[0]
    #
