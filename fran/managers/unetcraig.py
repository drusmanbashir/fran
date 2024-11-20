# %%
import ipdb
from torch.utils.data import DataLoader, Subset
from fran.evaluation.craigloss import DeepSupervisionLossCraig, CombinedLoss
from fran.extra.deepcore.deepcore.met.met_utils import submodular_optimizer
from fran.extra.deepcore.deepcore.met.met_utils.euclidean import euclidean_dist_pair_np
tr = ipdb.set_trace
import numpy as np
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import itertools as il
import operator
from fran.architectures.create_network import (
    create_model_from_conf_nnUNetCraig,
    pool_op_kernels_nnunet,
)
from fran.architectures.create_network import pool_op_kernels_nnunet
from fran.managers.unet import UNetManager, UNetManagerFabric

class UNetManagerCraig(UNetManagerFabric):
    def __init__(
        self,
        project,
        config,
        lr=None,
    ):
        self.grad_z_l = None
        self.capture_grads = True
        super().__init__(
            project=project,
            config=config,
            lr=lr,
        )

        # self.dataset_params=config['dataset_params']

    # def on_validation_model_eval(self) -> None:
    # self.model.train()
    # return super().on_validation_model_eval()

    def create_loss_fnc(self):
        if self.model_params["arch"] == "nnUNet":
            num_pool = 5
            self.net_num_pool_op_kernel_sizes = pool_op_kernels_nnunet(
                self.plan["patch_size"]
            )
            self.deep_supervision_scales = [[1, 1, 1]] + list(
                list(i)
                for i in 1
                / np.cumprod(np.vstack(self.net_num_pool_op_kernel_sizes), axis=0)
            )[:-1]
            loss_func = DeepSupervisionLossCraig(
                levels=num_pool,
                deep_supervision_scales=self.deep_supervision_scales,
                fg_classes=self.model_params["out_channels"] - 1,
                softmax=True,
                compute_grad=True,  # i want to capture grad in the model
            )
            return loss_func

        elif (
            self.model_params["arch"] == "DynUNet"
            or self.model_params["arch"] == "DynUNet_UB"
        ):
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
            loss_func = DeepSupervisionLossCraig(
                levels=num_pool,
                deep_supervision_scales=self.deep_supervision_scales,
                fg_classes=self.model_params["out_channels"] - 1,
                compute_grad=self.capture_grads,
            )
            return loss_func

        else:
            loss_func = CombinedLoss(
                **self.loss_params, fg_classes=self.model_params["out_channels"] - 1
            )
            return loss_func

    def compute_gradient_norm(self):
        """
        Computes the norm of the stored gradient of the pre-activation tensor z_L.
        """
        if self.model.grad_L_x is not None:
            grad_norm = torch.norm(self.model.grad_L_x, p=2, dim=1)
            return grad_norm
        return None

    def calc_gradient(self, index=None):
        """
        This function calculates the gradients for the data points.
        """
        self.model.eval()  # Set the model to evaluation mode

        batch_loader = DataLoader(
            (
                self.dataset_params
                if index is None
                else Subset(self.dataset_params, index)
            ),
            batch_size=5,
            num_workers=5,
        )

        gradients = []
        self.embedding_dim = self.model.get_last_layer().in_channels

        for inputs, targets in batch_loader:
            self.model_optimizer.zero_grad()  # Reset gradients
            outputs = self.model(inputs.to(self.args.device))
            loss = self.criterion(
                outputs.requires_grad_(True), targets.to(self.args.device)
            ).sum()
            batch_num = targets.shape[0]

            with torch.no_grad():
                # Compute bias and weight gradients
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                tr()
                weight_parameters_grads = self.model.embedding_recorder.embedding.view(
                    batch_num, 1, self.embedding_dim
                ).repeat(1, self.args.num_classes, 1) * bias_parameters_grads.view(
                    batch_num, self.args.num_classes, 1
                ).repeat(
                    1, 1, self.embedding_dim
                )
                gradients.append(
                    torch.cat(
                        [bias_parameters_grads, weight_parameters_grads.flatten(1)],
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )

        gradients = np.concatenate(gradients, axis=0)
        self.model.train()  # Switch back to training mode
        return euclidean_dist_pair_np(gradients)


    def _common_step(self, batch, batch_idx):
        loss, loss_dict = super()._common_step(batch,batch_idx)
        grad_L_z= self.loss_fnc.grad_L_z
        return loss,loss_dict,grad_L_z

    def validation_step(self, batch, batch_idx):
        loss, loss_dict ,grad_L_z = self._common_step(batch, batch_idx)
        self.log_losses(loss_dict, prefix="val")
        return loss, grad_L_z


    def training_step(self, batch, batch_idx):

        loss, loss_dict ,grad_L_z = self._common_step(batch, batch_idx)
        self.log_losses(loss_dict, prefix="train")
        return loss, grad_L_z


    def select_subset_with_craig(self):
        """
        Select a subset of the training data using CRAIG.
        """
        selected_subset = submodular_optimizer(
            model=self.model,
            dst_train=self.dataset_params,
            dst_val=self.val_dataloader.dataset,
            args=self.args,
            greedy=self.args.greedy,
            balance=self.args.balance,
        )
        return selected_subset

    def create_model(self):
        print("*" * 100)
        print("Fixing CRAIG model temporarily. Alter code when CRAIG is implemented")
        model = create_model_from_conf_nnUNetCraig(
            self.model_params, self.model_params['deep_supervision']
        )
        return model


def init_unet_trainer(project, config, lr):
    N = UNetManagerCraig(
        project,
        config,
        lr=lr,
    )
    return N


# %%
