# %o%
from torch.autograd.functional import jacobian
from fastcore.basics import store_attr
import numpy as np
from monai.utils.enums import DiceCEReduction, LossReduction
import lightning.pytorch as pl
from monai.utils.module import look_up_option
from typing import Callable, Optional
import torch.nn as nn
import itertools as il

import torch
import ipdb

from nnunet.utilities.nd_softmax import softmax_helper
from torchvision.utils import Union

from fran.utils.helpers import range_inclusive

tr = ipdb.set_trace
from monai.losses import DiceLoss, DiceCELoss
import torch.nn.functional as F

# Cell


def softmax_wrapper(input_tensor):
    return F.softmax(input_tensor, 1)


def softmax_derivative(softmax_output):
    """
    Compute the derivative (Jacobian matrix) of the softmax function.

    Args:
    - softmax_output (torch.Tensor): The output from the softmax function (probabilities).

    Returns:
    - torch.Tensor: The derivative of the softmax function, represented as a Jacobian matrix.
    """
    # Initialize a tensor to hold the Jacobian matrix

    B, C, W, H, D = softmax_output.shape
    softmax_output = softmax_output.view(B, C, -1)  # Shape [B, C, W*H*D]

    # Create an empty tensor for the Jacobian of shape [B, C, C, W*H*D]
    jacobian = torch.zeros(B, C, C, W * H * D, device=softmax_output.device)
    for i in range(C):
        for j in range(C):
            L = softmax_output[:, i, :]
            R = softmax_output[:, j, :]
            if i == j:
                jacobian[:, i, j, :] = L * (1 - R)  # Diagonal term
            else:
                jacobian[:, i, j, :] = -L * R  # Off-diagonal term

    jacobian = jacobian.view(B, C, C, W, H, D)
    return jacobian


class _DiceCELossMultiOutput(DiceCELoss):
    def __init__(
        self,
        include_background: bool = True,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        super().__init__()
        dice_reduction = LossReduction.NONE
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=True,
            sigmoid=sigmoid,
            softmax=False,  # softmax will be applied here
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=dice_reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.softmax = softmax
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        # self.reduction=reduction

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, compute_grad=False
    ) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )

        if self.softmax == True:
            input_activated = F.softmax(input, 1)

        dice_loss_unreduced = self.dice(input_activated, target)
        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean
        else:
            f = torch.sum
        dice_loss_reduced = f(dice_loss_unreduced)
        ce_loss = self.ce(input, target)
        total_loss: torch.Tensor = (
            self.lambda_dice * dice_loss_reduced + self.lambda_ce * ce_loss
        )
        # return total_loss, ce_loss, dice_loss_reduced , dice_loss_unreduced.squeeze((2,3,4))
        losses_for_logging = {
            "loss":total_loss.item(),
            "loss_ce": ce_loss.item(),
            "loss_dice": dice_loss_reduced.detach(),
        }

        dice_loss_unreduced =  dice_loss_unreduced.squeeze(2).squeeze(2).squeeze(2).detach()
        dice_loss_channels= self.set_labels_channels(dice_loss_unreduced)
        dice_loss_batches = self.set_labels_batches(dice_loss_unreduced)
        losses_for_logging.update(dice_loss_channels)
        losses_for_logging.update(dice_loss_batches)


        dici_out = {
                "loss": total_loss,
            "losses_for_logging": losses_for_logging
            }
        if compute_grad == True:
            jac = softmax_derivative(input_activated)
            grad_sigma_z = jac
            grad_L_sigma = torch.autograd.grad(dice_loss_reduced, input_activated, retain_graph=True)[0]
            grad_L_sigma = grad_L_sigma.unsqueeze(2)
            grad_L_z = torch.einsum('ijk..., ijl...->ilk...', grad_L_sigma,grad_sigma_z)

            grad_sigma_z_ch23 = grad_sigma_z[:, 2:, 2:, :]
            grad_L_sigma_ch23 = grad_L_sigma[:, 2:, 0:1, :]
            grad_L_z_ch23=torch.einsum('ijk..., ijl...->ilk...', grad_L_sigma_ch23,grad_sigma_z_ch23)
            # grad_L_sigma_ch23.shape
            # grad_sigma_z_ch23.shape
            # grad_L_z_ch23.shape
            # grad_L_z_normed = torch.linalg.norm(grad_L_z_flat, dim=1)
            # self.grad_prod = self.grad_L_x * self.model.grad_sigma_z[0]

            dici_grad = {
                "grad_L_z": grad_L_z,
                "grad_L_z_ch23": grad_L_z_ch23,
            }
            dici_out.update(dici_grad)
        return dici_out


    def set_labels_channels(self, dice_loss_unreduced):
        num_channels = dice_loss_unreduced.shape[1]
        channel_losses_labels = ["loss_dice_label{}".format(x) for x in range(num_channels)]
        channel_losses = {x:y for x,y in zip(channel_losses_labels, dice_loss_unreduced.mean(0))}
        return channel_losses

    def set_labels_batches(self, dice_loss_unreduced):
        bs,  num_channels = dice_loss_unreduced.shape[0],dice_loss_unreduced.shape[1]
        label_maker = lambda x: "loss_dice_batch{0}_label{1}".format(*x)

        separate_case_losses_labels =[]
        for i in range(bs):
            for j in range(num_channels):
                label = label_maker([i,j])
                separate_case_losses_labels.append(label)

        batch_losses = {x:y for x,y in zip(separate_case_losses_labels, dice_loss_unreduced.view(-1))}
        return batch_losses

class DeepSupervisionLossCraig(pl.LightningModule):
    def __init__(
        self, levels: int, deep_supervision_scales, fg_classes: int,softmax=True, compute_grad=False
    ):
        super().__init__()
        store_attr()
        self.LossFunc = _DiceCELossMultiOutput(include_background=False, softmax=softmax)
        self.compute_grad = [False] * levels
        if compute_grad == True:
            self.compute_grad[0] = True  # at the last level compute jacobian of softmax

    def create_labels(self, bs, fg_classes):
        label_maker = lambda x: "loss_dice_batch{0}_label{1}".format(*x)
        batches = list(range(bs))
        classes = range_inclusive(1, fg_classes)

        self.batch_dice_labels= [
            "loss_dice_label{}".format(x) for x in classes
        ]

        self.case_recorder_labels = il.product(batches, classes)
        self.case_recorder_labels = list(map(label_maker, self.case_recorder_labels))


    def create_weights(self, device):
        weights = torch.tensor(
            [1 / (2**i) for i in range(self.levels)], requires_grad=False, device=device
        )

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        mask = torch.tensor(
            [True]
            + [True if i < self.levels - 1 else False for i in range(1, self.levels)]
        )
        weights[~mask] = 0
        self.weights = weights / weights.sum()

    def maybe_create_labels(self, target):
        if not hasattr(self, "labels"):
            if isinstance(target, list):
                bs = target[0].shape[0]
            else:
                bs = target.shape[0]
            self.create_labels(bs, self.fg_classes)

    def maybe_create_weights(self, preds):
        if not hasattr(self, "weights"):
            self.create_weights(preds[0].device)

    def apply_ds_scales(self, tnsr_inp: Union[list, torch.Tensor], mode):
        tnsr_listed = []
        for ind, sc in enumerate(self.deep_supervision_scales):
            if isinstance(tnsr_inp, Union[tuple, list]):
                tnsr = tnsr_inp[ind]
            else:
                tnsr = tnsr_inp
            if all([i == 1 for i in sc]):
                tnsr_listed.append(tnsr)
            else:
                t_shape = list(tnsr.shape[2:])
                size = [int(np.round(ss * aa)) for ss, aa in zip(sc, t_shape)]
                tnsr_downsampled = F.interpolate(tnsr, size=size, mode=mode)
                tnsr_listed.append(tnsr_downsampled)
        return tnsr_listed

    def forward(self, preds, target,compute_grad=False):

        self.maybe_create_weights(preds)
        self.maybe_create_labels(target)

        if isinstance(preds, Union[list, tuple]) and isinstance(
            target, torch.Tensor
        ):  # multires lists in training
            target = self.apply_ds_scales(target, "nearest")
            losses = [
                self.LossFunc(xx, yy, zz)
                for xx, yy, zz in zip(preds, target, self.compute_grad)
            ]

        elif isinstance(preds, torch.Tensor):  # tensor
            if (
                preds.dim() == target.dim() + 1
            ):  # i.e., training phase has deep supervision:
                preds = torch.unbind(preds, dim=1)
                preds = self.apply_ds_scales(preds, "trilinear")
                target = self.apply_ds_scales(target, "nearest")
                losses = [self.LossFunc(xx, yy,compute_grad=compute_grad) for xx, yy in zip(preds, target)]
            else:  # validation loss
                losses = [self.LossFunc(preds, target,compute_grad=compute_grad)]
        else:
            raise NotImplementedError("preds must be either list or stacked tensor")

        # self.set_loss_dict(losses[0])
        self.loss_dict = losses[0]['losses_for_logging']
        self.grad_L_z = losses[0]['grad_L_z']
        self.grad_L_z_ch23 = losses[0]['grad_L_z_ch23']
        losses_weighted = torch.stack(
            [self.weights * loss['loss'] for loss in losses]
        )  # total_loss * weight for each level
        losses_out= losses_weighted.sum()
        return losses_out


    def set_loss_dict(self,losses):
        class_losses = losses[-1].mean(0)
        separate_case_losses = list(losses[-1].view(-1))
        self.loss_dict = {x: y.item() for x, y in zip(self.labels, il.chain(losses[:3],class_losses,separate_case_losses))}
        if 'grad_L_z' in losses[0].keys():
            self.loss_dict['grad_L_z'] = losses[0]['grad_L_z']
            # losses_out['grad_L_z'] = losses[0]['grad_L_z']

class CombinedLoss(_DiceCELossMultiOutput):
    def __init__(self, fg_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        losses = super().forward(input, target)
        if not hasattr(self, "labels"):
            bs = target.shape[0]
            self.create_labels(bs, self.fg_classes)
        return losses[0]

    def set_loss_dict(self, loss):
        loss[1:] = [
            ll.detach() for ll in loss[1:]
        ]  # only l[0] needs gradient. rest are for logging
        class_losses = loss[-1].mean(0)
        separate_case_losses = list(loss[-1].view(-1))
        self.loss_dict = {
            x: y.item()
            for x, y in zip(
                self.labels, il.chain(loss[:3], class_losses, separate_case_losses)
            )
        }


# %%
if __name__ == "__main__":

    pred = torch.rand([1, 4, 64, 64, 48])
    targ = torch.rand([1, 1, 64, 64, 48])
    # pred = torch.load("fran/tmp/pred.pt")
    # %%
    x = torch.randn(
        2, 4, 4, 4, 4, requires_grad=True
    )  # Random input with requires_grad for autograd

    # Compute softmax manually
    softmax_output = softmax_wrapper(x)

    x.shape
    softmax_output.shape
    # Manually compute the Jacobian using the softmax_derivative function
    B, C, W, H, D = softmax_output.shape

    jacobian = torch.zeros(B, C, C, W * H * D, device=softmax_output.device)
    jacobian.shape
    i = 0
    softmax_output[:, i, :].shape
    # %%
    # Compute the diagonal and off-diagonal elements of the Jacobian
    for i in range(C):
        for j in range(C):
            L = softmax_output[:, i, :]
            R = softmax_output[:, j, :]
            if i == j:
                jacobian[:, i, j, :] = L * (1 - R)  # Diagonal term
            else:
                jacobian[:, i, j, :] = -L * R  # Off-diagonal term

    jacobian = jacobian.view(B, C, C, W, H, D)

    # %%
    # Reshape back to the original spatial dimensions [B, C, C, W, H, D]

    # Initialize a tensor to store the autograd-computed Jacobian
    B, C, W, H, D = softmax_output.shape

    # %%
    softmax_output = softmax_output.view(B, C, -1)  # Shape [B, C, W*H*D]

    # Create an empty tensor for the Jacobian of shape [B, C, C, W*H*D]
    jacobian = torch.zeros(B, C, C, W * H * D, device=softmax_output.device)
    # %%
    # Compute the Jacobian using autograd by backpropagating through each output channel
    for i in range(C):
        grad_outputs = torch.zeros_like(softmax_output)
        grad_outputs[:, i, :, :, :, :] = (
            1.0  # Set the gradient of the i-th output channel to 1
        )
        grad = torch.autograd.grad(
            softmax_output,
            x,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True,
        )[0]
        autograd_jacobian[:, i, :, :, :, :] = grad

    # Check if the manually computed Jacobian and the autograd Jacobian are close
    if torch.allclose(manual_jacobian, autograd_jacobian, atol=1e-5):
        print("The manually computed Jacobian and autograd Jacobian match!")
    else:
        print(
            "There is a difference between the manually computed Jacobian and autograd Jacobian."
        )

    # Optionally, print the difference between them
    diff = torch.abs(manual_jacobian - autograd_jacobian)
    print("Max difference:", diff.max())
    # %%
    # Define the tensors A and B for the short example
    A = torch.tensor(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],  # Tensor A of shape [2, 2, 2]
        dtype=torch.float32,
    )

    B_tensor = torch.tensor(
        [
            [[[1, 0], [0, 1]], [[2, 3], [4, 5]]],  # Tensor B of shape [2, 2, 2, 2]
            [[[6, 7], [8, 9]], [[0, 1], [2, 3]]],
        ],
        dtype=torch.float32,
    )

    # Reshape A to allow for broadcasting with B
    A_expanded = A.unsqueeze(2)  # Shape becomes [2, 2, 1, 2]

    # Perform element-wise multiplication and sum over the first channel (C) dimension of A
    result = (A_expanded * B_tensor).sum(dim=1)  # Summing over the first dimension (C)

    print(result)


# %%
