# %%
import matplotlib.pyplot as plt

from fran.localizer.helpers import iou, iou_wh
plt.ion()
# matplotlib.use('Agg')
import numpy as np
import lightning as L
import torch

import ipdb

tr = ipdb.set_trace
# class YOLOLoss(torch.nn.modules.loss._Loss):
class YOLOLoss(L.LightningModule):
    """A loss function to train YOLO v2

    Args:
        anchors (optional, list): the list of anchors (should be the same anchors as the ones defined in the YOLO class)
        seen (optional, torch.Tensor): the number of images the network has already been trained on
        coord_prefill (optional, int): the number of images for which the predicted bboxes will be centered in the image
        threshold (optional, float): minimum iou necessary to have a predicted bbox match a target bbox
        lambda_coord (optional, float): hyperparameter controlling the importance of the bbox coordinate predictions
        lambda_noobj (optional, float): hyperparameter controlling the importance of the bboxes containing no objects
        lambda_obj (optional, float): hyperparameter controlling the importance of the bboxes containing objects
        lambda_cls (optional, float): hyperparameter controlling the importance of the class prediction if the bbox contains an object
    """

    def __init__(
        self,
        anchors=(
            (1.08, 1.19),
            (3.42, 4.41),
            (6.63, 11.38),
            (9.42, 5.11),
            (16.62, 10.52),
        ),
        seen=0,
        coord_prefill=12800,
        threshold=0.6,
        lambda_coord=1.0,
        lambda_noobj=1.0,
        lambda_obj=5.0,
        lambda_cls=1.0,
    ):
        super().__init__()

        if not torch.is_tensor(anchors):
            anchors = torch.tensor(anchors, dtype=torch.get_default_dtype())
        else:
            anchors = anchors.data.to(torch.get_default_dtype())
        self.register_buffer("anchors", anchors)

        self.seen = int(seen + 0.5)
        self.coord_prefill = int(coord_prefill + 0.5)

        self.threshold = float(threshold)
        self.lambda_coord = float(lambda_coord)
        self.lambda_noobj = float(lambda_noobj)
        self.lambda_obj = float(lambda_obj)
        self.lambda_cls = float(lambda_cls)

        self.mse = torch.nn.MSELoss(reduction="sum")
        self.cel = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, x, y):
        # x : pred
        # y : target
        nT = y.shape[1]
        nA = self.anchors.shape[0]
        nB, _, nH, nW = x.shape
        nPixels = nH * nW
        nAnchors = nA * nPixels
        y = y.to(dtype=x.dtype, device=x.device)
        x = x.view(nB, nA, -1, nH, nW).permute(0, 1, 3, 4, 2)
        nC = x.shape[-1] - 5
        self.seen += nB

        anchors = self.anchors.to(dtype=x.dtype, device=x.device)
        coord_mask = torch.zeros(
            nB, nA, nH, nW, 1, requires_grad=False, dtype=x.dtype, device=x.device
        )
        conf_mask = (
            torch.ones(
                nB, nA, nH, nW, requires_grad=False, dtype=x.dtype, device=x.device
            )
            * self.lambda_noobj
        )
        cls_mask = torch.zeros(
            nB, nA, nH, nW, requires_grad=False, dtype=torch.bool, device=x.device
        )
        tcoord = torch.zeros(
            nB, nA, nH, nW, 4, requires_grad=False, dtype=x.dtype, device=x.device
        )
        tconf = torch.zeros(
            nB, nA, nH, nW, requires_grad=False, dtype=x.dtype, device=x.device
        )
        tcls = torch.zeros(
            nB, nA, nH, nW, requires_grad=False, dtype=x.dtype, device=x.device
        )

        coord = torch.cat(
            [
                x[:, :, :, :, 0:1].sigmoid(),  # X center
                x[:, :, :, :, 1:2].sigmoid(),  # Y center
                x[:, :, :, :, 2:3],  # Width
                x[:, :, :, :, 3:4],  # Height
            ],
            -1,
        )

        range_y, range_x = torch.meshgrid(
            torch.arange(nH, dtype=x.dtype, device=x.device),
            torch.arange(nW, dtype=x.dtype, device=x.device),
        )
        anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

        x = torch.cat(
            [
                (
                    x[:, :, :, :, 0:1].sigmoid() + range_x[None, None, :, :, None]
                ),  # X center
                (
                    x[:, :, :, :, 1:2].sigmoid() + range_y[None, None, :, :, None]
                ),  # Y center
                (
                    x[:, :, :, :, 2:3].exp() * anchor_x[None, :, None, None, None]
                ),  # Width
                (
                    x[:, :, :, :, 3:4].exp() * anchor_y[None, :, None, None, None]
                ),  # Height
                x[:, :, :, :, 4:5].sigmoid(),  # confidence
                x[
                    :, :, :, :, 5:
                ],  # classes (NOTE: no softmax here bc CEL is used later, which works on logits)
            ],
            -1,
        )

        conf = x[..., 4]
        cls = x[..., 5:].reshape(-1, nC)
        x = x[..., :4].detach()  # gradients are tracked in coord -> not here anymore.

        if self.seen < self.coord_prefill:
            coord_mask.fill_(np.sqrt(0.01 / self.lambda_coord))
            tcoord[..., 0].fill_(0.5)
            tcoord[..., 1].fill_(0.5)

        for b in range(nB):
            gt = y[b][(y[b, :, -1] >= 0)[:, None].expand_as(y[b])].view(-1, 6)[:, :4]
            gt[:, ::2] *= nW
            gt[:, 1::2] *= nH
            if gt.numel() == 0:  # no ground truth for this image
                continue

            # Set confidence mask of matching detections to 0
            iou_gt_pred = iou(gt, x[b : (b + 1)].view(-1, 4))
            mask = (iou_gt_pred > self.threshold).sum(0) >= 1
            conf_mask[b][mask.view_as(conf_mask[b])] = 0

            # Find best anchor for each gt
            iou_gt_anchors = iou_wh(gt[:, 2:], anchors)
            _, best_anchors = iou_gt_anchors.max(1)

            # Set masks and target values for each gt
            nGT = gt.shape[0]
            gi = gt[:, 0].clamp(0, nW - 1).long()
            gj = gt[:, 1].clamp(0, nH - 1).long()

            conf_mask[b, best_anchors, gj, gi] = self.lambda_obj
            tconf[b, best_anchors, gj, gi] = iou_gt_pred.view(nGT, nA, nH, nW)[
                torch.arange(nGT), best_anchors, gj, gi
            ]
            coord_mask[b, best_anchors, gj, gi, :] = (
                2 - (gt[:, 2] * gt[:, 3]) / nPixels
            )[..., None]
            tcoord[b, best_anchors, gj, gi, 0] = gt[:, 0] - gi.float()
            tcoord[b, best_anchors, gj, gi, 1] = gt[:, 1] - gj.float()
            tcoord[b, best_anchors, gj, gi, 2] = (
                gt[:, 2] / anchors[best_anchors, 0]
            ).log()
            tcoord[b, best_anchors, gj, gi, 3] = (
                gt[:, 3] / anchors[best_anchors, 1]
            ).log()
            cls_mask[b, best_anchors, gj, gi] = 1
            tcls[b, best_anchors, gj, gi] = y[b, torch.arange(nGT), -1]

        coord_mask = coord_mask.sqrt()
        conf_mask = conf_mask.sqrt()
        tcls = tcls[cls_mask].view(-1).long()
        cls_mask = cls_mask.view(-1, 1).expand(nB * nA * nH * nW, nC)
        cls = cls[cls_mask].view(-1, nC)

        loss_coord = (
            self.lambda_coord
            * self.mse(coord * coord_mask, tcoord * coord_mask)
            / (2 * nB)
        )
        loss_conf = self.mse(conf * conf_mask, tconf * conf_mask) / (2 * nB)
        # loss_cls = self.lambda_cls * self.cel(cls, tcls) / nB
        loss_cls = 0
        return loss_coord + loss_conf + loss_cls


# Function to convert cells to bounding boxes
def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True):
    # Batch size used on predictions
    batch_size = predictions.shape[0]
    # Number of anchors
    num_anchors = len(anchors)
    # List of all the predictions
    box_predictions = predictions[..., 1:5]

    # If the input is predictions then we will pass the x and y coordinate
    # through sigmoid function and width and height to exponent function and
    # calculate the score and best class.
    if is_predictions:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)

    # Else we will just calculate scores and best class.
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    # Calculate cell indices
    cell_indices = (
        torch.arange(s)
        .repeat(predictions.shape[0], 3, s, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )

    # Calculate x, y, width and height with proper scaling
    x = 1 / s * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / s * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    width_height = 1 / s * box_predictions[..., 2:4]

    # Concatinating the values and reshaping them in
    # (BATCH_SIZE, num_anchors * S * S, 6) shape
    converted_bboxes = torch.cat(
        (best_class, scores, x, y, width_height), dim=-1
    ).reshape(batch_size, num_anchors * s * s, 6)

    # Returning the reshaped and converted bounding box list
    return converted_bboxes.tolist()


# %%
