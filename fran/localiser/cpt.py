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
#link :https://blog.flaport.net/yolo-part-1.html
from fran.localizer.data import *
import torch
import lightning as L
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from fran.localizer.loss import YOLOLoss
from fran.localizer.helpers import *



class TinyYOLOv2(L.LightningModule):
    def __init__(
        self,
        bs,
        lr,
        num_classes=1,
        anchors=(
            (1.08, 1.19),
            (3.42, 4.41),
            (6.63, 11.38),
            (9.42, 5.11),
            (16.62, 10.52),
        ),
    ):
        super().__init__()

        # Parameters
        self.register_buffer("anchors", torch.tensor(anchors))
        self.num_classes = num_classes

        self.bs = bs

        self.save_hyperparameters()
        self.create_model(anchors)
        self.lossfunc = YOLOLoss(anchors = anchors)
        # Layers

    def create_model(self,anchors):
        self.relu = torch.nn.LeakyReLU(0.1, inplace=True)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.slowpool = torch.nn.MaxPool2d(2, 1)
        self.pad = torch.nn.ReflectionPad2d((0, 1, 0, 1))
        self.norm1 = torch.nn.BatchNorm2d(16, momentum=0.1)
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.norm2 = torch.nn.BatchNorm2d(32, momentum=0.1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1, 1, bias=False)
        self.norm3 = torch.nn.BatchNorm2d(64, momentum=0.1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.norm4 = torch.nn.BatchNorm2d(128, momentum=0.1)
        self.conv4 = torch.nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.norm5 = torch.nn.BatchNorm2d(256, momentum=0.1)
        self.conv5 = torch.nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.norm6 = torch.nn.BatchNorm2d(512, momentum=0.1)
        self.conv6 = torch.nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.norm7 = torch.nn.BatchNorm2d(1024, momentum=0.1)
        self.conv7 = torch.nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.norm8 = torch.nn.BatchNorm2d(1024, momentum=0.1)
        self.conv8 = torch.nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)
        self.conv9 = torch.nn.Conv2d(1024, len(anchors) * (5 + self.num_classes), 1, 1, 0)

    def forward(self, x, yolo=True):
        x = self.relu(self.pool(self.norm1(self.conv1(x))))
        x = self.relu(self.pool(self.norm2(self.conv2(x))))
        x = self.relu(self.pool(self.norm3(self.conv3(x))))
        x = self.relu(self.pool(self.norm4(self.conv4(x))))
        x = self.relu(self.pool(self.norm5(self.conv5(x))))
        x = self.relu(self.slowpool(self.pad(self.norm6(self.conv6(x)))))
        x = self.relu(self.norm7(self.conv7(x)))
        x = self.relu(self.norm8(self.conv8(x)))
        x = self.conv9(x)
        if yolo:
            x = self.yolo(x)
        return x

    def yolo(self, x):
        # store the original shape of x
        nB, _, nH, nW = x.shape

        # reshape the x-tensor: (batch size, # anchors, height, width, 5+num_classes)
        x = x.view(nB, self.anchors.shape[0], -1, nH, nW).permute(0, 1, 3, 4, 2)

        # get normalized auxiliary tensors
        anchors = self.anchors.to(dtype=x.dtype, device=x.device)
        range_y, range_x = torch.meshgrid(
            torch.arange(nH, dtype=x.dtype, device=x.device),
            torch.arange(nW, dtype=x.dtype, device=x.device),
        )
        anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

        # compute boxes.
        x = torch.cat(
            [
                (x[:, :, :, :, 0:1].sigmoid() + range_x[None, None, :, :, None])
                / nW,  # X center
                (x[:, :, :, :, 1:2].sigmoid() + range_y[None, None, :, :, None])
                / nH,  # Y center
                (x[:, :, :, :, 2:3].exp() * anchor_x[None, :, None, None, None])
                / nW,  # Width
                (x[:, :, :, :, 3:4].exp() * anchor_y[None, :, None, None, None])
                / nH,  # Height
                x[:, :, :, :, 4:5].sigmoid(),  # confidence
                x[:, :, :, :, 5:].softmax(-1),  # classes
            ],
            -1,
        )

        return x  # (batch_size, # anchors, height, width, 5+num_classes)

    def evaluate(self, batch, stage=None):
        image = batch["image"]
        batchsize = image.shape[0]
        bbox = batch["bbox_yolo"]

        classes_probs = torch.ones(batchsize,2,device = bbox.device)
        # cals = torch.zeros(self.bs, 1, 1, device=bbox.device)
        bbox_clas = torch.cat([bbox, classes_probs], 1)
        bbox_clas = bbox_clas.unsqueeze(1)

        pred = self.forward(image, yolo=False)
        loss = self.lossfunc(pred, bbox_clas)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)

    def training_step(self, batch, batch_idx):
        self.evaluate(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // self.bs
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


# %
if __name__ == "__main__":
# %%
#SECTION:-------------------- VOC--------------------------------------------------------------------------------------

    from tqdm import tqdm
    device = 'cuda'
    network = TinyYOLOv2(bs=32,lr=1e-2,num_classes=20)
    # model = TinyYOLOv2.load_from_checkpoint("/home/ub/code/fran/fran/logs/lightning_logs/version_2/checkpoints/last.ckpt")
    network.to(device)
# %%
    import glob
    batch_size = 192
    all_idxs = np.array([int(fn.split("2008_")[-1].split(".jpg")[0]) for fn in sorted(glob.glob("/s/datasets_bkp/VOCdevkit/VOC2012/JPEGImages/2008_*"))], dtype=int)
    lossfunc = YOLOLoss(anchors=network.anchors, coord_prefill=int(5*all_idxs.shape[0]))
    optimizer = torch.optim.Adam(network.conv9.parameters(), lr=0.003)
    np.random.RandomState(seed=42).shuffle(all_idxs)
    valid_idxs = all_idxs[-4*batch_size:]
    train_idxs = all_idxs[:-4*batch_size]


# %%

    for e in range(20):
        np.random.shuffle(train_idxs)
        range_ = tqdm(np.array_split(train_idxs, batch_size))
        with torch.no_grad():
            valid_imgs = load_image_batch(valid_idxs, size=320).to(device)
            valid_labels = load_bboxes_batch(valid_idxs, size=320, num_bboxes=10)
            valid_predictions = network(valid_imgs, yolo=False)
            valid_loss = lossfunc(valid_predictions, valid_labels).item()
            range_.set_postfix(valid_loss=valid_loss)
        for i, idxs in enumerate(range_):
            optimizer.zero_grad()
            batch_imgs = load_image_batch(idxs, size=320).to(device)
            batch_labels = load_bboxes_batch(idxs, size=320, num_bboxes=10)
            batch_predictions = network(batch_imgs, yolo=False)
            loss = lossfunc(batch_predictions, batch_labels)
            range_.set_postfix(loss=loss.item(), valid_loss=valid_loss)
            loss.backward()
            optimizer.step()

#
