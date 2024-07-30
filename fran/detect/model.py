# %%


import tarfile
from torchmetrics.detection import IntersectionOverUnion 
from IPython.display import display




from fastcore.net import urllib
import matplotlib.patches as patches
from PIL import ImageDraw
from lightning.pytorch.callbacks import LearningRateMonitor, TQDMProgressBar
from fran.detect.data import *
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.trainer import Trainer
import torch
import lightning as L
from torch.optim.lr_scheduler import OneCycleLR

from fran.detect.data import DetectDataModule
from fran.detect.loss import YOLOLoss, filter_boxes, iou, load_bboxes_batch, load_image_batch, nms, show_images_with_boxes

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
    torch.set_warn_always(False)
    dm = DetectDataModule(data_dir = "/s/xnat_shadow/lidc2d/")
    dm.prepare_data()
    dm.setup(stage="fit")
    dl = dm.train_dataloader()
    model = TinyYOLOv2(bs=32,lr=1e-2)

    model.to('cuda')
    devices = [0]

# %%
    trainer = Trainer(
        max_epochs=300,
        accelerator="gpu",
        devices=devices,
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
        ],
    )
# %%
    trainer.fit(model,dm)
# %%
# %%
    import glob
    batch_size = 2
    all_idxs = np.array([int(fn.split("2008_")[-1].split(".jpg")[0]) for fn in sorted(glob.glob("VOCdevkit/VOC2012/JPEGImages/2008_*"))], dtype=int)
    np.random.RandomState(seed=42).shuffle(all_idxs)
    valid_idxs = all_idxs[-4*batch_size:]
    train_idxs = all_idxs[:-4*batch_size]
    from tqdm import tqdm

# %%
    device = 'cuda'
    lossfunc = YOLOLoss(anchors = model.anchors)
    model.to(device)
    for e in range(20):
        np.random.shuffle(train_idxs)
        range_ = tqdm(np.array_split(train_idxs, batch_size))
        with torch.no_grad():
            valid_imgs = load_image_batch(valid_idxs, size=320).to(device)
            valid_labels = load_bboxes_batch(valid_idxs, size=320, num_bboxes=2)
            valid_predictions = model(valid_imgs, yolo=False)
            vp2 = model(valid_imgs, yolo=True)
            valid_loss = lossfunc(valid_predictions, valid_labels).item()
            range_.set_postfix(valid_loss=valid_loss)

# %%
    valid_labels = load_bboxes_batch(valid_idxs, size=320, num_bboxes=10)

# %%
    for i, idxs in enumerate(range_):
        optimizer.zero_grad()
        batch_imgs = load_image_batch(idxs, size=320).to(device)
        batch_labels = load_bboxes_batch(idxs, size=320, num_bboxes=10)
        batch_predictions = model(batch_imgs, yolo=False)
    # batch_imgs = load_image_batch(idxs, size=320).to(device)
    batch_labels = load_bboxes_batch(idxs, size=320, num_bboxes=10)

# %%
    device ='cuda'
    model.to(device)
    iteri = iter(dl)
    aa= next(iteri)
    img = aa['image'].to(device)
    target = aa['bbox_yolo']
    classes_probs = torch.zeros(32,2)
    target2 = torch.cat([target,classes_probs],1)
    target2= target2.unsqueeze(1) # adds number of bboxes which is onlyu1

# %%

    with torch.no_grad():
        preds2 = model(img,yolo=True)
        preds = model(img,yolo=False)

    filtered_preds = filter_boxes(preds2,.7)
    nms_preds = nms(filtered_preds,.5)
    [a.shape for a in filtered_preds]
    [b.shape for b in nms_preds]


    nms_preds[0]
# %%

    model.lossfunc(preds,target2)
    lossfunc(preds,target)
# %%
# %%
    plt.show()
    show_images_with_boxes(img,preds)
# %%
    batch_imgs = load_image_batch([8,16,33,60], size=320)
    batch_labels = load_bboxes_batch([8,16,33,60], size=330, num_bboxes=10)
    batch_predictions = model(batch_imgs)
# %%
    predsa = [
       {
           "boxes": torch.tensor([
                [296.55, 93.96, 314.97, 152.79],
                [298.55, 98.96, 314.97, 151.79]]),
           "labels": torch.tensor([4, 5]),
       }
    ]

# %%
    labels = torch.zeros(32,1)
    box = aa['bbox_yolo']
    box2 = aa['bbox_yolo'].clone()

    preds = [{'boxes':box, 'labels':labels}]
    target = [{'boxes': box,'labels':labels}]

    la = I (preds,target)

    l= I.formard(box,box2)
# %%

    preds = [
       {
           "boxes": torch.tensor([
                [296.55, 93.96, 314.97, 152.79],
                [298.55, 98.96, 314.97, 151.79]]),
           "labels": torch.tensor([4, 5]),
       }
    ]
    target = [
       {
           "boxes": torch.tensor([[300.00, 100.00, 315.00, 150.00]]),
           "labels": torch.tensor([5]),
       }
    ]
    metric = IntersectionOverUnion()
    preds[0]['boxes']
    metric(preds, target)
# %%
    CLASSES = ['image']
    import torchvision
    to_img = torchvision.transforms.ToPILImage()
    input_tensor = img.clone()
    output_tensor = preds.clone()
    img = input_tensor[0]
    predictions = output_tensor[0]

# for img, predictions in zip(input_tensor, output_tensor):
    img_pil = to_img(img)
    img = torch.permute(img,[1,2,0])
    confidences = predictions[..., 4].flatten()
    boxes = (
        predictions[..., :4].contiguous().view(-1, 4)
    )  # only take first four features: x0, y0, w, h
    classes = predictions[..., 5:].contiguous().view(boxes.shape[0], -1)
    boxes[:, ::2] *= img_pil.width
    boxes[:, 1::2] *= img_pil.height
    boxes = (torch.stack([
                boxes[:, 0] - boxes[:, 2] / 2,
                boxes[:, 1] - boxes[:, 3] / 2,
                boxes[:, 0] + boxes[:, 2] / 2,
                boxes[:, 1] + boxes[:, 3] / 2,
    ], -1, ).cpu().to(torch.int32).numpy())


# %%
    x = valid_predictions.clone()
    y = valid_labels.clone()
# %%
    x = preds.clone()
    y = target2.clone()
# %%

    print(y.shape)
    print(x.shape)
    nT = y.shape[1]
    nA = lossfunc.anchors.shape[0]
    nB, _, nH, nW,_ = x.shape
    nCells = nH * nW
    nAnchors = nA * nCells
    y = y.to(dtype=x.dtype, device=x.device)
    # x = x.view(nB, nA, -1, nH, nW).permute(0, 1, 3, 4, 2)
    nC = x.shape[-1] - 5
    lossfunc.seen += nB

    anchors = lossfunc.anchors.to(dtype=x.dtype, device=x.device)
    coord_mask = torch.zeros(nB, nA, nH, nW, 1, requires_grad=False, dtype=x.dtype, device=x.device)
    conf_mask = torch.ones(nB, nA, nH, nW, requires_grad=False, dtype=x.dtype, device=x.device) * lossfunc.lambda_noobj
    cls_mask = torch.zeros(nB, nA, nH, nW, requires_grad=False, dtype=torch.bool, device=x.device)
    tcoord = torch.zeros(nB, nA, nH, nW, 4, requires_grad=False, dtype=x.dtype, device=x.device)
    tconf = torch.zeros(nB, nA, nH, nW, requires_grad=False, dtype=x.dtype, device=x.device)
    tcls = torch.zeros(nB, nA, nH, nW, requires_grad=False, dtype=x.dtype, device=x.device)

# %%
    coord = torch.cat([
        x[:, :, :, :, 0:1].sigmoid(),  # X center
        x[:, :, :, :, 1:2].sigmoid(),  # Y center
        x[:, :, :, :, 2:3],  # Width
        x[:, :, :, :, 3:4],  # Height
    ], -1)

# %%
    range_y, range_x = torch.meshgrid(
        torch.arange(nH, dtype=x.dtype, device=x.device),
        torch.arange(nW, dtype=x.dtype, device=x.device),
    )
    anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

    x = torch.cat([
        (x[:, :, :, :, 0:1].sigmoid() + range_x[None,None,:,:,None]),  # X center
        (x[:, :, :, :, 1:2].sigmoid() + range_y[None,None,:,:,None]),  # Y center
        (x[:, :, :, :, 2:3].exp() * anchor_x[None,:,None,None,None]),  # Width
        (x[:, :, :, :, 3:4].exp() * anchor_y[None,:,None,None,None]),  # Height
        x[:, :, :, :, 4:5].sigmoid(), # confidence
        x[:, :, :, :, 5:], # classes (NOTE: no softmax here bc CEL is used later, which works on logits)
    ], -1)

    conf = x[..., 4]
    cls = x[..., 5:].reshape(-1, nC)
    x = x[..., :4].detach() # gradients are tracked in coord -> not here anymore.

    if lossfunc.seen < lossfunc.coord_prefill:
        coord_mask.fill_(np.sqrt(.01 / lossfunc.lambda_coord))
        tcoord[..., 0].fill_(0.5)
        tcoord[..., 1].fill_(0.5)

    for b in range(nB):
        gt = y[b][(y[b, :, -1] >= 0)[:, None].expand_as(y[b])].view(-1, 6)[:,:4]
        gt[:, ::2] *= nW
        gt[:, 1::2] *= nH
        if gt.numel() == 0:  # no ground truth for this image
            continue

        # Set confidence mask of matching detections to 0
        iou_gt_pred = iou(gt, x[b:(b+1)].view(-1, 4))
        mask = (iou_gt_pred > lossfunc.threshold).sum(0) >= 1
        conf_mask[b][mask.view_as(conf_mask[b])] = 0

        # Find best anchor for each gt
        iou_gt_anchors = iou_wh(gt[:,2:], anchors)
        _, best_anchors = iou_gt_anchors.max(1)

        # Set masks and target values for each gt
        nGT = gt.shape[0]
        gi = gt[:, 0].clamp(0, nW-1).long()
        gj = gt[:, 1].clamp(0, nH-1).long()

        conf_mask[b, best_anchors, gj, gi] = lossfunc.lambda_obj
        tconf[b, best_anchors, gj, gi] = iou_gt_pred.view(nGT, nA, nH, nW)[torch.arange(nGT), best_anchors, gj, gi]
        coord_mask[b, best_anchors, gj, gi, :] = (2 - (gt[:, 2] * gt[:, 3]) / nCells)[..., None]
        tcoord[b, best_anchors, gj, gi, 0] = gt[:, 0] - gi.float()
        tcoord[b, best_anchors, gj, gi, 1] = gt[:, 1] - gj.float()
        tcoord[b, best_anchors, gj, gi, 2] = (gt[:, 2] / anchors[best_anchors, 0]).log()
        tcoord[b, best_anchors, gj, gi, 3] = (gt[:, 3] / anchors[best_anchors, 1]).log()
        cls_mask[b, best_anchors, gj, gi] = 1
        tcls[b, best_anchors, gj, gi] = y[b, torch.arange(nGT), -1]

    coord_mask = coord_mask.sqrt()
    conf_mask = conf_mask.sqrt()
    tcls = tcls[cls_mask].view(-1).long()
    cls_mask = cls_mask.view(-1, 1).expand(nB*nA*nH*nW, nC)
    cls = cls[cls_mask].view(-1, nC)

    loss_coord = lossfunc.lambda_coord * lossfunc.mse(coord*coord_mask, tcoord*coord_mask) / (2 * nB)
    loss_conf = lossfunc.mse(conf*conf_mask, tconf*conf_mask) / (2 * nB)
    loss_cls = lossfunc.lambda_cls * lossfunc.cel(cls, tcls) / nB

# %%
