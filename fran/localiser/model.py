# %%
#link :https://blog.flaport.net/yolo-part-1.html
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from fran.localiser.data import *
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.trainer import Trainer
import torch
import lightning as L
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from fran.localiser.data import DetectDataModule
from fran.localiser.loss import YOLOLoss
from fran.localiser.helpers import *


class TinyYOLOv2a(torch.nn.Module):
    def __init__(
        self,
        num_classes=20,
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

        # Layers
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
        self.conv9 = torch.nn.Conv2d(1024, len(anchors) * (5 + num_classes), 1, 1, 0)

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
        x = torch.cat([
            (x[:, :, :, :, 0:1].sigmoid() + range_x[None,None,:,:,None]) / nW,  # X center
            (x[:, :, :, :, 1:2].sigmoid() + range_y[None,None,:,:,None]) / nH,  # Y center
            (x[:, :, :, :, 2:3].exp() * anchor_x[None,:,None,None,None]) / nW,  # Width
            (x[:, :, :, :, 3:4].exp() * anchor_y[None,:,None,None,None]) / nH,  # Height
            x[:, :, :, :, 4:5].sigmoid(), # confidence
            x[:, :, :, :, 5:].softmax(-1), # classes
        ], -1)

        return x # (batch_size, # anchors, height, width, 5+num_classes)

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

    def forward(self, x, yolo=False):
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

    import glob
    from tqdm import tqdm
    device = 'cuda'
# %%
    network = TinyYOLOv2(bs=32,lr=1e-2,num_classes=20)
    import glob
    batch_size = 256
    all_idxs = np.array([int(fn.split("2008_")[-1].split(".jpg")[0]) for fn in sorted(glob.glob("/s/datasets_bkp/VOCdevkit/VOC2012/JPEGImages/2008_*"))], dtype=int)
    lossfunc = YOLOLoss(anchors=network.anchors, coord_prefill=int(5*all_idxs.shape[0]))
    optimizer = torch.optim.Adam(network.conv9.parameters(), lr=0.003)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.003)
    np.random.RandomState(seed=42).shuffle(all_idxs)
    valid_idxs = all_idxs[-4*batch_size:]
    train_idxs = all_idxs[:-4*batch_size]


# %%
    network.to(device)

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

# %%
    input_tensor = load_image_batch([33], 320)  # batch with single image of an airplane
    output_tensor = network(input_tensor.cuda())
    show_images_with_boxes(input_tensor, output_tensor)
    show_image_with_boxes(input_tensor, output_tensor)
# %%
#SECTION:-------------------- WEIGHTS--------------------------------------------------------------------------------------

    network2 = TinyYOLOv2a()
    # model = TinyYOLOv2.load_from_checkpoint("/home/ub/code/fran/fran/logs/lightning_logs/version_2/checkpoints/last.ckpt")
    input_tensor = load_image_batch([33], 320)  # batch with single image of an airplane
    network2.to(device)
    load_weights(network2)
# %%
    output_tensor2 = network2(input_tensor.cuda())
    show_image_with_boxes(input_tensor, output_tensor2)
# %%
    with torch.no_grad():
        preds = network(input_tensor.cuda(),False)

# %%

    show_images_with_boxes(input_tensor,output_tensor2)
    filtered_tensor = filter_boxes(output_tensor2,.2)
    show_images_with_boxes(input_tensor,filtered_tensor)

# %%
#SECTION:-------------------- RETRAIN--------------------------------------------------------------------------------------
    for p in network2.conv9.parameters():
        try:
            torch.nn.init.kaiming_normal_(p)
        except ValueError:
            torch.nn.init.normal_(p)
    batch_predictions = network2(input_tensor.cuda())

    show_images_with_boxes(input_tensor,batch_predictions)
# %%
    batch_size = 256
    all_idxs = np.array([int(fn.split("2008_")[-1].split(".jpg")[0]) for fn in sorted(glob.glob("/s/datasets_bkp/VOCdevkit/VOC2012/JPEGImages/2008_*"))], dtype=int)
    lossfunc = YOLOLoss(anchors=network2.anchors, coord_prefill=int(5*all_idxs.shape[0]))
    optimizer = torch.optim.Adam(network2.conv9.parameters(), lr=0.003)
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
            valid_predictions = network2(valid_imgs, yolo=False)
            valid_loss = lossfunc(valid_predictions, valid_labels).item()
            range_.set_postfix(valid_loss=valid_loss)
        for i, idxs in enumerate(range_):
            optimizer.zero_grad()
            batch_imgs = load_image_batch(idxs, size=320).to(device)
            batch_labels = load_bboxes_batch(idxs, size=320, num_bboxes=10)
            batch_predictions = network2(batch_imgs, yolo=False)
            loss = lossfunc(batch_predictions, batch_labels)
            range_.set_postfix(loss=loss.item(), valid_loss=valid_loss)
            loss.backward()
            optimizer.step()

# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------

    from fran.utils.common import *
    torch.set_warn_always(False)
    dm = DetectDataModule(data_dir = "/s/xnat_shadow/lidc2d/",batch_size=32)
    dm.prepare_data()
    dm.setup(stage="fit")
    dl = dm.train_dataloader()
    network = TinyYOLOv2(bs=64,lr=1e-2)
    network = TinyYOLOv2.load_from_checkpoint("/s/fran_storage/checkpoints/detection/last-v2.ckpt")


    network.to('cuda')
    devices = [0]

# %%
    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=devices,
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
            ModelCheckpoint(
                dirpath="/s/fran_storage/checkpoints/detection",
                save_last=True,
                monitor="val_loss",
                every_n_epochs=10,
                # mode="min",
                filename="{epoch}-{val_loss:.2f}",
                enable_version_counter=True,
                auto_insert_metric_name=True,
            ),
        ],
    )
# %%
    trainer.fit(network,dm)


# %%
#SECTION:-------------------- TROUBLE--------------------------------------------------------------------------------------

    network.to(device)
    dm = DetectDataModule(data_dir = "/s/xnat_shadow/lidc2d/",batch_size = 1)
    dm.prepare_data()
    dm.setup(stage="fit")

    vdl = dm.val_dataloader()
    iteri = iter(vdl)
    bb = next(iteri)
    img = bb['image']
# %%
    with torch.no_grad():
        preds = network(img.cuda(),True)

# %%

    show_images_with_boxes(img,preds)
    filtered_tensor = filter_boxes(preds,.2)
    show_images_with_boxes(img,filtered_tensor)

    nms_tensor = nms(filtered_tensor, 0.5)
    show_images_with_boxes(img, nms_tensor)
# %%
    ind = 0
    im =img[ind][0].detach().cpu()
    bb = preds[ind]
    bb.shape
# %%
    for e in range(200):
        np.random.shuffle(train_idxs)
        range_ = tqdm(np.array_split(train_idxs, batch_size))
        with torch.no_grad():
            valid_imgs = load_image_batch(valid_idxs, size=320).to(device)
            valid_labels = load_bboxes_batch(valid_idxs, size=320, num_bboxes=2)
            valid_predictions = network(valid_imgs, yolo=False)
            vp2 = network(valid_imgs, yolo=True)
            valid_loss = lossfunc(valid_predictions, valid_labels).item()
            range_.set_postfix(valid_loss=valid_loss)

# %%
    import matplotlib.pyplot as plt
    plt.ion()
    # matplotlib.use('Agg')
    import matplotlib.image as mpimg
    pil_img = mpimg.imread('/s/datasets_bkp/VOCdevkit/VOC2007/SegmentationObject/000039.png')
    plt.imshow(pil_img)
    plt.show()
# %%
# %%


    valid_labels = load_bboxes_batch(valid_idxs, size=320, num_bboxes=10)
    input_tensor = load_image_batch([33], 320)
    show_images(input_tensor)

    show_images(input_tensor)


    input_tensor = load_image_batch([8, 16, 33, 60], size=320)
    input_tensor = load_image_batch([33], size=320)
    output_tensor= network(input_tensor.cuda())
# %%
    show_images_with_boxes(input_tensor,output_tensor)
    filtered_tensor = filter_boxes(output_tensor,.2)
    show_images_with_boxes(input_tensor,filtered_tensor)

# %%
    nms_tensor = nms(filtered_tensor, 0.5)
    show_images_with_boxes(input_tensor, nms_tensor)

# %%

