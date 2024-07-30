# %%
from torch import nn
import numpy as np
import lightning as L
import torch
from PIL import ImageDraw
import torchvision
from IPython.display import display

import ipdb
tr = ipdb.set_trace

CLASSES = ["img"]

CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
CLASS2NUM = {class_:idx for idx, class_ in enumerate(CLASSES)}
from PIL import Image

def filter_boxes(output_tensor, threshold):
    b, a, h, w, c = output_tensor.shape
    x = output_tensor.contiguous().view(b, a * h * w, c)

    boxes = x[:, :, 0:4]
    confidence = x[:, :, 4]
    scores, idx = torch.max(x[:, :, 5:], -1)
    idx = idx.float()
    scores = scores * confidence
    mask = scores > threshold

    filtered = []
    for c, s, i, m in zip(boxes, scores, idx, mask):
        if m.any():
            detected = torch.cat([c[m, :], s[m, None], i[m, None]], -1)
        else:
            detected = torch.zeros((0, 6), dtype=x.dtype, device=x.device)
        filtered.append(detected)
    return filtered


def load_image(idx, size, device="cpu"):
    filename = f"VOCdevkit/VOC2012/JPEGImages/2008_{str(idx).zfill(6)}.jpg"
    img = Image.open(filename)
    transforms = [torchvision.transforms.ToTensor()]
    try:
        width, height = size
    except TypeError:
        width = height = size
    scale = min(width / img.width, height / img.height)
    new_width, new_height = int(img.width * scale), int(img.height * scale)
    diff_width, diff_height = width - new_width, height - new_height
    resize = torchvision.transforms.Resize(size=(new_height, new_width))
    pad = torchvision.transforms.Pad(
        padding=(
            diff_width // 2,
            diff_height // 2,
            diff_width // 2 + diff_width % 2,
            diff_height // 2 + diff_height % 2,
        )
    )
    transforms = [resize, pad] + transforms
    transformation = torchvision.transforms.Compose(transforms)
    x = transformation(img).to(device)
    return x
def load_image_batch(idxs, size, device="cpu"):
    imgs = [load_image(idx, size=size, device="cpu") for idx in idxs]
    x = torch.stack(imgs, 0)
    return x.to(device)
def load_bboxes(idx, size, num_bboxes=None, device="cpu"):
    boxfilename = f"VOCdevkit/VOC2012/Annotations/2008_{str(idx).zfill(6)}.xml"
    imgfilename = f"VOCdevkit/VOC2012/JPEGImages/2008_{str(idx).zfill(6)}.jpg"
    img = Image.open(imgfilename) # img won't be loaded into memory, just needed for metadata.
    try:
        width, height = size
    except TypeError:
        width = height = size
    scale = min(width / img.width, height / img.height)
    new_width, new_height = int(img.width * scale), int(img.height * scale)
    diff_width  = (width - new_width)/width*img.width
    diff_height = (height - new_height)/height*img.height
    bboxes = []
    with open(boxfilename, "r") as file:
        for line in file:
            if "<name>" in line:
                class_ = line.split("<name>")[-1].split("</name>")[0].strip()
            elif "<xmin>" in line:
                x0 = float(line.split("<xmin>")[-1].split("</xmin>")[0].strip())
            elif "<xmax>" in line:
                
                x1 = float(line.split("<xmax>")[-1].split("</xmax>")[0].strip())
            elif "<ymin>" in line:
                y0 = float(line.split("<ymin>")[-1].split("</ymin>")[0].strip())
            elif "<ymax>" in line:
                y1 = float(line.split("<ymax>")[-1].split("</ymax>")[0].strip())
            elif "</object>" in line:
                if class_ not in CLASS2NUM:
                    continue
                bbox = [
                    (diff_width/2 + (x0+x1)/2) / (img.width + diff_width), # center x
                    (diff_height/2 + (y0+y1)/2) / (img.height + diff_height), # center y
                    (max(x0,x1) - min(x0,x1)) / (img.width + diff_width), # width
                    (max(y0,y1) - min(y0,y1)) / (img.height + diff_height), # height
                    1.0, # confidence
                    CLASS2NUM[class_], # class idx
                ]
                bboxes.append((bbox[2]*bbox[3]*(img.width+diff_width)*(img.height+diff_height),bbox))

    bboxes = torch.tensor([bbox for _, bbox in sorted(bboxes)], dtype=torch.get_default_dtype(), device=device)
    if num_bboxes:
        if num_bboxes > len(bboxes):
            zeros = torch.zeros((num_bboxes - bboxes.shape[0], 6), dtype=torch.get_default_dtype(), device=device)
            zeros[:, -1] = -1
            bboxes = torch.cat([bboxes, zeros], 0)
        elif num_bboxes < len(bboxes):
            bboxes = bboxes[:num_bboxes]

    return bboxes

def load_bboxes_batch(idxs, size, num_bboxes, device="cpu"):
    bboxes = [load_bboxes(idx, size, num_bboxes=num_bboxes, device="cpu") for idx in idxs]
    bboxes = torch.stack(bboxes, 0)
    return bboxes
def show_images_with_boxes(input_tensor, output_tensor):
    to_img = torchvision.transforms.ToPILImage()
    for img, predictions in zip(input_tensor, output_tensor):
        img = to_img(img)
        if 0 in predictions.shape: # empty tensor
            display(img)
            continue
        confidences = predictions[..., 4].flatten()
        boxes = (
            predictions[..., :4].contiguous().view(-1, 4)
        )  # only take first four features: x0, y0, w, h
        classes = predictions[..., 5:].contiguous().view(boxes.shape[0], -1)
        boxes[:, ::2] *= img.width
        boxes[:, 1::2] *= img.height
        boxes = (torch.stack([
                    boxes[:, 0] - boxes[:, 2] / 2,
                    boxes[:, 1] - boxes[:, 3] / 2,
                    boxes[:, 0] + boxes[:, 2] / 2,
                    boxes[:, 1] + boxes[:, 3] / 2,
        ], -1, ).cpu().to(torch.int32).numpy())
        for box, confidence, class_ in zip(boxes, confidences, classes):
            if confidence < 0.01:
                continue # don't show boxes with very low confidence
            # make sure the box fits within the picture:
            box = [
                max(0, int(box[0])),
                max(0, int(box[1])),
                min(img.width - 1, int(box[2])),
                min(img.height - 1, int(box[3])),
            ]
            try:  # either the class is given as the sixth feature
                idx = int(class_.item())
            except ValueError:  # or the 20 softmax probabilities are given as features 6-25
                idx = int(torch.max(class_, 0)[1].item())
            try:
                class_ = CLASSES[idx]  # the first index of torch.max is the argmax.
            except IndexError: # if the class index does not exist, don't draw anything:
                continue

            
            color = (  # green color when confident, red color when not confident.
                int((1 - (confidence.item())**0.8 ) * 255),
                int((confidence.item())**0.8 * 255),
                0,
            )
            draw = ImageDraw.Draw(img)
            draw.rectangle(box, outline=color)
            draw.text(box[:2], class_, fill=color)
            
        display(img)

def iou(bboxes1, bboxes2):
    """ calculate iou between each bbox in `bboxes1` with each bbox in `bboxes2`"""
    px, py, pw, ph = bboxes1[...,:4].reshape(-1, 4).split(1, -1)
    lx, ly, lw, lh = bboxes2[...,:4].reshape(-1, 4).split(1, -1)
    px1, py1, px2, py2 = px - 0.5 * pw, py - 0.5 * ph, px + 0.5 * pw, py + 0.5 * ph
    lx1, ly1, lx2, ly2 = lx - 0.5 * lw, ly - 0.5 * lh, lx + 0.5 * lw, ly + 0.5 * lh
    zero = torch.tensor(0.0, dtype=px1.dtype, device=px1.device)
    dx = torch.max(torch.min(px2, lx2.T) - torch.max(px1, lx1.T), zero)
    dy = torch.max(torch.min(py2, ly2.T) - torch.max(py1, ly1.T), zero)
    intersections = dx * dy
    pa = (px2 - px1) * (py2 - py1) # area
    la = (lx2 - lx1) * (ly2 - ly1) # area
    unions = (pa + la.T) - intersections
    ious = (intersections/unions).reshape(*bboxes1.shape[:-1], *bboxes2.shape[:-1])
    
    return ious
def iou_wh(bboxes1, bboxes2):
    """ calculate iou between each bbox in `bboxes1` with each bbox in `bboxes2`
    
    The bboxes should be defined by their width and height and are centered around (0,0)
    
    """ 
    
    w1 = bboxes1[..., 0].view(-1)
    h1 = bboxes1[..., 1].view(-1)
    w2 = bboxes2[..., 0].view(-1)
    h2 = bboxes2[..., 1].view(-1)

    intersections = torch.min(w1[:,None],w2[None,:]) * torch.min(h1[:,None],h2[None,:])
    unions = (w1 * h1)[:,None] + (w2 * h2)[None,:] - intersections
    ious = (intersections / unions).reshape(*bboxes1.shape[:-1], *bboxes2.shape[:-1])

    return ious

def nms(filtered_tensor, threshold):
    result = []
    for x in filtered_tensor:
        # Sort coordinates by descending confidence
        scores, order = x[:, 4].sort(0, descending=True)
        x = x[order]
        ious = iou(x,x) # get ious between each bbox in x

        # Filter based on iou
        keep = (ious > threshold).long().triu(1).sum(0, keepdim=True).t().expand_as(x) == 0

        result.append(x[keep].view(-1, 6).contiguous())
    return result
# class YOLOLoss(torch.nn.modules.loss._Loss):
class YOLOLoss(L.LightningModule):
    """ A loss function to train YOLO v2

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

        self.seen = int(seen+.5)
        self.coord_prefill = int(coord_prefill+.5)

        self.threshold = float(threshold)
        self.lambda_coord = float(lambda_coord)
        self.lambda_noobj = float(lambda_noobj)
        self.lambda_obj = float(lambda_obj)
        self.lambda_cls = float(lambda_cls)

        self.mse = torch.nn.MSELoss(reduction='sum')
        self.cel = torch.nn.CrossEntropyLoss(reduction='sum')

    def forward(self, x, y):
        #x : pred
        #y : target
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
        coord_mask = torch.zeros(nB, nA, nH, nW, 1, requires_grad=False, dtype=x.dtype, device=x.device)
        conf_mask = torch.ones(nB, nA, nH, nW, requires_grad=False, dtype=x.dtype, device=x.device) * self.lambda_noobj
        cls_mask = torch.zeros(nB, nA, nH, nW, requires_grad=False, dtype=torch.bool, device=x.device)
        tcoord = torch.zeros(nB, nA, nH, nW, 4, requires_grad=False, dtype=x.dtype, device=x.device)
        tconf = torch.zeros(nB, nA, nH, nW, requires_grad=False, dtype=x.dtype, device=x.device)
        tcls = torch.zeros(nB, nA, nH, nW, requires_grad=False, dtype=x.dtype, device=x.device)

        coord = torch.cat([
            x[:, :, :, :, 0:1].sigmoid(),  # X center
            x[:, :, :, :, 1:2].sigmoid(),  # Y center
            x[:, :, :, :, 2:3],  # Width
            x[:, :, :, :, 3:4],  # Height
        ], -1)

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

        if self.seen < self.coord_prefill:
            coord_mask.fill_(np.sqrt(.01 / self.lambda_coord))
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
            mask = (iou_gt_pred > self.threshold).sum(0) >= 1
            conf_mask[b][mask.view_as(conf_mask[b])] = 0

            # Find best anchor for each gt
            iou_gt_anchors = iou_wh(gt[:,2:], anchors)
            _, best_anchors = iou_gt_anchors.max(1)

            # Set masks and target values for each gt
            nGT = gt.shape[0]
            gi = gt[:, 0].clamp(0, nW-1).long()
            gj = gt[:, 1].clamp(0, nH-1).long()

            conf_mask[b, best_anchors, gj, gi] = self.lambda_obj
            tconf[b, best_anchors, gj, gi] = iou_gt_pred.view(nGT, nA, nH, nW)[torch.arange(nGT), best_anchors, gj, gi]
            coord_mask[b, best_anchors, gj, gi, :] = (2 - (gt[:, 2] * gt[:, 3]) / nPixels)[..., None]
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

        loss_coord = self.lambda_coord * self.mse(coord*coord_mask, tcoord*coord_mask) / (2 * nB)
        loss_conf = self.mse(conf*conf_mask, tconf*conf_mask) / (2 * nB)
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
		box_predictions[..., 2:] = torch.exp( 
			box_predictions[..., 2:]) * anchors 
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
	y = 1 / s * (box_predictions[..., 1:2] +
				cell_indices.permute(0, 1, 3, 2, 4)) 
	width_height = 1 / s * box_predictions[..., 2:4] 

	# Concatinating the values and reshaping them in 
	# (BATCH_SIZE, num_anchors * S * S, 6) shape 
	converted_bboxes = torch.cat( 
		(best_class, scores, x, y, width_height), dim=-1
	).reshape(batch_size, num_anchors * s * s, 6) 

	# Returning the reshaped and converted bounding box list 
	return converted_bboxes.tolist()
# %%

