# %%
from fran.localiser.inference import get_model
import supervision as sv
import SimpleITK as sitk
from torch.nn.functional import interpolate
import torchvision.transforms as T
from torchvision.io import read_image

# Use the model
# results = model.train(data="coco128.yaml", epochs=3)  # train the model
# results = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = YOLO("yolov8n.pt").export(format="onnx")  
from ultralytics import YOLO
import lightning as L
from pathlib import Path
import torch
from ultralytics import YOLO
from roboflow import Roboflow

if __name__ == "__main__":

# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
    fn  ="/s/fran_storage/roboflow.txt"

    with open (fn,'r') as fl:
        api_k = fl.read().strip()

    rf = Roboflow(api_key=api_k)
    project = rf.workspace("roboflow-100").project("parasites-1s07h")
    version = project.version(2)
    ds_location = Path("/s/fran_storage/parasites-2")
    model = YOLO("yolo11n.pt")
    imsize = 256
# %%
# %%
#SECTION:-------------------- TRAIN--------------------------------------------------------------------------------------
    results = model.train(data="/s/xnat_shadow/lidc2d_yolo/data.yaml", epochs=100, imgsz=imsize,project="/s/yolo11_localiser")
# yolo task=detect mode=train model=yolo11s.pt data=/s/fran_storage/parasites-2/data.yaml epochs=40 imgsz=640 plots=True
# %%
#SECTION:-------------------- Inference--------------------------------------------------------------------------------------
    # Read JPG image using torchvision
    model = YOLO("/s/yolo11_localiser/train4/weights/last.pt")
    img_path = "/s/xnat_shadow/lidc2d_yolo/valid/images/lidc2_0001_2.jpg"
    tnsr_path = "/s/xnat_shadow/lidc2d/images/lidc2_0001_2.pt"
    tnsr = torch.load(tnsr_path)
    tnsr = tnsr.permute(0,2,1)
    
    # Convert to float and add batch dimension
    tnsr = tnsr.float()
    tnsr = tnsr.unsqueeze(0)  # [B,C,H,W]
    
    # Scale tensor between 0 and 1
    tnsr = (tnsr - tnsr.min()) / (tnsr.max() - tnsr.min())
    t2 = interpolate(tnsr, (imsize,imsize))
    t3 = t2.repeat(1,3,1,1)
    
    # Display original tensor
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.imshow(t2[0].permute(1, 2, 0).cpu())  # Convert [C,H,W] to [H,W,C] for display
    plt.axis('off')
    plt.title('Original Image')
    plt.show()
    
# %%
    # Resize to 256x256
    
    # Run inference
    res = model(t3)[0]
    rr = res[0]
    rr.save()
    bbox=  rr.boxes
    rr.show()
# %%
    detections = sv.Detections(rr.boxes.data, class_ids=rr.boxes.cls.cpu().numpy())

    # Convert tensor to numpy array for visualization
    image_np = (t2[0].permute(1,2,0).cpu().numpy() * 255).astype('uint8')

    # Create supervision annotators
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=image_np, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    # Display the image
    sv.plot_image(annotated_image)
# %%
