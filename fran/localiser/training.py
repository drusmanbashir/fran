# %%


# Use the model
# results = model.train(data="coco128.yaml", epochs=3)  # train the model
# results = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = YOLO("yolov8n.pt").export(format="onnx")

import torch
if __name__ == "__main__":
    from pathlib import Path

    import SimpleITK as sitk
    import supervision as sv
    from roboflow import Roboflow
    from torch.nn.functional import interpolate
    from ultralytics import YOLO

# %%
# SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
    fn = "/s/fran_storage/roboflow.txt"

    with open(fn, "r") as fl:
        api_k = fl.read().strip()

    rf = Roboflow(api_key=api_k)
    project = rf.workspace("roboflow-100").project("parasites-1s07h")
    version = project.version(2)
    ds_location = Path("/s/fran_storage/parasites-2")
    model = YOLO("yolo11n.pt")
    imsize = 256
# %%
# %%
# SECTION:-------------------- TRAIN--------------------------------------------------------------------------------------
    results = model.train(
        data="/s/xnat_shadow/lidc2d_yolo/data.yaml",
        epochs=100,
        imgsz=imsize,
        project="/s/yolo11_localiser",
    )
    # yolo task=detect mode=train model=yolo11s.pt data=/s/fran_storage/parasites-2/data.yaml epochs=40 imgsz=640 plots=True
# %%
# SECTION:-------------------- Inference--------------------------------------------------------------------------------------
    # Read JPG image using torchvision
    model = YOLO("/s/yolo11_localiser/train4/weights/last.pt")
    img_path = "/s/xnat_shadow/lidc2d_yolo/valid/images/lidc2_0001_2.jpg"
    nii_path = "/home/ub/Documents/wb.nii.gz"
    # tnsr_path = "/s/xnat_shadow/lidc2d/images/lidc2_0001_2.pt"
    # tnsr = torch.load(tnsr_path)
    img = sitk.ReadImage(nii_path)  # [H,W,C]
    arr = sitk.GetArrayFromImage(img)  # [C,H,W]
# %%
    tnsr = torch.from_numpy(arr)  # Convert to PyTorch tensor

    tnsr = tnsr.float()
    tnsr2d = torch.mean(tnsr,dim=0)  
    t2 = tnsr2d
    t2 = t2.unsqueeze(0).unsqueeze(0)  # Add channel dimension [1,H,W]
    t2.shape

# %%


    # Convert to float and add batch dimension

    # Scale tensor between 0 and 1
    pads = 80
    t2 = (t2 - t2.min()) / (t2.max() - t2.min())
    from torch.nn import functional as F
    t2p = F.pad(t2,(pads,pads,pads,pads))
    t3 = interpolate(t2p, (imsize, imsize))
    t3 = t3.repeat(1, 3, 1, 1)

    # Display original tensor
# %%
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.imshow(t3[0].permute(1, 2, 0).cpu())  # Convert [C,H,W] to [H,W,C] for display
    plt.axis("off")
    plt.title("Original Image")
    plt.show()

# %%
    # Resize to 256x256

    # Run infrence
    res = model(t3)[0]
    rr = res[0]
    rr.save()
    bbox = rr.boxes
    rr.show()
# %%
    detections = sv.Detections(rr.boxes.data, class_ids=rr.boxes.cls.cpu().numpy())

    # Convert tensor to numpy array for visualization
    image_np = (t2[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

    # Create supervision annotators
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=image_np, detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections
    )

    # Display the image
    sv.plot_image(annotated_image)
# %%


