# %%
# Use the model
# results = model.train(data="coco128.yaml", epochs=3)  # train the model
# results = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = YOLO("yolov8n.pt").export(format="onnx")

import torch
from fran.localiser.yolo_ct_augment import CTAugDetectionTrainer
from fran.transforms.spatialtransforms import Project2D
from utilz.imageviewers import ImageMaskViewer
if __name__ == "__main__":
    import os
    from pathlib import Path

    import SimpleITK as sitk
    import supervision as sv
    from roboflow import Roboflow
    from torch.nn.functional import interpolate
    from ultralytics import YOLO
    from utilz.fileio import load_yaml

# %%
# SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
    fn = "/s/fran_storage/conf/roboflow.txt"

    with open(fn, "r") as fl:
        api_k = fl.read().strip()

    rf = Roboflow(api_key=api_k)
    project = rf.workspace("roboflow-100").project("parasites-1s07h")
    version = project.version(2)
    ds_location = Path("/s/fran_storage/parasites-2")
    imsize = 256
    device = 1
    n_workers = 16
    bs =256
    data_folder = Path("/s/xnat_shadow/totalseg2d")
    data_spec = data_folder / "jpg/data.yaml"
    project_name = "totalseg_localiser"
    single_cls = False
    # Ultralytics keeps label-validation results in labels.cache and does not
    # invalidate on class-count/name changes, so clear stale caches each run.
    for cache_fn in (
        data_folder / "jpg/train/labels.cache",
        data_folder / "jpg/valid/labels.cache",
        data_folder / "jpg/test/labels.cache",
    ):
        if cache_fn.exists():
            cache_fn.unlink()
# %%
    n_epochs = 600
    common_vars_filename = os.environ["FRAN_CONF"] + "/config.yaml"
    COMMON_PATHS = load_yaml(common_vars_filename)
    yolo_projec_folder = Path(COMMON_PATHS["yolo_output_folder"]) / project_name
    resume_ckpt = None
    # resume_ckpt = Path("/s/fran_storage/yolo_output/totalseg_localiser/train4/weights/last.pt")
    patience = 40
    optimiser = "AdamW"
    lr0 = 1e-3
    lrf = 1e-2
    weight_decay = 5e-4
    warmup_epochs = 3
    close_mosaic = 10
    mosaic = 0.5
    scale = 0.3
    translate = 0.1
    fliplr = 0.5
    flipud=0.1
    cache = True
    seed = 0
    deterministic = True
    plots = True
    cos_lr = True
    # model = YOLO("yolo11n.pt") if resume_ckpt is None else YOLO(str(resume_ckpt))
    model = YOLO("yolo11m.pt") if resume_ckpt is None else YOLO(str(resume_ckpt))
# %%
# SECTION:-------------------- TRAIN--------------------------------------------------------------------------------------
    results = model.train(
        trainer=CTAugDetectionTrainer,
        data=data_spec,
        epochs=n_epochs,
        imgsz=imsize,
        project=yolo_projec_folder,
        batch=bs,
        device=device,
        workers=n_workers,
        patience=patience,
        optimizer=optimiser,
        lr0=lr0,
        lrf=lrf,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        close_mosaic=close_mosaic,
        mosaic=mosaic,
        scale=scale,
        translate=translate,
        fliplr=fliplr,
        flipud=flipud,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        cache=cache,
        seed=seed,
        deterministic=deterministic,
        plots=plots,
        cos_lr=cos_lr,
        resume=resume_ckpt is not None,
        val=True,
    )
    # yolo task=detect mode=train model=yolo11s.pt data=/s/fran_storage/parasites-2/data.yaml epochs=40 imgsz=640 plots=True
# %%
# SECTION:-------------------- Inference--------------------------------------------------------------------------------------
    # Read JPG image using torchvision
    model = YOLO("/s/fran_storage/yolo_output/totalseg_localiser/train15/weights/best.pt")
    img_path = "/s/xnat_shadow/lidc2d_yolo/valid/images/lidc2_0001_2.jpg"
    nii_path = "/s/xnat_shadow/nodes/images_pending/nodes_72_20210714_CAP1p5SoftTissue.nii.gz"
    # tnsr_path = "/s/xnat_shadow/lidc2d/images/lidc2_0001_2.pt"
    # tnsr = torch.load(tnsr_path)
    img = sitk.ReadImage(nii_path)  # [H,W,C]
    arr = sitk.GetArrayFromImage(img)  # [C,H,W]
# %%
    tnsr = torch.from_numpy(arr)  # Convert to PyTorch tensor

    tnsr = tnsr.float()
    tnsr2d = torch.mean(tnsr,dim=1)  
    t2 = tnsr2d
    t2 = t2.unsqueeze(0).unsqueeze(0)  # Add channel dimension [1,H,W]
    t2.shape

# %%


    # Convert to float and add batch dimension

    # Scale tensor between 0 and 1
    pads = 0
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
# %%
    for rr in res:
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
