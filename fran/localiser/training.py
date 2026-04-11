# %%
# use the model
# results = model.train(data="coco128.yaml", epochs=3)  # train the model
# results = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = YOLO("yolov8n.pt").export(format="onnx")


# %%
from fran.transforms.intensitytransforms import standardize


if __name__ == "__main__":
    import os
    from pathlib import Path

    import SimpleITK as sitk
    import supervision as sv
    import torch
    from fran.localiser.yolo_ct_augment import (
        WINDOW_PRESETS,
        CTAugDetectionTrainer,
        apply_window_tensor,
    )
    from roboflow import Roboflow
    from torch.nn import functional as F
    from torch.nn.functional import interpolate
    from ultralytics import YOLO
    from utilz.fileio import load_yaml

# SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
# %%
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
    bs = 256
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
    flipud = 0.1
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
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    model = YOLO(
        "/s/fran_storage/yolo_output/totalseg_localiser/train21/weights/best.pt"
    )
# %%
    nii_path = "/media/UB/datasets/kits23/images/kits23_00005.nii.gz"
    img = sitk.ReadImage(nii_path)
    arr = sitk.GetArrayFromImage(img)

    x = torch.from_numpy(arr).float().unsqueeze(0)
    chs = []
    for window in WINDOW_PRESETS:
        chs.append(apply_window_tensor(x, window))
    x = torch.cat(chs, dim=0)
    x = torch.mean(x, dim=1)
    x = x.unsqueeze(0)
    x = interpolate(x, (imsize, imsize))

    res = model(x)
    rr = res[0]

    image_np = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    image_np = np.ascontiguousarray(image_np.copy())
    boxes = rr.boxes.xyxy.cpu().numpy().astype(int)
    confs = rr.boxes.conf.cpu().numpy()
    clss = rr.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, clss):
        x1, y1, x2, y2 = box
        label = f"{rr.names[cls_id]} {conf:.2f}"
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_np,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    plt.figure(figsize=(8, 8))
    plt.imshow(image_np)
    plt.axis("off")
    plt.title("YOLO prediction")
    plt.show()

# %%
