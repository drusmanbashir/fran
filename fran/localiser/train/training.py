# %%
# use the model
# results = model.train(data="coco128.yaml", epochs=3)  # train the model
# results = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = YOLO("yolov8n.pt").export(format="onnx")

# %%


if __name__ == "__main__":
    import os
    from pathlib import Path

    import SimpleITK as sitk
    import torch
    from fran.localiser.helpers import make_multiwindow_inference_tensor
    from utilz.helpers import set_autoreload

    set_autoreload()
    from fran.localiser.helpers import jpg_to_tensor
    from fran.localiser.yolo_ct_augment import (
        CTAugDetectionTrainer,
        apply_window_tensor,
    )
    from roboflow import Roboflow
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
    device = 0
    n_workers = 16
    bs = 300
    data_folder = Path("/s/xnat_shadow/totalseg2d")
    data_spec = data_folder / "jpg/data.yaml"
    project_name = "totalseg_localiser"
    single_cls = False
    # Ultralytics keeps label-validation results in labels.cache and does not
    # invalidate on class-count/name changes, so clear stale caches each run.
# %%
    for cache_fn in (
        data_folder / "jpg/train/labels.cache",
        data_folder / "jpg/valid/labels.cache",
        data_folder / "jpg/test/labels.cache",
    ):
        if cache_fn.exists():
            cache_fn.unlink()
# %%
    wandb: bool = True
    n_epochs = 600
    common_vars_filename = os.environ["FRAN_CONF"] + "/config.yaml"
    COMMON_PATHS = load_yaml(common_vars_filename)
    yolo_projec_folder = Path(COMMON_PATHS["yolo_output_folder"]) / project_name
    # resume_ckpt = Path(
    #     "/s/fran_storage/yolo_output/totalseg_localiser/train32/weights/last.pt"
    # )
    resume_ckpt = None
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
#     if wandb:
#         from ultralytics.utils import RANK
#         from wandb.integration.ultralytics import add_wandb_callback
#         from wandb.integration.ultralytics import callback as wandb_ultralytics_callback
#
#         wandb_ultralytics_callback.RANK = RANK
#         add_wandb_callback(model, enable_model_checkpointing=True)
# # %%
# SECTION:-------------------- TRAIN--------------------------------------------------------------------------------------
# %%
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

# SECTION:-------------------- Inference--------------------------------------------------------------------------------------
# %%

    model = YOLO(
        "/s/fran_storage/yolo_output/totalseg_localiser/train32/weights/best.pt"
    )
# %%

    nii_path = Path("/media/UB/datasets/kits23/images/kits23_00007.nii.gz")
    assert nii_path.exists(), f"{nii_path} does not exist"
    img = sitk.ReadImage(nii_path)

    arr = sitk.GetArrayFromImage(img)

# %%
    imsize=256
    raw_x = torch.from_numpy(arr).float().unsqueeze(0)
    rx_win = apply_window_tensor(raw_x, "a")
    rx2 = torch.mean(rx_win, 1)
    rx2_perm = rx2.permute(0, 2, 1)
    rx4 = rx2.unsqueeze(0)
    rx4 = rx4.permute(0, 1, 3, 2)
    rx4 = interpolate(rx4, (imsize, imsize))
    min_ = rx4.min()
    max_ = rx4.max()
    rx4 = (rx4 - min_) / (max_ - min_)

    # min_t = rx2.min()
    # max_t = rx2.max()
    # rx3 = (rx2-min_t)/(max_t-min_t)
    # rx4 = rx3.unsqueeze(0)
    rx4 = rx4.repeat(1, 3, 1, 1)
    rx4.shape
    res2 = model(rx4)
    rr2 = res2[0]

    from fran.localiser.helpers import draw_tensor_boxes

    draw_tensor_boxes(rx4, rr2)

# %%

# %%
# SECTION:-------------------- jpg--------------------------------------------------------------------------------------

    import torch

# %%
    fn = "/s/xnat_shadow/totalseg2d/jpg/valid/images/totalseg_s0029_a2.jpg"
    img = jpg_to_tensor(fn)
    img_torch = img.permute(2, 0, 1).float()
    img_torch = img_torch / img_torch.max()
    img_torch = img_torch.unsqueeze(0)
    im2 = interpolate(img_torch, (imsize, imsize))

    im3 = torch.flip(im2, dims=[2, 3])
    res = model(im3)
    rr = res[0]
    # draw_tensor_boxes(im3, rr)

# %%

# %%

# SECTION:-------------------- 3window--------------------------------------------------------------------------------------
    x = make_multiwindow_inference_tensor(raw_x)

    res = model(x)

# %%

# %%
# SECTION:-------------------- one window--------------------------------------------------------------------------------------
    x2 = make_singlewindow_inference_tensor(raw_x, "b")

# %%
    res = model(x2)

# %%
    rr = res[0]
    imtnsr = x2
# %%


# %%
