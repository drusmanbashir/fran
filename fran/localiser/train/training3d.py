import os
from pathlib import Path

from fran.configs.parser import ConfigMaker
from fran.localiser.data.dataset import CTAugDetectionTrainer
from fran.localiser.transforms.tsl import TSLRegions
from fran.managers.project import Project
from ultralytics import YOLO
from utilz.fileio import load_yaml


# %%
if __name__ == "__main__":
# SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
# %%
    imsize = 256
    device = 0
    n_workers = 0
    bs = 4
    data_folder = Path("/s/tmp/nii2pt_tsl3d_debug")
    data_spec = data_folder / "data.yaml"
    project_title = "totalseg"
    project_name = "totalseg_localiser3d"
    P = Project(project_title=project_title)
    C = ConfigMaker(P)

    T = TSLRegions()
    names_ap = []
    names_lat = []
    for region in T.regions:
        names_ap.append(region + "_ap")
        names_lat.append(region + "_lat")
    names = names_ap + names_lat

    lines = [
        f"path: {data_folder}",
        "train: images",
        "val: images",
        f"nc: {len(names)}",
        "names:",
    ]
    for index, name in enumerate(names):
        lines.append(f"  {index}: {name}")
    data_spec.write_text("\n".join(lines) + "\n")
# %%
    n_epochs = 600
    common_vars_filename = os.environ["FRAN_CONF"] + "/config.yaml"
    COMMON_PATHS = load_yaml(common_vars_filename)
    yolo_project_folder = Path(COMMON_PATHS["yolo_output_folder"]) / project_name
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
    cache = False
    seed = 0
    debug = False
    deterministic = True
    plots = True
    cos_lr = True
    model = YOLO("yolo11m.pt") if resume_ckpt is None else YOLO(str(resume_ckpt))

# SECTION:-------------------- TRAIN--------------------------------------------------------------------------------------
# %%
    results = model.train(
        trainer=CTAugDetectionTrainer,
        data=data_spec,
        epochs=n_epochs,
        imgsz=imsize,
        project=yolo_project_folder,
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
        fran_project=P,
        fran_configs=C,
        fran_data_folder=data_folder,
        fran_debug=debug,
    )
# %%
