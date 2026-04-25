import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import SimpleITK as sitk
from fran.inference.cascade import CascadeInferer
from fran.inference.scoring import compute_dice_fran
from fran.managers.project import Project


def chunk_list(items, n_chunks):
    if n_chunks <= 1 or len(items) <= 1:
        return [items]
    n_chunks = min(n_chunks, len(items))
    q, r = divmod(len(items), n_chunks)
    chunks = []
    start = 0
    for i in range(n_chunks):
        size = q + (1 if i < r else 0)
        end = start + size
        chunks.append(items[start:end])
        start = end
    return [c for c in chunks if c]


def case_processed_already(img_fn: Path, output_folder: Path) -> bool:
    pred_fn = output_folder / img_fn.name
    if pred_fn.exists():
        print(
            "Prediction already exists for {0} in {1}".format(
                img_fn.name, output_folder
            )
        )
        return True
    return False


def run_single_gpu(
    images,
    project_title,
    run_w,
    run_p,
    localiser_labels,
    gpu,
    overwrite,
    chunksize,
    safe_mode,
    save_localiser,
    patch_overlap,
    k_largest,
):
    inferer = CascadeInferer(
        run_w=run_w,
        run_p=run_p,
        project_title=project_title,
        devices=[gpu],
        localiser_labels=localiser_labels,
        safe_mode=safe_mode,
        save_localiser=save_localiser,
        patch_overlap=patch_overlap,
        k_largest=k_largest,
        save=True,
        save_channels=False,
    )
    inferer.run(images, chunksize=chunksize, overwrite=overwrite)
    return len(images), str(inferer.output_folder)


@dataclass
class RayConfig:
    project_title: str
    run_w: str
    run_p: str
    localiser_labels: list[int]
    overwrite: bool
    chunksize: int
    safe_mode: bool
    save_localiser: bool
    patch_overlap: float
    k_largest: int | None


def run_multi_gpu_ray(images, gpus, cfg: RayConfig):
    import ray

    # Restrict Ray to the exact GPUs selected by user.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    if not ray.is_initialized():
        ray.init(num_gpus=len(gpus), ignore_reinit_error=True)

    @ray.remote(num_cpus=4, num_gpus=1)
    class PredictActor:
        def process(self, cfg_dict, img_paths):
            cfg_obj = RayConfig(**cfg_dict)
            # Inside actor, assigned GPU appears as cuda:0.
            inferer = CascadeInferer(
                run_w=cfg_obj.run_w,
                run_p=cfg_obj.run_p,
                project_title=cfg_obj.project_title,
                devices=[0],
                localiser_labels=cfg_obj.localiser_labels,
                safe_mode=cfg_obj.safe_mode,
                save_localiser=cfg_obj.save_localiser,
                patch_overlap=cfg_obj.patch_overlap,
                k_largest=cfg_obj.k_largest,
                save=True,
                save_channels=False,
            )
            inferer.run(
                img_paths, chunksize=cfg_obj.chunksize, overwrite=cfg_obj.overwrite
            )
            return {
                "processed": len(img_paths),
                "output_folder": str(inferer.output_folder),
            }

    chunks = chunk_list(images, len(gpus))
    actors = [PredictActor.remote() for _ in chunks]
    futures = [a.process.remote(asdict(cfg), chunk) for a, chunk in zip(actors, chunks)]
    return ray.get(futures)


def compute_scores(images, masks_folder, output_folder, n_classes=None):
    if masks_folder is None:
        return []

    scores = []
    masks_folder = Path(masks_folder)
    output_folder = Path(output_folder)

    for img in images:
        img = Path(img)
        gt_fn = masks_folder / img.name
        pred_fn = output_folder / img.name
        if not (gt_fn.exists() and pred_fn.exists()):
            continue

        gt = sitk.ReadImage(str(gt_fn))
        pred = sitk.ReadImage(str(pred_fn))
        if n_classes is None:
            gt_max = int(sitk.GetArrayViewFromImage(gt).max())
            pr_max = int(sitk.GetArrayViewFromImage(pred).max())
            classes = max(gt_max, pr_max) + 1
        else:
            classes = n_classes

        dice = compute_dice_fran(pred, gt, classes)
        scores.append(
            {
                "case": img.name,
                "dice": dice.detach().cpu().tolist()
                if hasattr(dice, "detach")
                else dice,
                "n_classes": classes,
            }
        )

    return scores


def main(args):
    imgs_folder = Path(args.images_folder)
    images = sorted([p for p in imgs_folder.glob("*") if p.is_file()])

    if not images:
        raise RuntimeError("No images found in {0}".format(imgs_folder))

    # Pre-filter existing predictions for clearer work split/logging.
    p_tmp = Project(project_title=args.title)
    tmp_inf = CascadeInferer(
        run_w=args.run_w,
        run_p=args.run_p,
        project_title=args.title,
        devices=[args.gpus[0]],
        localiser_labels=args.localiser_labels,
        safe_mode=args.safe_mode,
        save_localiser=args.save_localiser,
        patch_overlap=args.patch_overlap,
        k_largest=args.k_largest,
        save=True,
        save_channels=False,
    )
    output_folder = tmp_inf.output_folder
    del p_tmp
    del tmp_inf

    if not args.overwrite:
        images = [
            img for img in images if not case_processed_already(img, output_folder)
        ]

    if len(images) == 0:
        print("No images left after filtering existing predictions.")
        return

    if len(args.gpus) > 1:
        cfg = RayConfig(
            project_title=args.title,
            run_w=args.run_w,
            run_p=args.run_p,
            localiser_labels=args.localiser_labels,
            overwrite=args.overwrite,
            chunksize=args.chunksize,
            safe_mode=args.safe_mode,
            save_localiser=args.save_localiser,
            patch_overlap=args.patch_overlap,
            k_largest=args.k_largest,
        )
        ray_res = run_multi_gpu_ray([str(p) for p in images], args.gpus, cfg)
        print("Ray results:", ray_res)
    else:
        n_done, _ = run_single_gpu(
            images=[str(p) for p in images],
            project_title=args.title,
            run_w=args.run_w,
            run_p=args.run_p,
            localiser_labels=args.localiser_labels,
            gpu=args.gpus[0],
            overwrite=args.overwrite,
            chunksize=args.chunksize,
            safe_mode=args.safe_mode,
            save_localiser=args.save_localiser,
            patch_overlap=args.patch_overlap,
            k_largest=args.k_largest,
        )
        print("Processed on single GPU:", n_done)

    scores = compute_scores(images, args.masks_folder, output_folder, args.n_classes)
    if scores:
        out_json = output_folder / "predict_scores.json"
        out_json.write_text(json.dumps(scores, indent=2))
        print("Wrote scores to", out_json)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cascade predictor")
    parser.add_argument("-t", "--title", required=True, help="project title")
    parser.add_argument("--run-w", required=True, help="localiser run id")
    parser.add_argument("--run-p", required=True, help="patch run id")
    parser.add_argument("--localiser-labels", nargs="+", type=int, required=True)
    parser.add_argument("-i", "--images-folder", required=True)
    parser.add_argument("-m", "--masks-folder")
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--chunksize", type=int, default=4)
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--save-localiser", action="store_true")
    parser.add_argument("--patch-overlap", type=float, default=0.2)
    parser.add_argument("--k-largest", type=int, default=None)
    parser.add_argument("--n-classes", type=int, default=None)

    main(parser.parse_args())
