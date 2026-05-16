import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from fran.inference.base import filter_existing_files
from fran.run.inference.infer import (
    InferenceSpec,
    build_inferer,
    resolve_inferer_cls,
    resolve_input_images,
)
from utilz.rayz import shutdown_actors

N_WORKERS = 2


@dataclass(frozen=True)
class WorkerConfig:
    spec: InferenceSpec
    chunksize: int
    overwrite: bool
    safe_mode: bool
    patch_overlap: float | None


def filter_pending_images(
    input_images: list[Path], cfg: WorkerConfig, gpus: list[int]
) -> list[Path]:
    inferer = build_inferer(
        cfg.spec,
        gpus=[gpus[0]],
        safe_mode=cfg.safe_mode,
        patch_overlap=cfg.patch_overlap,
    )
    if cfg.overwrite is False:
        input_images = filter_existing_files(input_images, inferer.output_folder)
    return input_images


def split_file_chunks(input_images: list[Path]) -> list[list[Path]]:
    midpoint = (len(input_images) + 1) // N_WORKERS
    return [input_images[:midpoint], input_images[midpoint:]]


def run_multi_gpu(cfg: WorkerConfig, input_images: list[Path], gpus: list[int]):
    import ray

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
    if ray.is_initialized() is False:
        ray.init(num_gpus=len(gpus), ignore_reinit_error=True)

    @ray.remote(num_cpus=4, num_gpus=1)
    class InfererActor:
        def process(self, worker_cfg: WorkerConfig, img_paths, requested_gpu):
            inferer = build_inferer(
                worker_cfg.spec,
                gpus=[0],
                safe_mode=worker_cfg.safe_mode,
                patch_overlap=worker_cfg.patch_overlap,
            )
            inferer.run(
                img_paths,
                overwrite=worker_cfg.overwrite,
                chunksize=worker_cfg.chunksize,
            )
            return {
                "requested_gpu": requested_gpu,
                "ray_gpu_ids": ray.get_gpu_ids(),
                "processed": len(img_paths),
            }

    chunks = split_file_chunks(input_images)
    chunk_jobs = [(chunk, gpu) for chunk, gpu in zip(chunks, gpus) if len(chunk) > 0]
    actors = [InfererActor.remote() for _ in chunk_jobs]
    futures = [
        actor.process.remote(worker_cfg=cfg, img_paths=chunk, requested_gpu=gpu)
        for actor, (chunk, gpu) in zip(actors, chunk_jobs)
    ]
    try:
        return ray.get(futures)
    finally:
        shutdown_actors(actors)


def main(args):
    if len(args.gpus) != N_WORKERS:
        raise ValueError("Pass exactly two GPUs to --gpus")

    spec = resolve_inferer_cls(args)
    worker_cfg = WorkerConfig(
        spec=spec,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
        safe_mode=True,
        patch_overlap=args.patch_overlap,
    )
    input_images = resolve_input_images(args.folder, args.dataset)
    if len(input_images) == 0:
        raise RuntimeError("No input images found")
    input_images = filter_pending_images(input_images, worker_cfg, args.gpus)
    if len(input_images) == 0:
        print("No images left after filtering existing predictions.")
        return []
    results = run_multi_gpu(worker_cfg, input_images, args.gpus)
    print("Ray results:", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mnemonic-driven multi-GPU inference runner"
    )
    parser.add_argument("mnemonic")
    parser.add_argument(
        "--localiser-type", type=str.lower, choices=["full", "whole", "yolo"]
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--folder", nargs="+")
    source.add_argument("--dataset", nargs="+")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--chunksize", type=int, default=4)
    parser.add_argument("--patch-overlap", type=float, default=0.2)
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("--localiser_labels", type=str)
    args = parser.parse_known_args()[0]
    main(args)
