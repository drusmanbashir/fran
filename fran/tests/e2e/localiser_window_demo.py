from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch
from fran.localiser.preprocessing.data.nii2pt import _PreprocessorNII2PTWorkerBase

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pick_case(data_folder: Path, case_stem: str | None) -> dict[str, Path]:
    fldr_imgs = data_folder / "images"
    fldr_lms = data_folder / "lms"
    if case_stem is not None:
        return {
            "image": fldr_imgs / f"{case_stem}.nii.gz",
            "lm": fldr_lms / f"{case_stem}.nii.gz",
        }
    for image in sorted(fldr_imgs.glob("*.nii.gz")):
        lm = fldr_lms / image.name
        if lm.exists():
            return {"image": image, "lm": lm}
    raise FileNotFoundError(f"No matching image/lm pair found in {data_folder}")


def run_demo(
    data_folder: str,
    case_stem: str | None = None,
    projection: int = 1,
    output: str = "/tmp/localiser_window_demo.png",
) -> dict[str, object]:
    worker = _PreprocessorNII2PTWorkerBase(output_folder="/tmp")
    dici = _pick_case(Path(data_folder), case_stem)
    steps = ["L", "E", "O", "Win", f"P{projection}"]

    for step in steps:
        if step[0] == "P":
            dici["image"] = dici["image"].float()
            dici["lm"] = dici["lm"].float()
        dici = worker.transforms_dict[step](dici)
        image_key = f"image{projection}" if step[0] == "P" else "image"
        print(step, tuple(dici[image_key].shape))

    tensor_key = f"image{projection}"
    tensor_out = Path(output).with_suffix(".pt")
    image = dici[tensor_key].contiguous()
    torch.save(image, tensor_out)
    image = torch.load(tensor_out, weights_only=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["bone", "chest", "abdomen"]
    for ind, ax in enumerate(axes):
        ax.imshow(image[ind].T.cpu(), cmap="gray")
        ax.set_title(titles[ind])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("tensor_out", tensor_out)
    print("figure_out", output)
    return {
        "tensor_out": tensor_out,
        "figure_out": Path(output),
        "shape": tuple(image.shape),
        "source_image": dici["image"].meta["filename_or_obj"],
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("localiser_window_demo")
    p.add_argument("--data-folder", default="/s/xnat_shadow/totalseg")
    p.add_argument("--case-stem", default=None)
    p.add_argument("--projection", type=int, default=1)
    p.add_argument("--output", default="/tmp/localiser_window_demo.png")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    run_demo(
        data_folder=args.data_folder,
        case_stem=args.case_stem,
        projection=args.projection,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
