from __future__ import annotations

import os

import pandas as pd
import pytest
import torch

from fran.transforms.fg_indices import FgBgToIndicesSubsampled
from fran.transforms.fg_indices import FgBgToIndicesd2


pytestmark = pytest.mark.skipif(
    "FRAN_CONF" not in os.environ,
    reason="Fixed-spacing smoke tests require FRAN_CONF dataset registry.",
)


def test_bg_subsample_indices_transform_ignores_extra_labels_and_strides_bg():
    lm = torch.tensor(
        [[
            [[0, 1, 2], [3, 0, 2]],
            [[1, 3, 0], [2, 2, 0]],
        ]],
        dtype=torch.uint8,
    )
    image = torch.ones_like(lm, dtype=torch.float32)
    out = FgBgToIndicesSubsampled(
        keys=["lm"], ignore_labels=[2], image_key="image", subsample_bg=2
    )({"lm": lm, "image": image})
    stock = FgBgToIndicesd2(keys=["lm"], ignore_labels=[2], image_key="image")(
        {"lm": lm, "image": image}
    )

    assert torch.equal(out["lm_fg_indices"], stock["lm_fg_indices"])
    assert torch.equal(out["lm_bg_indices"], stock["lm_bg_indices"][::2])


def test_bg_subsample_none_matches_stock():
    lm = torch.tensor([[[0, 1, 2], [3, 0, 2]]], dtype=torch.uint8)
    image = torch.ones_like(lm, dtype=torch.float32)
    out = FgBgToIndicesSubsampled(
        keys=["lm"], ignore_labels=[2], image_key="image", subsample_bg=None
    )({"lm": lm, "image": image})
    stock = FgBgToIndicesd2(keys=["lm"], ignore_labels=[2], image_key="image")(
        {"lm": lm, "image": image}
    )

    assert torch.equal(out["lm_fg_indices"], stock["lm_fg_indices"])
    assert torch.equal(out["lm_bg_indices"], stock["lm_bg_indices"])


def test_fixed_spacing_fg_only_nifti_sampler_smoke_four_files(tmp_path):
    from fran.data.dataregistry import DS
    from fran.managers.project import Project
    from fran.preprocessing.fixed_spacing_fgonly import NiftiSubsampledBgResamplerLocal

    src_folder = DS["kits23_short"].folder
    if not src_folder.exists():
        pytest.skip(f"Short test dataset is unavailable: {src_folder}")

    image_fns = sorted((src_folder / "images").glob("*.nii.gz"))[:4]
    if len(image_fns) < 4:
        pytest.skip(f"Need four short dataset cases in: {src_folder}")

    project = Project("test")
    plan = {
        "spacing": [0.8, 0.8, 1.5],
        "patch_size": [32, 32, 32],
        "expand_by": 0,
        "datasources": "kits23_short",
        "remapping_source": "None",
        "fg_indices_exclude": [2],
        "subsample_bg": 5,
    }
    worker = NiftiSubsampledBgResamplerLocal(
        project=project,
        plan=plan,
        data_folder=src_folder,
        output_folder=tmp_path,
        device="cpu",
        debug=False,
    )
    worker.create_output_folders()

    for image_fn in image_fns:
        lm_fn = src_folder / "lms" / image_fn.name
        row = pd.Series(
            {
                "image": str(image_fn),
                "lm": str(lm_fn),
                "remapping_source": None,
                "ds": "kits23",
            }
        )
        worker._process_row(row)

        out = torch.load(
            worker.indices_subfolder / f"{image_fn.name.removesuffix('.nii.gz')}.pt",
            map_location="cpu",
            weights_only=False,
        )

        assert "lm_fg_indices" in out
        assert "lm_bg_indices" in out
        assert (worker.output_folder / "images" / f"{image_fn.name.removesuffix('.nii.gz')}.pt").exists()
        assert (worker.output_folder / "lms" / f"{image_fn.name.removesuffix('.nii.gz')}.pt").exists()

        lm = torch.load(
            worker.output_folder / "lms" / f"{image_fn.name.removesuffix('.nii.gz')}.pt",
            map_location="cpu",
            weights_only=False,
        )
        fg = out["lm_fg_indices"]
        assert fg.numel() > 0
        vals = lm.reshape(-1)[fg.long()]
        assert torch.all(vals != 0)
        assert torch.all(vals != 2)
        bg = out["lm_bg_indices"]
        assert bg.numel() > 0
        assert bg.numel() < lm.numel()

    assert len(list(worker.indices_subfolder.glob("*.pt"))) == 4
    assert len(list((worker.output_folder / "images").glob("*.pt"))) == 4
    assert len(list((worker.output_folder / "lms").glob("*.pt"))) == 4
