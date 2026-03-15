from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import numpy as np
import torch
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import (
    convert_dim_to_conv_op,
    get_matching_instancenorm,
)
from nnunetv2.experiment_planning.experiment_planners.network_topology import (
    get_pool_and_conv_props,
)
from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import (
    ResEncUNetPlanner,
    nnUNetPlannerResEncL,
    nnUNetPlannerResEncM,
    nnUNetPlannerResEncXL,
)

RESENC_PRESET_TARGETS_GB = {
    "M": 8.0,
    "L": 24.0,
    "XL": 40.0,
}

_ENCODER_BLOCKS = (1, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)
_DECODER_BLOCKS = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
_REFERENCE_VAL_3D = 680000000
_REFERENCE_VAL_2D = 135000000
_REFERENCE_GB = 8
_REFERENCE_BS_2D = 12
_REFERENCE_BS_3D = 2
_MAX_FEATURES_2D = 512
_MAX_FEATURES_3D = 320
_BASE_FEATURES = 32
_MIN_FEATURE_EDGE = 4
_MIN_BATCH_SIZE = 2
_MAX_DATASET_COVERED = 0.05


def infer_resenc_preset_name(memory_target_gb: float) -> str:
    if memory_target_gb <= 12:
        return "M"
    if memory_target_gb <= 32:
        return "L"
    return "XL"


def _features_per_stage(num_stages: int, max_num_features: int) -> tuple[int, ...]:
    return tuple(
        min(max_num_features, _BASE_FEATURES * 2**i) for i in range(num_stages)
    )


def _planner_class_for_target(memory_target_gb: float):
    preset = infer_resenc_preset_name(memory_target_gb)
    if preset == "M":
        return nnUNetPlannerResEncM
    if preset == "L":
        return nnUNetPlannerResEncL
    return nnUNetPlannerResEncXL


def _reference_value_for_dim(dim: int) -> tuple[int, int]:
    if dim == 2:
        return _REFERENCE_VAL_2D, _REFERENCE_BS_2D
    if dim == 3:
        return _REFERENCE_VAL_3D, _REFERENCE_BS_3D
    raise ValueError(f"Unsupported dimensionality: {dim}")


def _build_arch_kwargs(
    spacing: tuple[float, ...],
    patch_size: tuple[int, ...],
) -> tuple[dict, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    dim = len(spacing)
    conv_op = convert_dim_to_conv_op(dim)
    norm_op = get_matching_instancenorm(conv_op)
    max_num_features = _MAX_FEATURES_2D if dim == 2 else _MAX_FEATURES_3D
    _, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, _ = get_pool_and_conv_props(
        spacing,
        patch_size,
        _MIN_FEATURE_EDGE,
        999999,
    )
    num_stages = len(pool_op_kernel_sizes)
    arch_kwargs = {
        "n_stages": num_stages,
        "features_per_stage": _features_per_stage(num_stages, max_num_features),
        "conv_op": conv_op,
        "kernel_sizes": conv_kernel_sizes,
        "strides": pool_op_kernel_sizes,
        "n_blocks_per_stage": _ENCODER_BLOCKS[:num_stages],
        "n_conv_per_stage_decoder": _DECODER_BLOCKS[: num_stages - 1],
        "conv_bias": True,
        "norm_op": norm_op,
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": torch.nn.LeakyReLU,
        "nonlin_kwargs": {"inplace": True},
    }
    return arch_kwargs, tuple(patch_size), tuple(pool_op_kernel_sizes), tuple(
        conv_kernel_sizes
    )


def _estimate_vram(
    patch_size: tuple[int, ...],
    input_channels: int,
    output_channels: int,
    arch_kwargs: dict,
    deep_supervision: bool,
) -> int:
    net = ResidualEncoderUNet(
        input_channels=input_channels,
        num_classes=output_channels,
        deep_supervision=deep_supervision,
        **arch_kwargs,
    )
    return int(net.compute_conv_feature_map_size(patch_size))


def build_resenc_model(
    spacing: Iterable[float],
    patch_size: Iterable[int],
    input_channels: int,
    output_channels: int,
    deep_supervision: bool = True,
):
    spacing = tuple(float(x) for x in spacing)
    patch_size = tuple(int(x) for x in patch_size)
    arch_kwargs, _, _, _ = _build_arch_kwargs(spacing, patch_size)
    return ResidualEncoderUNet(
        input_channels=input_channels,
        num_classes=output_channels,
        deep_supervision=deep_supervision,
        **arch_kwargs,
    )


def plan_like_resenc_config(
    spacing: Iterable[float],
    median_shape: Iterable[int],
    num_training_cases: int,
    input_channels: int,
    output_channels: int,
    memory_target_gb: float,
    deep_supervision: bool = True,
) -> dict:
    spacing = tuple(float(x) for x in spacing)
    median_shape = tuple(int(x) for x in median_shape)
    dim = len(spacing)
    if dim not in (2, 3):
        raise ValueError(f"Unsupported dimensionality: {dim}")

    reference_val, reference_bs = _reference_value_for_dim(dim)
    reference = reference_val * (float(memory_target_gb) / _REFERENCE_GB)

    tmp = 1 / np.array(spacing)
    if dim == 3:
        initial_patch_size = [
            round(i) for i in tmp * (256**3 / np.prod(tmp)) ** (1 / 3)
        ]
    else:
        initial_patch_size = [
            round(i) for i in tmp * (2048**2 / np.prod(tmp)) ** (1 / 2)
        ]
    initial_patch_size = tuple(
        min(i, j) for i, j in zip(initial_patch_size, median_shape[:dim])
    )

    arch_kwargs, patch_size, strides, kernel_sizes = _build_arch_kwargs(
        spacing, initial_patch_size
    )
    estimate = _estimate_vram(
        patch_size, input_channels, output_channels, arch_kwargs, deep_supervision
    )

    while (estimate / reference_bs * 2) > reference:
        axis_to_be_reduced = int(
            np.argsort([i / j for i, j in zip(patch_size, median_shape[:dim])])[-1]
        )
        patch_size_reduced = list(deepcopy(patch_size))
        shape_must_be_divisible_by = [int(np.prod(s)) for s in strides]
        patch_size_reduced[axis_to_be_reduced] -= max(1, shape_must_be_divisible_by[axis_to_be_reduced])
        patch_size_reduced[axis_to_be_reduced] = max(
            patch_size_reduced[axis_to_be_reduced], 2 * _MIN_FEATURE_EDGE
        )

        arch_kwargs, patch_size, strides, kernel_sizes = _build_arch_kwargs(
            spacing, tuple(patch_size_reduced)
        )
        estimate = _estimate_vram(
            patch_size, input_channels, output_channels, arch_kwargs, deep_supervision
        )

    batch_size = round((reference / estimate) * reference_bs)
    approximate_n_voxels_dataset = float(
        np.prod(np.asarray(median_shape), dtype=np.float64) * num_training_cases
    )
    bs_corresponding_to_5_percent = round(
        approximate_n_voxels_dataset
        * _MAX_DATASET_COVERED
        / np.prod(patch_size, dtype=np.float64)
    )
    batch_size = max(min(batch_size, bs_corresponding_to_5_percent), _MIN_BATCH_SIZE)

    planner_cls = _planner_class_for_target(memory_target_gb)
    return {
        "memory_target_gb": float(memory_target_gb),
        "preset_hint": infer_resenc_preset_name(memory_target_gb),
        "planner_class_hint": planner_cls.__name__,
        "base_planner_class": ResEncUNetPlanner.__name__,
        "patch_size": tuple(int(x) for x in patch_size),
        "median_shape": median_shape,
        "batch_size": int(batch_size),
        "n_stages": int(arch_kwargs["n_stages"]),
        "features_per_stage": tuple(int(x) for x in arch_kwargs["features_per_stage"]),
        "strides": tuple(tuple(int(y) for y in x) for x in arch_kwargs["strides"]),
        "kernel_sizes": tuple(
            tuple(int(y) for y in x) for x in arch_kwargs["kernel_sizes"]
        ),
        "n_blocks_per_stage": tuple(
            int(x) for x in arch_kwargs["n_blocks_per_stage"]
        ),
        "n_conv_per_stage_decoder": tuple(
            int(x) for x in arch_kwargs["n_conv_per_stage_decoder"]
        ),
        "vram_estimate": int(estimate),
        "reference_budget": float(reference),
    }


def explore_resenc_configs(
    spacing: Iterable[float],
    median_shape: Iterable[int],
    num_training_cases: int,
    input_channels: int,
    output_channels: int,
    memory_targets_gb: Iterable[float] = (8.0, 24.0, 40.0),
    deep_supervision: bool = True,
) -> list[dict]:
    rows = []
    for target in memory_targets_gb:
        rows.append(
            plan_like_resenc_config(
                spacing=spacing,
                median_shape=median_shape,
                num_training_cases=num_training_cases,
                input_channels=input_channels,
                output_channels=output_channels,
                memory_target_gb=float(target),
                deep_supervision=deep_supervision,
            )
        )
    return rows


def format_exploration_rows(rows: list[dict]) -> str:
    header = (
        "preset  gb    batch  patch_size         stages  features_per_stage"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        patch = "x".join(str(x) for x in row["patch_size"])
        feats = ",".join(str(x) for x in row["features_per_stage"])
        lines.append(
            f"{row['preset_hint']:<6} {row['memory_target_gb']:<5.1f} "
            f"{row['batch_size']:<6} {patch:<18} {row['n_stages']:<7} {feats}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sample_spacing = (1.5, 0.8, 0.8)
    sample_median_shape = (192, 256, 256)
    rows = explore_resenc_configs(
        spacing=sample_spacing,
        median_shape=sample_median_shape,
        num_training_cases=100,
        input_channels=1,
        output_channels=3,
        memory_targets_gb=(8.0, 24.0, 40.0),
    )
    print(format_exploration_rows(rows))
