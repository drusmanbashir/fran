from pathlib import Path

import numpy as np
import torch
from utilz.helpers import multiprocess_multiarg
from utilz.fileio import save_json


def _shape_array(shapes):
    if len(shapes) == 0:
        return np.empty((0, 0), dtype=np.float64)
    max_dims = max(len(s) for s in shapes)
    arr = np.full((len(shapes), max_dims), np.nan, dtype=np.float64)
    for i, shape in enumerate(shapes):
        arr[i, : len(shape)] = shape
    return arr


def _shape_stat(shape_arr: np.ndarray, reducer) -> list:
    if shape_arr.size == 0:
        return np.nan
    vals = reducer(shape_arr, axis=0)
    return [int(v) if np.isfinite(v) else np.nan for v in vals.tolist()]


def _tensorfile_stats_with_profile(filename):
    tnsr = torch.load(filename, weights_only=False)
    arr = tnsr.detach().cpu().numpy() if isinstance(tnsr, torch.Tensor) else np.asarray(tnsr)
    arr = arr.astype(np.float64, copy=False)
    return {
        "shape": [*arr.shape],
        "n_voxels": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "sum": float(np.sum(arr)),
        "sum_sq": float(np.sum(arr * arr)),
        "p01": float(np.percentile(arr, 1)),
        "p99": float(np.percentile(arr, 99)),
    }


def _tensorfile_histogram(filename, hist_range, bins):
    tnsr = torch.load(filename, weights_only=False)
    arr = tnsr.detach().cpu().numpy() if isinstance(tnsr, torch.Tensor) else np.asarray(tnsr)
    hist, _ = np.histogram(arr, bins=bins, range=hist_range)
    return hist.astype(np.int64)


def _quantile_from_hist(hist: np.ndarray, edges: np.ndarray, q: float) -> float:
    if hist.sum() == 0:
        return np.nan
    target = q * (hist.sum() - 1)
    cdf = np.cumsum(hist)
    idx = int(np.searchsorted(cdf, target, side="left"))
    idx = max(0, min(idx, len(hist) - 1))
    return float((edges[idx] + edges[idx + 1]) / 2.0)


def analyze_tensor_data_folder(
    data_folder,
    glob_pattern: str = "*.pt",
    num_processes: int = 16,
    debug: bool = False,
    recursive: bool = False,
    output_filename=None,
    include_per_file_stats: bool = False,
    histogram_bins: int = 8192,
):
    data_folder = Path(data_folder)
    file_iter = data_folder.rglob(glob_pattern) if recursive else data_folder.glob(glob_pattern)
    files = sorted(file_iter)
    if len(files) == 0:
        out = {
            "file_count": 0,
            "min_shape": np.nan,
            "median_shape": np.nan,
            "max_shape": np.nan,
            "intensity_profile": np.nan,
        }
        if include_per_file_stats:
            out["per_file_stats"] = []
        if output_filename:
            save_json(out, output_filename)
        return out

    args = [[fn] for fn in files]
    stats = multiprocess_multiarg(
        func=_tensorfile_stats_with_profile,
        arguments=args,
        num_processes=num_processes,
        debug=debug,
        io=True,
    )

    shape_arr = _shape_array([x["shape"] for x in stats])
    dataset_min = float(np.min([x["min"] for x in stats]))
    dataset_max = float(np.max([x["max"] for x in stats]))

    total_voxels = int(np.sum([x["n_voxels"] for x in stats]))
    total_sum = float(np.sum([x["sum"] for x in stats]))
    total_sum_sq = float(np.sum([x["sum_sq"] for x in stats]))
    dataset_mean = total_sum / total_voxels
    dataset_var = max(0.0, (total_sum_sq / total_voxels) - (dataset_mean ** 2))
    dataset_std = float(np.sqrt(dataset_var))

    if dataset_max == dataset_min:
        dataset_p01 = dataset_min
        dataset_median = dataset_min
        dataset_p99 = dataset_min
    else:
        hist_args = [[fn, (dataset_min, dataset_max), int(histogram_bins)] for fn in files]
        hists = multiprocess_multiarg(
            func=_tensorfile_histogram,
            arguments=hist_args,
            num_processes=num_processes,
            debug=debug,
            io=True,
        )
        hist_total = np.sum(np.stack(hists, axis=0), axis=0)
        edges = np.linspace(dataset_min, dataset_max, int(histogram_bins) + 1, dtype=np.float64)
        dataset_p01 = _quantile_from_hist(hist_total, edges, 0.01)
        dataset_median = _quantile_from_hist(hist_total, edges, 0.50)
        dataset_p99 = _quantile_from_hist(hist_total, edges, 0.99)

    prof = {
        "dataset_min": dataset_min,
        "dataset_max": dataset_max,
        "dataset_median": dataset_median,
        "dataset_mean": dataset_mean,
        "dataset_std": dataset_std,
        "dataset_p01": dataset_p01,
        "dataset_p99": dataset_p99,
        "total_voxels": total_voxels,
    }

    out = {
        "file_count": len(stats),
        "min_shape": _shape_stat(shape_arr, np.nanmin),
        "median_shape": _shape_stat(shape_arr, np.nanmedian),
        "max_shape": _shape_stat(shape_arr, np.nanmax),
        "intensity_profile": prof,
    }
    if include_per_file_stats:
        out["per_file_stats"] = stats
    if output_filename:
        save_json(out, output_filename)
    return out
