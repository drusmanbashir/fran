import ast
from typing import Any

import numpy as np
import pandas as pd


def is_excel_None(value) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"", "nan", "na", "null", "none"}
    return False


def to_native_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()

    if value is pd.NA:
        return None

    if isinstance(value, float):
        if pd.isna(value):
            return None
        if value.is_integer():
            return int(value)

    return value


def parse_excel_leaf(value: Any) -> Any:
    value = to_native_scalar(value)

    if is_excel_None(value):
        return None

    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()

        if lowered == "true":
            return True
        if lowered == "false":
            return False

        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            return stripped

        return to_native_scalar(parsed)

    return value


def normalize_tree(obj: Any, *, tuple_mode: str = "keep") -> Any:
    if isinstance(obj, pd.DataFrame):
        out = obj.copy()
        for col in out.columns:
            out[col] = out[col].map(parse_excel_leaf)
        return out.astype(object)

    if isinstance(obj, pd.Series):
        return {
            to_native_scalar(k): normalize_tree(v, tuple_mode=tuple_mode)
            for k, v in obj.items()
        }

    if isinstance(obj, dict):
        return {
            to_native_scalar(k): normalize_tree(v, tuple_mode=tuple_mode)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [normalize_tree(v, tuple_mode=tuple_mode) for v in obj]

    if isinstance(obj, tuple):
        vals = [normalize_tree(v, tuple_mode=tuple_mode) for v in obj]
        return vals if tuple_mode == "list" else tuple(vals)

    return parse_excel_leaf(obj)


def parse_excel_cell(cell_val):
    return parse_excel_leaf(cell_val)


def normalize_logging_payload(value):
    return normalize_tree(value, tuple_mode="list")


def make_patch_size(patch_dim0, patch_dim1):
    patch_dim0 = int(parse_excel_leaf(patch_dim0))
    patch_dim1 = int(parse_excel_leaf(patch_dim1))
    return [patch_dim0, patch_dim0, patch_dim1]


def make_src_dims_from_patch_size(patch_size):
    try:
        def _even(x):
            x = int(parse_excel_leaf(x))
            x = int(x * 1.1)
            return x if x % 2 == 0 else x + 1

        return [_even(dim) for dim in patch_size]
    except Exception:
        print(f"No valid patch_size: {patch_size}. Making dummy src_dims.")
        return [None, None, None]
