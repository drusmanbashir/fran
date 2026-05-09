import json
import math
from pathlib import Path

import numpy as np
import torch


def to_json_safe(value):
    if isinstance(value, dict):
        return {str(key): to_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return to_json_safe(value.item())
        return to_json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, float):
        return None if math.isnan(value) else value
    return value


def write_jsonl_rows(output_path, rows):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(to_json_safe(row)) + "\n")


def append_jsonl_rows(output_path, rows):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(to_json_safe(row)) + "\n")
