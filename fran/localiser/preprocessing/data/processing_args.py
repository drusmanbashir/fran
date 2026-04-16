import json
from pathlib import Path


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_processing_args(output_folder, args, filename="processing_args.json"):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    out_fn = output_folder / filename
    with open(out_fn, "w") as f:
        json.dump(json_ready(args), f, indent=2, sort_keys=True)
    return out_fn
