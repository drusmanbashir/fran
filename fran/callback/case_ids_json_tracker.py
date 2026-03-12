import json
from pathlib import Path

import pandas as pd


def update_case_ids_json(
    *,
    trainer,
    pl_module,
    stage: str,
    epoch: int,
    df_long: pd.DataFrame,
    json_name: str = "case_ids_tracker.json",
) -> Path:
    project = getattr(pl_module, "project", None)
    log_folder = getattr(project, "log_folder", None)
    root = Path(log_folder) if log_folder else Path(getattr(trainer, "default_root_dir", "/tmp"))
    root.mkdir(parents=True, exist_ok=True)

    out_fn = root / json_name
    try:
        payload = json.loads(out_fn.read_text()) if out_fn.exists() else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    case_ids = sorted({str(cid) for cid in df_long["case_id"].dropna().tolist()})
    history = payload.get("history", [])
    history.append({"epoch": int(epoch), "stage": str(stage), "case_ids": case_ids})

    current = payload.get("current", {})
    current[str(stage)] = case_ids

    payload["current"] = current
    payload["history"] = history
    out_fn.write_text(json.dumps(payload, indent=2))
    return out_fn

