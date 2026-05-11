import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fran.preprocessing.helpers import import_h5py


def _verify_manifest_shard(manifest_dir, shard_entry):
    manifest_dir = Path(manifest_dir)
    shard_name = shard_entry["shard"]
    shard_fn = manifest_dir / shard_name
    expected_case_ids = [str(case_id) for case_id in shard_entry["case_ids"]]
    out = {
        "shard": shard_name,
        "shard_fn": str(shard_fn),
        "expected_case_ids": expected_case_ids,
        "error": None,
    }
    try:
        h5py = import_h5py()
        with h5py.File(shard_fn, "r") as h5f:
            cases_grp = h5f["cases"]
            actual_case_ids = [str(case_id) for case_id in json.loads(h5f.attrs["case_ids_json"])]
            if set(actual_case_ids) != set(expected_case_ids):
                raise ValueError(
                    f"case_ids mismatch: manifest={expected_case_ids} shard={actual_case_ids}"
                )
            for case_id in expected_case_ids:
                case_grp = cases_grp[case_id]
                fg = case_grp["lm_fg_indices"][:]
                bg = case_grp["lm_bg_indices"][:]
                _ = tuple(int(v) for v in case_grp["lm"].shape)
                _ = fg.shape
                _ = bg.shape
        out["num_cases"] = len(expected_case_ids)
    except Exception as e:
        out["error"] = str(e)
    return out


def verify_shards_from_manifest(manifest_fn, max_workers=0):
    manifest_fn = Path(manifest_fn)
    manifest = json.loads(manifest_fn.read_text(encoding="utf-8"))
    manifest_dir = manifest_fn.parent
    shard_entries = manifest["shards"]
    max_workers = int(max_workers)

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(
                ex.map(lambda entry: _verify_manifest_shard(manifest_dir, entry), shard_entries)
            )
    else:
        results = [_verify_manifest_shard(manifest_dir, entry) for entry in shard_entries]

    bad_files = [result for result in results if result["error"] is not None]
    return {
        "manifest_fn": str(manifest_fn),
        "num_shards_manifest": len(shard_entries),
        "num_shards_checked": len(results),
        "num_bad_files": len(bad_files),
        "bad_file_names": [bad_file["shard_fn"] for bad_file in bad_files],
        "bad_files": bad_files,
    }


def delete_bad_files(file_names):
    for file_name in file_names:
        Path(file_name).unlink()
    return list(file_names)
