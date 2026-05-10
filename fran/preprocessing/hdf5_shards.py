import json
import multiprocessing as mp
import os
import errno
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from fran.preprocessing.helpers import import_h5py, sanitize_meta_for_monai
from utilz.fileio import maybe_makedirs


def _sendfile_copy(src: "Path", dst: "Path") -> None:
    """
    Kernel zero-copy via os.sendfile() (Linux). Falls back to 256 MB buffer on non-Linux.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "sendfile"):
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            size = src.stat().st_size
            offset = 0
            while offset < size:
                try:
                    sent = os.sendfile(
                        fdst.fileno(), fsrc.fileno(), offset, size - offset
                    )
                except OSError as e:
                    if e.errno == errno.EINTR:
                        continue
                    raise
                if sent == 0:
                    break
                offset += sent
    else:
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=256 * 1024 * 1024)


def copy_folder_to_rapid_access(
    src_folder: "Path",
    dst_folder: "Path",
    glob: str = "*",
    overwrite: bool = False,
) -> "Path":
    """
    Copy files matching `glob` from src_folder to dst_folder using os.sendfile()
    zero-copy. Works for .pt files (images/, lms/, indices/) or .h5 shards.

    Cold storage originals are never touched.
    Per-file size verification — partial writes are detected and cleaned up.
    Resumable: skips already-copied files when overwrite=False.

    Returns dst_folder.
    """
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)

    if not src_folder.exists():
        raise FileNotFoundError(f"Source folder not found: {src_folder}")

    files = sorted(f for f in src_folder.glob(glob) if f.is_file())
    if not files:
        print(f"No files matching '{glob}' in {src_folder}, nothing to copy.")
        return dst_folder

    dst_folder.mkdir(parents=True, exist_ok=True)
    print(f"Copying {len(files)} files [{glob}]: {src_folder} → {dst_folder}")

    for i, src_file in enumerate(files, 1):
        dst_file = dst_folder / src_file.name
        if dst_file.exists() and not overwrite:
            print(f"  [{i}/{len(files)}] skip existing: {src_file.name}")
            continue
        size_mb = src_file.stat().st_size / 1e6
        print(f"  [{i}/{len(files)}] {src_file.name} ({size_mb:.1f} MB)")
        _sendfile_copy(src_file, dst_file)
        if dst_file.stat().st_size != src_file.stat().st_size:
            dst_file.unlink()
            raise RuntimeError(f"Size mismatch after copy: {src_file} → {dst_file}")

    print(f"Done: {dst_folder}")
    return dst_folder


class HDF5ShardWorker:
    HDF5_SHARD_CHUNKS = {
        (192, 192, 128): {
            "image": (192, 192, 128),
            "lm": (192, 192, 128),
            "indices": (262144,),
        }
    }

    @staticmethod
    def _normalize_src_dims(src_dims):
        dims = tuple(int(v) for v in src_dims)
        if len(dims) != 3 or any(v <= 0 for v in dims):
            raise ValueError(f"src_dims must be 3 positive ints, got {src_dims}")
        return dims

    def _hdf5_chunks_for(self, shape, key, src_dims):
        shape = tuple(int(v) for v in shape)
        if src_dims in self.HDF5_SHARD_CHUNKS:
            conf = self.HDF5_SHARD_CHUNKS[src_dims]
        else:
            conf = {
                "image": src_dims,
                "lm": src_dims,
                "indices": (262144,),
            }

        if key in ("image", "lm"):
            if len(shape) != 3:
                raise ValueError(f"{key} expected 3D shape, got {shape}")
            return tuple(min(int(dim), int(chunk_dim)) for dim, chunk_dim in zip(shape, conf[key]))

        if key in ("indices", "lm_fg_indices", "lm_bg_indices"):
            if len(shape) != 1:
                raise ValueError(f"{key} expected 1D shape, got {shape}")
            idx_chunk = int(conf["indices"][0])
            return (max(1, min(int(shape[0]), idx_chunk)),)

        raise KeyError(f"Unknown HDF5 chunk key: {key}")

    @staticmethod
    def _load_torch(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    @staticmethod
    def _to_numpy_cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _create_index_dataset(self, case_grp, name, data, ds_kwargs, src_dims):
        if int(data.shape[0]) > 0:
            case_grp.create_dataset(
                name,
                data=data,
                chunks=self._hdf5_chunks_for(data.shape, name, src_dims),
                **ds_kwargs,
            )
            return
        case_grp.create_dataset(name, data=data, **ds_kwargs)

    def _write_case(
        self,
        h5f,
        case_id,
        image_pt,
        lm_pt,
        indices_pt,
        src_dims,
        compression,
        compression_opts,
    ):
        image = self._to_numpy_cpu(self._load_torch(image_pt))
        lm = self._to_numpy_cpu(self._load_torch(lm_pt))
        indices = self._load_torch(indices_pt)

        if not isinstance(indices, dict):
            raise ValueError(f"indices file must be a dict: {indices_pt}")
        if "lm_fg_indices" not in indices or "lm_bg_indices" not in indices:
            raise KeyError(f"indices dict missing lm_fg_indices/lm_bg_indices: {indices_pt}")

        fg = self._to_numpy_cpu(indices["lm_fg_indices"]).reshape(-1)
        bg = self._to_numpy_cpu(indices["lm_bg_indices"]).reshape(-1)

        ds_kwargs = {}
        if compression is not None:
            ds_kwargs["compression"] = compression
            if compression_opts is not None:
                ds_kwargs["compression_opts"] = compression_opts
            ds_kwargs["shuffle"] = True

        cases_grp = h5f.require_group("cases")
        case_grp = cases_grp.create_group(case_id)
        case_grp.create_dataset(
            "image",
            data=image,
            chunks=self._hdf5_chunks_for(image.shape, "image", src_dims),
            **ds_kwargs,
        )
        case_grp.create_dataset(
            "lm",
            data=lm,
            chunks=self._hdf5_chunks_for(lm.shape, "lm", src_dims),
            **ds_kwargs,
        )
        self._create_index_dataset(case_grp, "lm_fg_indices", fg, ds_kwargs, src_dims)
        self._create_index_dataset(case_grp, "lm_bg_indices", bg, ds_kwargs, src_dims)

        case_grp.attrs["image_pt"] = str(image_pt)
        case_grp.attrs["lm_pt"] = str(lm_pt)
        case_grp.attrs["indices_pt"] = str(indices_pt)
        case_grp.attrs["image_shape"] = list(image.shape)
        case_grp.attrs["lm_shape"] = list(lm.shape)

        if "meta" not in indices:
            return
        meta = indices["meta"]
        if isinstance(meta, dict):
            meta = sanitize_meta_for_monai(dict(meta))
            case_grp.attrs["meta_json"] = json.dumps(meta, default=str)
            if "filename_or_obj" in meta and meta["filename_or_obj"] is not None:
                case_grp.attrs["source_meta_filename_or_obj"] = str(meta["filename_or_obj"])
            return
        if meta is not None:
            case_grp.attrs["meta_json"] = json.dumps(meta, default=str)

    def process_shard(
        self,
        shard_fn,
        shard_idx,
        shard_cases,
        src_dims,
        cases_per_shard,
        compression,
        compression_opts,
    ):
        shard_fn = Path(shard_fn)
        src_dims = self._normalize_src_dims(src_dims)
        h5py = import_h5py()
        case_ids_shard = [rec["case_id"] for rec in shard_cases]
        with h5py.File(shard_fn, "w") as h5f:
            h5f.attrs["format"] = "fran_hdf5_shards_v1"
            h5f.attrs["src_dims"] = list(src_dims)
            h5f.attrs["cases_per_shard"] = int(cases_per_shard)
            h5f.attrs["case_ids_json"] = json.dumps(case_ids_shard)
            h5f.attrs["compression"] = "" if compression is None else str(compression)
            h5f.attrs["compression_opts"] = (
                -1 if compression_opts is None else int(compression_opts)
            )
            for rec in shard_cases:
                try:
                    self._write_case(
                        h5f=h5f,
                        case_id=rec["case_id"],
                        image_pt=rec["image_pt"],
                        lm_pt=rec["lm_pt"],
                        indices_pt=rec["indices_pt"],
                        src_dims=src_dims,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"bad file case_id={rec['case_id']} shard_fn={shard_fn} image_pt={rec['image_pt']} lm_pt={rec['lm_pt']} indices_pt={rec['indices_pt']}"
                    ) from e
        return {
            "shard_idx": int(shard_idx),
            "shard": shard_fn.name,
            "case_ids": case_ids_shard,
        }


def _process_hdf5_shard_worker(kwargs):
    worker = HDF5ShardWorker()
    return worker.process_shard(**kwargs)


class HDF5ShardGenerator:
    def __init__(
        self,
        pt_folder,
        shard_folder,
        src_dims,
        cases_per_shard=5,
        max_shard_bytes=None,
        overwrite=False,
        compression="gzip",
        compression_opts=1,
    ):
        self.pt_folder = Path(pt_folder)
        self.shard_folder = Path(shard_folder)
        self.src_dims = HDF5ShardWorker._normalize_src_dims(src_dims)
        self.cases_per_shard = cases_per_shard
        self.max_shard_bytes = max_shard_bytes
        self.overwrite = overwrite
        self.compression = compression
        self.compression_opts = compression_opts

    def _build_shard_groups(self, case_records):
        assert self.cases_per_shard is None or self.max_shard_bytes is None
        if self.max_shard_bytes is not None:
            max_shard_bytes = int(self.max_shard_bytes)
            if max_shard_bytes <= 0:
                raise ValueError(f"max_shard_bytes must be > 0, got {max_shard_bytes}")
            groups = []
            current = []
            current_bytes = 0
            for rec in case_records:
                rec_bytes = rec["total_bytes"]
                if current and current_bytes + rec_bytes > max_shard_bytes:
                    groups.append(current)
                    current = [rec]
                    current_bytes = rec_bytes
                    continue
                current.append(rec)
                current_bytes += rec_bytes
            if current:
                groups.append(current)
            return groups

        cases_per_shard = int(self.cases_per_shard)
        if cases_per_shard <= 0:
            raise ValueError(f"cases_per_shard must be > 0, got {cases_per_shard}")

        return [
            case_records[idx : idx + cases_per_shard]
            for idx in range(0, len(case_records), cases_per_shard)
        ]

    @staticmethod
    def _manifest_entry(shard_idx, shard_fn, shard_cases):
        return {
            "shard_idx": int(shard_idx),
            "shard": Path(shard_fn).name,
            "case_ids": [rec["case_id"] for rec in shard_cases],
        }

    def _manifest_payload(self, shard_manifest, pending_shards=None):
        if pending_shards is None:
            pending_shards = []
        return {
            "format": "fran_hdf5_shards_v1",
            "src_dims": list(self.src_dims),
            "compression": self.compression,
            "compression_opts": self.compression_opts,
            "cases_per_shard": int(self.cases_per_shard),
            "max_shard_bytes": self.max_shard_bytes,
            "num_cases": sum(len(shard["case_ids"]) for shard in shard_manifest),
            "num_shards": len(shard_manifest),
            "num_pending_cases": sum(
                len(shard["case_ids"]) for shard in pending_shards
            ),
            "num_pending_shards": len(pending_shards),
            "shards": shard_manifest,
            "pending_shards": pending_shards,
        }

    @staticmethod
    def _write_manifest_atomic(manifest_fn, manifest):
        manifest_tmp = manifest_fn.with_suffix(".json.tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest_tmp.replace(manifest_fn)

    def _persist_manifest(self, shard_manifest, pending_shards):
        manifest = self._manifest_payload(
            shard_manifest=shard_manifest,
            pending_shards=pending_shards,
        )
        self._write_manifest_atomic(self.manifest_fn, manifest)

    def create_from_df(self, df, num_processes=8):
        return self.create(
            case_ids=df["case_id"].tolist(),
            num_processes=num_processes,
        )

    def _cleanup_uncommitted_shards(self, existing_shards, shard_manifest):
        committed_shards = {shard_meta["shard"] for shard_meta in shard_manifest}
        for shard_fn in existing_shards:
            if shard_fn.name not in committed_shards:
                shard_fn.unlink()

    def setup(self, case_ids=None, num_processes=8):
        images_folder = self.pt_folder / "images"
        lms_folder = self.pt_folder / "lms"
        indices_folder = self.pt_folder / "indices"

        for folder in (images_folder, lms_folder, indices_folder):
            if not folder.exists():
                raise FileNotFoundError(f"Required folder missing: {folder}")

        src_tag = "_".join(str(v) for v in self.src_dims)
        shards_folder = self.shard_folder / f"src_{src_tag}"
        manifest_fn = shards_folder / "manifest.json"
        maybe_makedirs([self.shard_folder, shards_folder])
        self.num_processes = max(1, int(num_processes))
        self.shards_folder = shards_folder
        self.manifest_fn = manifest_fn
        self.completed_shards = []
        self.pending_shards = []
        self.shard_jobs = []
        self.shard_paths = []

        existing_shards = sorted(shards_folder.glob("shard_*.h5"))
        shard_manifest = []
        existing_case_ids = set()
        if self.overwrite:
            for pth in existing_shards:
                pth.unlink()
            if manifest_fn.exists():
                manifest_fn.unlink()
        elif manifest_fn.exists():
            manifest = json.loads(manifest_fn.read_text())
            shard_manifest = manifest["shards"]
            self._cleanup_uncommitted_shards(existing_shards, shard_manifest)
            existing_shards = sorted(shards_folder.glob("shard_*.h5"))
            existing_case_ids = {
                case_id for shard_meta in shard_manifest for case_id in shard_meta["case_ids"]
            }
        elif existing_shards:
            raise FileExistsError(
                f"Existing shard files found in {shards_folder}. Set overwrite=True to regenerate."
            )

        image_case_ids = {pth.stem for pth in images_folder.glob("*.pt")}
        lm_case_ids = {pth.stem for pth in lms_folder.glob("*.pt")}
        indices_case_ids = {pth.stem for pth in indices_folder.glob("*.pt")}
        available_case_ids = image_case_ids & lm_case_ids & indices_case_ids
        if case_ids is None:
            requested_case_ids = sorted(available_case_ids)
        else:
            requested_case_ids = sorted(set(case_ids) & available_case_ids)

        if not self.overwrite:
            requested_case_ids = [
                case_id for case_id in requested_case_ids if case_id not in existing_case_ids
            ]
            if len(requested_case_ids) == 0:
                print(f"No missing HDF5 shard cases requested for {shards_folder}")
                self.completed_shards = list(shard_manifest)
                self.shard_paths = sorted(shards_folder.glob("shard_*.h5"))
                return self
        if len(requested_case_ids) == 0:
            raise ValueError(
                f"No shared requested case IDs found across images/lms/indices in {self.pt_folder}"
            )

        case_records = []
        for case_id in requested_case_ids:
            image_pt = images_folder / f"{case_id}.pt"
            lm_pt = lms_folder / f"{case_id}.pt"
            indices_pt = indices_folder / f"{case_id}.pt"
            total_bytes = image_pt.stat().st_size + lm_pt.stat().st_size + indices_pt.stat().st_size
            case_records.append(
                {
                    "case_id": case_id,
                    "image_pt": image_pt,
                    "lm_pt": lm_pt,
                    "indices_pt": indices_pt,
                    "total_bytes": int(total_bytes),
                }
            )

        shard_groups = self._build_shard_groups(case_records)
        shard_offset = len(shard_manifest)
        for shard_rel_idx, shard_cases in enumerate(shard_groups):
            shard_idx = shard_offset + shard_rel_idx
            shard_fn = shards_folder / f"shard_{shard_idx:04d}.h5"
            self.shard_paths.append(shard_fn)
            self.shard_jobs.append(
                {
                    "shard_fn": shard_fn,
                    "shard_idx": shard_idx,
                    "shard_cases": shard_cases,
                    "src_dims": self.src_dims,
                    "cases_per_shard": self.cases_per_shard,
                    "compression": self.compression,
                    "compression_opts": self.compression_opts,
                }
            )
            self.pending_shards.append(
                self._manifest_entry(
                    shard_idx=shard_idx,
                    shard_fn=shard_fn,
                    shard_cases=shard_cases,
                )
            )
        self.completed_shards = list(shard_manifest)
        self._persist_manifest(
            shard_manifest=self.completed_shards,
            pending_shards=self.pending_shards,
        )
        return self

    def _persist_run_progress(self, completed_by_idx):
        completed_manifest = list(self.completed_shards) + [
            completed_by_idx[idx] for idx in sorted(completed_by_idx)
        ]
        pending_manifest = [
            shard_meta
            for shard_meta in self.pending_shards
            if shard_meta["shard_idx"] not in completed_by_idx
        ]
        self._persist_manifest(
            shard_manifest=completed_manifest,
            pending_shards=pending_manifest,
        )

    def run(self):
        if not hasattr(self, "manifest_fn"):
            raise RuntimeError("Call setup() before run().")

        completed_shards = {}
        failed_shards = []
        max_workers = (
            max(1, min(int(self.num_processes), len(self.shard_jobs)))
            if self.shard_jobs
            else 0
        )
        if max_workers == 0:
            return sorted(self.shards_folder.glob("shard_*.h5"))
        if max_workers == 1:
            for job in self.shard_jobs:
                try:
                    shard_info = _process_hdf5_shard_worker(job)
                    completed_shards[shard_info["shard_idx"]] = shard_info
                except Exception as e:
                    failed_shards.append(str(e))
                    continue
                self._persist_run_progress(completed_shards)
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                futures = [
                    ex.submit(_process_hdf5_shard_worker, job)
                    for job in self.shard_jobs
                ]
                for fut in as_completed(futures):
                    try:
                        shard_info = fut.result()
                    except Exception as e:
                        failed_shards.append(str(e))
                        continue
                    completed_shards[shard_info["shard_idx"]] = shard_info
                    self._persist_run_progress(completed_shards)
        self.completed_shards = list(self.completed_shards) + [
            completed_shards[idx] for idx in sorted(completed_shards)
        ]
        self.pending_shards = []
        self.shard_jobs = []
        if failed_shards:
            raise RuntimeError("HDF5 shard failures:\n" + "\n".join(failed_shards))
        print(f"Wrote {len(self.shard_paths)} HDF5 shards in {self.shards_folder}")
        return sorted(self.shards_folder.glob("shard_*.h5"))

    def create(self, case_ids=None, num_processes=8):
        self.setup(case_ids=case_ids, num_processes=num_processes)
        return self.run()


HDF5ShardWriter = HDF5ShardGenerator
