import json
from pathlib import Path

import numpy as np
import torch

from fran.preprocessing.helpers import import_h5py, sanitize_meta_for_monai
from utilz.fileio import maybe_makedirs, save_json


class HDF5ShardWriter:
    HDF5_SHARD_CHUNKS = {
        (192, 192, 128): {
            "image": (192, 192, 128),
            "lm": (192, 192, 128),
            "indices": (262144,),
        }
    }

    def __init__(
        self,
        output_folder,
        src_dims,
        cases_per_shard=5,
        max_shard_bytes=None,
        overwrite=False,
        compression="gzip",
        compression_opts=1,
    ):
        self.output_folder = Path(output_folder)
        self.src_dims = self._normalize_src_dims(src_dims)
        self.cases_per_shard = cases_per_shard
        self.max_shard_bytes = max_shard_bytes
        self.overwrite = overwrite
        self.compression = compression
        self.compression_opts = compression_opts

    @staticmethod
    def _normalize_src_dims(src_dims):
        dims = tuple(int(v) for v in src_dims)
        if len(dims) != 3 or any(v <= 0 for v in dims):
            raise ValueError(f"src_dims must be 3 positive ints, got {src_dims}")
        return dims

    def _hdf5_chunks_for(self, shape, key):
        shape = tuple(int(v) for v in shape)
        if self.src_dims in self.HDF5_SHARD_CHUNKS:
            conf = self.HDF5_SHARD_CHUNKS[self.src_dims]
        else:
            conf = {
                "image": self.src_dims,
                "lm": self.src_dims,
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

    def _create_index_dataset(self, case_grp, name, data, ds_kwargs):
        if int(data.shape[0]) > 0:
            case_grp.create_dataset(
                name,
                data=data,
                chunks=self._hdf5_chunks_for(data.shape, name),
                **ds_kwargs,
            )
            return
        case_grp.create_dataset(name, data=data, **ds_kwargs)

    def _write_case(self, h5f, case_id, image_pt, lm_pt, indices_pt):
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
        if self.compression is not None:
            ds_kwargs["compression"] = self.compression
            if self.compression_opts is not None:
                ds_kwargs["compression_opts"] = self.compression_opts
            ds_kwargs["shuffle"] = True

        cases_grp = h5f.require_group("cases")
        case_grp = cases_grp.create_group(case_id)
        case_grp.create_dataset(
            "image",
            data=image,
            chunks=self._hdf5_chunks_for(image.shape, "image"),
            **ds_kwargs,
        )
        case_grp.create_dataset(
            "lm",
            data=lm,
            chunks=self._hdf5_chunks_for(lm.shape, "lm"),
            **ds_kwargs,
        )
        self._create_index_dataset(case_grp, "lm_fg_indices", fg, ds_kwargs)
        self._create_index_dataset(case_grp, "lm_bg_indices", bg, ds_kwargs)

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

    def create_from_df(self, df):
        return self.create(case_ids=df["case_id"].tolist())

    def create(self, case_ids=None):
        images_folder = self.output_folder / "images"
        lms_folder = self.output_folder / "lms"
        indices_folder = self.output_folder / "indices"

        for folder in (images_folder, lms_folder, indices_folder):
            if not folder.exists():
                raise FileNotFoundError(f"Required folder missing: {folder}")

        src_tag = "_".join(str(v) for v in self.src_dims)
        shards_folder = self.output_folder / "hdf5_shards" / f"src_{src_tag}"
        manifest_fn = shards_folder / "manifest.json"
        maybe_makedirs([shards_folder])

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
                return existing_shards
        if len(requested_case_ids) == 0:
            raise ValueError(
                f"No shared requested case IDs found across images/lms/indices in {self.output_folder}"
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
        h5py = import_h5py()
        shard_paths = []
        for shard_idx, shard_cases in enumerate(shard_groups, start=len(shard_manifest)):
            shard_fn = shards_folder / f"shard_{shard_idx:04d}.h5"
            with h5py.File(shard_fn, "w") as h5f:
                case_ids_shard = [rec["case_id"] for rec in shard_cases]
                h5f.attrs["format"] = "fran_hdf5_shards_v1"
                h5f.attrs["src_dims"] = list(self.src_dims)
                h5f.attrs["cases_per_shard"] = int(self.cases_per_shard)
                h5f.attrs["case_ids_json"] = json.dumps(case_ids_shard)
                h5f.attrs["compression"] = "" if self.compression is None else str(self.compression)
                h5f.attrs["compression_opts"] = (
                    -1 if self.compression_opts is None else int(self.compression_opts)
                )

                for rec in shard_cases:
                    self._write_case(
                        h5f=h5f,
                        case_id=rec["case_id"],
                        image_pt=rec["image_pt"],
                        lm_pt=rec["lm_pt"],
                        indices_pt=rec["indices_pt"],
                    )

            shard_paths.append(shard_fn)
            shard_manifest.append({"shard": shard_fn.name, "case_ids": case_ids_shard})

        manifest = {
            "format": "fran_hdf5_shards_v1",
            "src_dims": list(self.src_dims),
            "compression": self.compression,
            "compression_opts": self.compression_opts,
            "cases_per_shard": int(self.cases_per_shard),
            "max_shard_bytes": self.max_shard_bytes,
            "num_cases": sum(len(shard["case_ids"]) for shard in shard_manifest),
            "num_shards": len(shard_manifest),
            "shards": shard_manifest,
        }
        save_json(manifest, manifest_fn)
        print(f"Wrote {len(shard_paths)} HDF5 shards in {shards_folder}")
        return sorted(shards_folder.glob("shard_*.h5"))
