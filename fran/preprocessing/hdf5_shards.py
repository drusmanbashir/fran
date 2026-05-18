import errno
import json
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re
from fran.preprocessing.preprocessor import Preprocessor
import numpy as np
import torch
from fran.preprocessing.helpers import (
    import_h5py,
    infer_indices_folder,
    sanitize_meta_for_monai,
)
from utilz.cprint import cprint
from utilz.fileio import maybe_makedirs

import h5py
from utilz.helpers import chunks
from utilz.stringz import int_to_str


def _normalize_src_dims(src_dims):
    dims = tuple(int(v) for v in src_dims)
    if len(dims) != 3 or any(v <= 0 for v in dims):
        raise ValueError(f"src_dims must be 3 positive ints, got {src_dims}")
    return dims


def generate_indices_fill_gaps(existing, n):
    """
    Return a set of size n containing:
    1. missing shard ids below the current max
    2. then new ids starting from max(existing) + 1
    """
    width = len(existing[0])
    existing_ints = {int(x) for x in existing}
    max_id = max(existing_ints) if existing_ints else 0

    missing = [i for i in range(1, max_id + 1) if i not in existing_ints]

    out = missing[:n]
    next_id = max_id + 1
    while len(out) < n:
        out.append(next_id)
        next_id += 1

    outs_str = []
    for outsi in out:
        outstr = int_to_str(outsi, width)
        outs_str.append(outstr)
    return outs_str


def _read_shard_case_ids(shard_fn):
    shard_fn = Path(shard_fn)
    try:
        h5py = import_h5py()
        with h5py.File(shard_fn, "r") as h5f:
            case_ids = list(h5f["cases"].keys())
        return {
            "shard_fn": str(shard_fn),
            "shard": shard_fn.name,
            "case_ids": case_ids,
            "error": None,
        }
    except Exception as e:
        return {
            "shard_fn": str(shard_fn),
            "shard": shard_fn.name,
            "case_ids": None,
            "error": str(e),
        }


def generate_indices(n, width=4):
    """
    Generate a list of n zero-padded string indices, e.g. ["0001", "0002", ...].
    """
    return [int_to_str(i, width) for i in range(1, n + 1)]


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
            return tuple(
                min(int(dim), int(chunk_dim))
                for dim, chunk_dim in zip(shape, conf[key])
            )

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
        image,
        lm,
        indiex,
        src_dims,
        compression,
        compression_opts,
    ):
        image = self._to_numpy_cpu(self._load_torch(image))
        lm = self._to_numpy_cpu(self._load_torch(lm))
        indices = self._load_torch(indiex)

        if not isinstance(indices, dict):
            raise ValueError(f"indices file must be a dict: {indiex}")
        if "lm_fg_indices" not in indices or "lm_bg_indices" not in indices:
            raise KeyError(
                f"indices dict missing lm_fg_indices/lm_bg_indices: {indiex}"
            )

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

        case_grp.attrs["image"] = str(image)
        case_grp.attrs["lm"] = str(lm)
        case_grp.attrs["indices"] = str(indiex)
        case_grp.attrs["image_shape"] = list(image.shape)
        case_grp.attrs["lm_shape"] = list(lm.shape)

        if "meta" not in indices:
            return
        meta = indices["meta"]
        if isinstance(meta, dict):
            meta = sanitize_meta_for_monai(dict(meta))
            case_grp.attrs["meta_json"] = json.dumps(meta, default=str)
            if "filename_or_obj" in meta and meta["filename_or_obj"] is not None:
                case_grp.attrs["source_meta_filename_or_obj"] = str(
                    meta["filename_or_obj"]
                )
            return
        if meta is not None:
            case_grp.attrs["meta_json"] = json.dumps(meta, default=str)

    def process_shard(
        self,
        shard_fn,
        shard_cases,
        src_dims,
        cases_per_shard,
        compression,
        compression_opts,
    ):
        shard_fn = Path(shard_fn)
        shard_tmp = shard_fn.with_suffix(".h5.tmp")
        src_dims = _normalize_src_dims(src_dims)
        h5py = import_h5py()
        case_ids_shard = [rec["case_id"] for rec in shard_cases]
        try:
            if shard_tmp.exists():
                shard_tmp.unlink()
            with h5py.File(shard_tmp, "w") as h5f:
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
                            image=rec["image"],
                            lm=rec["lm"],
                            indiex=rec["indices"],
                            src_dims=src_dims,
                            compression=compression,
                            compression_opts=compression_opts,
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"bad file case_id={rec['case_id']} shard_fn={shard_fn} image={rec['image']} lm={rec['lm']} indices={rec['indices']}"
                        ) from e
            shard_tmp.replace(shard_fn)
        except Exception:
            if shard_tmp.exists():
                shard_tmp.unlink()
            raise
        return {
            "shard": shard_fn.name,
            "case_ids": case_ids_shard,
        }


def _process_hdf5_shard_worker(kwargs):
    worker = HDF5ShardWorker()
    return worker.process_shard(**kwargs)


class HDF5ShardGenerator(Preprocessor):
    def __init__(
        self,
        project,
        plan,
        data_folder,
        output_folder,
        indices_folder=None,
        cases_per_shard=5,
        compression="gzip",
        compression_opts=1,
    ):
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self._indices_folder = (
            Path(indices_folder) if indices_folder is not None else None
        )
        src_dims = plan["src_dims"]
        self.plan = plan
        self.project = project

        self.src_dims = _normalize_src_dims(src_dims)
        self.cases_per_shard = cases_per_shard
        self.compression = compression
        self.compression_opts = compression_opts

    def setup(
        self,
    ):
        self.create_data_df()
        self.register_existing_cases()
        self.remove_completed_cases()
        self.shard_jobs = []  # T:self_ref|self.shard_jobs = []
        self.shard_paths = []  # T:self_ref|self.shard_paths = []

    def create_data_df(self):
        indices_folder = self.indices_subfolder
        if not indices_folder.exists():
            raise FileNotFoundError(
                f"indices folder not found for shard generation: {indices_folder}"
            )
        self.df = self._df_from_folder(indices_folder=indices_folder)
        assert len(self.df) > 0, "No valid case files found in {}".format(
            self.data_folder
        )
        self.case_ids = self.df["case_id"].tolist()
        self.df = self.df.map(lambda x: x.lower() if isinstance(x, str) else x)
        self.df["pt_processed"] = None
        self.df["hdf5_processed"] = None
        print("Total number of cases: ", len(self.df))
        self.df.drop(columns=["pt_processed"], inplace=True)

    @property
    def indices_subfolder(self):
        if self._indices_folder is not None:
            return self._indices_folder
        return infer_indices_folder(self.data_folder, self.plan)

    def _store_shard_ind(self, shard_fn):
        name = shard_fn.name
        pat = r"shard_(\d{4})\.h5"
        match = re.search(pat, name)
        if match:
            index = match.group(1)
        else:
            raise ValueError(
                f"Filename does not match expected pattern 'shard_{{index}}.h5': {name}"
            )
        self.shard_inds.append(index)

    def register_existing_cases(self):
        self.shard_inds = []
        shards = sorted(self.shards_folder.glob("shard_*.h5"))
        case_ids_done = []
        bad_names = []
        for shard_fn in shards:
            self._store_shard_ind(shard_fn)
            shard_info = _read_shard_case_ids(shard_fn)
            if shard_info["error"] is not None:
                bad_names.append(shard_info["shard"])
                continue
            case_ids_done.extend(shard_info["case_ids"])

        case_ids_done_unique = set(case_ids_done)
        if len(case_ids_done) != len(case_ids_done_unique):
            raise ValueError(
                f"Duplicate case IDs found across shards: {set(case_id for case_id in case_ids_done if case_ids_done.count(case_id) > 1)}"
            )

        self.df.loc[self.df["case_id"].isin(case_ids_done), "hdf5_processed"] = True
        if len(bad_names) > 0:
            raise RuntimeError(f"Failed to read the following shard files: {bad_names}")

    def _manifest_payload(self, shard_manifest):
        return {
            "format": "fran_hdf5_shards_v1",
            "src_dims": list(self.src_dims),
            "compression": self.compression,
            "compression_opts": self.compression_opts,
            "cases_per_shard": int(self.cases_per_shard),
            "num_cases": sum(len(shard["case_ids"]) for shard in shard_manifest),
            "num_shards": len(shard_manifest),
            "shards": shard_manifest,
        }

    @staticmethod
    def write_manifest(manifest_fn, manifest):
        cprint(f"Writing manifest with {manifest['num_shards']} shards and {manifest['num_cases']} cases to {manifest_fn}")
        manifest_tmp = manifest_fn.with_suffix(".json.tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest_tmp.replace(manifest_fn)

    def remove_completed_cases(self):
        hdf5_done = self.df["hdf5_processed"].eq(True)
        self.df_hdf5 = self.df[~hdf5_done].copy()
        print(
            "HDF5 cases remaining to process:",
            len(self.df_hdf5),
            "/",
            len(self.df),
        )

    def set_input_output_folders(self, data_folder, output_folder):
        pass

    def run(self, overwrite=False, num_processes=8):
        if not hasattr(self, "df"):
            print("No data frames have been created. Run setup")
            return 0
        if len(self.df) == 0:
            if getattr(self, "run_postprocess_if_empty", False):
                self.postprocess(overwrite=overwrite, num_processes=num_processes)
                return 0
            print("No data frames have been created. Run setup")
            return 0
        if overwrite == True:  # T:self_ref|if self.overwrite:
            df = self.df.copy()  # T:indent|    df = self.df.copy()
        else:
            df = (
                self.df[~self.df["hdf5_processed"].eq(True)].copy()
            )  # T:indent|    df = self.df[~self.df["hdf5_processed"].eq(True)].copy()
        case_records = df.to_dict(orient="records")  # T:self_ref|
        shard_groups = chunks(case_records, n_sized_chunks=self.cases_per_shard)
        if len(self.shard_inds) > 0:
            shard_indices = generate_indices_fill_gaps(
                existing=self.shard_inds, n=len(shard_groups)
            )
        else:
            shard_indices = generate_indices(n=len(shard_groups), width=4)

        self.shard_paths = []
        self.shard_jobs = []
        for shard_idx, shard_cases in zip(shard_indices, shard_groups):
            shard_fn = self.shards_folder / f"shard_{shard_idx}.h5"
            self.shard_paths.append(shard_fn)
            self.shard_jobs.append(
                    {
                        "shard_fn": shard_fn,
                        "shard_cases": shard_cases,
                        "src_dims": self.src_dims,
                        "cases_per_shard": self.cases_per_shard,
                    "compression": self.compression,
                    "compression_opts": self.compression_opts,
                }
            )
        self.process(num_processes=num_processes)
        self.postprocess(overwrite=overwrite, num_processes=num_processes)

    def process(self, num_processes=8):
        maybe_makedirs(self.shards_folder)
        completed_shards = []
        failed_shards = []
        max_workers = (
            max(1, min(int(num_processes), len(self.shard_jobs)))
            if self.shard_jobs
            else 0
        )
        if max_workers == 0:
            return sorted(self.shards_folder.glob("shard_*.h5"))
        if max_workers == 1:
            for job in self.shard_jobs:
                try:
                    shard_info = _process_hdf5_shard_worker(job)
                    completed_shards.append(shard_info)
                except Exception as e:
                    failed_shards.append(str(e))
                    continue
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
                    completed_shards.append(shard_info)
        self.shard_jobs = []
        if failed_shards:
            raise RuntimeError("HDF5 shard failures:\n" + "\n".join(failed_shards))
        print(f"Wrote {len(self.shard_paths)} HDF5 shards in {self.shards_folder}")
        return sorted(self.shards_folder.glob("shard_*.h5"))

    def postprocess(self, overwrite=False, num_processes=8):
        shard_manifest = []
        bad_names = []
        shards = sorted(self.shards_folder.glob("shard_*.h5"))
        for shard_fn in shards:
            shard_info = _read_shard_case_ids(shard_fn)
            if shard_info["error"] is not None:
                bad_names.append(shard_info["shard"])
                continue
            shard_manifest.append(
                {
                    "shard": shard_info["shard"],
                    "case_ids": shard_info["case_ids"],
                }
            )
        self.write_manifest(
            self.manifest_fn,
            self._manifest_payload(shard_manifest),
        )
        if len(bad_names) > 0:
            raise RuntimeError(f"Failed to read the following shard files: {bad_names}")

    @property
    def manifest_fn(self):
        manifest_fn = self.shards_folder / "manifest.json"
        return manifest_fn

    @property
    def shards_folder(self):
        src_tag = "_".join(str(v) for v in self.src_dims)
        shards_folder = self.output_folder / f"src_{src_tag}"
        return shards_folder


# %%
if __name__ == "__main__":
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- P = Project("nodes") <CR> <CR> <CR> <CR>
    import numpy as np
    from fran.configs.parser import ConfigMaker
    from fran.managers import Project
    from utilz.fileio import maybe_makedirs
    from utilz.helpers import chunks

# %%
    P = Project("totalseg")

    # P._create_plans_table()
    # P.add_data([DS.totalseg])
    C = ConfigMaker(P)
    C.setup(8)
    C.plans
    plan = C.configs["plan_train"]
    conf = C.configs

# %%

    plan = conf["plan_train"]
    plan["mode"]
    print(plan)
    print(P.global_properties)
    # add_plan_to_db(plan,"/r/datasets/preprocessed/totalseg/lbd/spc_100_100_100_plan5",P.db)
# %%
    overwrite = True
    src_dims = plan["src_dims"]
    cases_per_shard = 5
    overwrite = False
    hdf5_compression = "gzip"
    hdf5_compression_opts = 1
    num_processes = 8
    hdf5_output_folder = Path(
        "/r/datasets/preprocessed/totalseg/fixed_spacing/spc_100_100_100_rsc6fdbff67/hdf5_shards/"
    )
    output_folder = Path(
        "/s/fran_storage/datasets/preprocessed/totalseg/fixed_spacing/spc_100_100_100_rsc6fdbff67"
    )
# %%
# SECTION:-------------------- process--------------------------------------------------------------------------------------  # T:block_meta|FGBGIndicesResampleDataset.process
    G = HDF5ShardGenerator(
        project=P,
        plan=plan,
        data_folder=output_folder,  # T:self_ref|    pt_folder=self.output_folder,
        output_folder=hdf5_output_folder,  # T:self_ref|    shard_folder=self.hdf5_output_folder,
        cases_per_shard=cases_per_shard,
        compression=hdf5_compression,
        compression_opts=hdf5_compression_opts,
    )

# %%
    G.setup()
    G.run()
  
# %%
