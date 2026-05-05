from __future__ import annotations

import json
import shutil
from pathlib import Path

from tqdm.auto import tqdm as pbar
from utilz.cprint import cprint

from .training import (
    COMMON_PATHS,
    DataManagerBaseline,
    DataManagerDual,
    DataManagerLBD,
    DataManagerPatch,
    DataManagerRBD,
    DataManagerSource,
    DataManagerWhole,
)


class DualSSDStagingMixin:
    def create_staged_data_dicts(self, cases):
        data = super().create_staged_data_dicts(cases)
        if len(data) == 0:
            return data
        if self.has_hdf5_shard_manifest():
            return self.copy_hdf5_shards_to_rapid_access_folder2(data)
        required_keys = {"image", "lm", "indices"}
        if required_keys.issubset(data[0].keys()):
            return self.copy_data_dicts_to_rapid_access_folder2(data)
        cprint(
            "Skipping dual_ssd staging; data dicts do not contain image/lm/indices paths.",
            color="yellow",
        )
        return data

    def _rapid_access_folder2_data_folder(self):
        src_root = Path(COMMON_PATHS["rapid_access_folder"])
        dst_root = Path(COMMON_PATHS["rapid_access_folder2"])
        try:
            data_rel = self.data_folder.relative_to(src_root)
        except ValueError:
            data_rel = Path(self.data_folder.name)
        return dst_root / data_rel

    def copy_hdf5_shards_to_rapid_access_folder2(self, data):
        cprint(
            "Copying half of the HDF5 shards to rapid_access_folder2",
            color="green",
        )
        src_root = Path(COMMON_PATHS["rapid_access_folder"])
        dst_root = Path(COMMON_PATHS["rapid_access_folder2"])
        staged_data_folder = self._rapid_access_folder2_data_folder()
        staged_manifest_fn = staged_data_folder / self.hdf5_shard_manifest_path.relative_to(
            self.data_folder
        )

        manifest = json.loads(self.hdf5_shard_manifest_path.read_text())
        manifest_parent = self.hdf5_shard_manifest_path.parent
        for i, shard_info in enumerate(pbar(manifest["shards"])):
            shard_path = Path(shard_info["shard"])
            if not shard_path.is_absolute():
                shard_path = manifest_parent / shard_path
            if i % 2 == 1:
                try:
                    shard_rel = shard_path.relative_to(src_root)
                except ValueError:
                    shard_rel = Path(shard_path.name)
                src_shard_path = shard_path
                shard_path = dst_root / shard_rel
                if (
                    not shard_path.exists()
                    or shard_path.stat().st_size != src_shard_path.stat().st_size
                ):
                    shard_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_shard_path, shard_path)
            shard_info["shard"] = str(shard_path)

        staged_manifest_fn.parent.mkdir(parents=True, exist_ok=True)
        staged_manifest_fn.write_text(json.dumps(manifest, indent=2))

        staged = []
        for dici in data:
            out = dict(dici)
            out["data_folder"] = str(staged_data_folder)
            staged.append(out)
        return staged

    def copy_data_dicts_to_rapid_access_folder2(self, data):
        cprint("Copying half of the data to rapid_access_folder2", color="green")
        src_root = Path(COMMON_PATHS["rapid_access_folder"])
        dst_root = Path(COMMON_PATHS["rapid_access_folder2"])
        keys = ("image", "lm", "indices")
        copied = list(data)
        for i, dici in enumerate(pbar(data)):
            if i % 2 == 0:
                continue
            out = dict(dici)
            for key in keys:
                out[key] = self._copy_value_to_rapid_access_folder2(
                    out[key], src_root, dst_root
                )
            copied[i] = out
        return copied

    def _copy_value_to_rapid_access_folder2(self, value, src_root, dst_root):
        src = Path(value)
        dst = dst_root / src.relative_to(src_root)
        if dst.exists():
            return str(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cprint(f"Copying {src} to {dst}", color="green")
        shutil.copy2(src, dst)
        return str(dst)


class DataManagerSourceDualSSD(DualSSDStagingMixin, DataManagerSource):
    pass


class DataManagerWholeDualSSD(DualSSDStagingMixin, DataManagerWhole):
    pass


class DataManagerPatchDualSSD(DualSSDStagingMixin, DataManagerPatch):
    pass


class DataManagerLBDDualSSD(DualSSDStagingMixin, DataManagerLBD):
    pass


class DataManagerRBDDualSSD(DualSSDStagingMixin, DataManagerRBD):
    pass


class DataManagerBaselineDualSSD(DualSSDStagingMixin, DataManagerBaseline):
    pass


DUAL_SSD_MANAGER_CLASSES = {
    DataManagerSource: DataManagerSourceDualSSD,
    DataManagerWhole: DataManagerWholeDualSSD,
    DataManagerPatch: DataManagerPatchDualSSD,
    DataManagerLBD: DataManagerLBDDualSSD,
    DataManagerRBD: DataManagerRBDDualSSD,
    DataManagerBaseline: DataManagerBaselineDualSSD,
}


def dual_ssd_manager_class(manager_class: type) -> type:
    if issubclass(manager_class, DualSSDStagingMixin):
        return manager_class
    return DUAL_SSD_MANAGER_CLASSES[manager_class]


class DataManagerDualSSD(DataManagerDual):
    def _build_managers(self):
        inf_tr, inf_val = self.infer_manager_classes(self.configs)
        cls_tr = dual_ssd_manager_class(self.manager_class_train or inf_tr)
        cls_val = dual_ssd_manager_class(self.manager_class_valid or inf_val)

        self.train_manager = cls_tr(
            project=self.project,
            configs=self.configs,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            split="train",
            device=self.device,
            ds_type=self.ds_type,
            keys=self.keys_tr,
            data_folder=self.data_folder,
            debug=self.debug,
        )
        self.valid_manager = cls_val(
            project=self.project,
            configs=self.configs,
            batch_size=self.batch_size,
            cache_rate=self.cache_rate,
            split="valid",
            device=self.device,
            ds_type=self.ds_type,
            keys=self.keys_val,
            data_folder=self.data_folder,
            val_sampling=self.val_sampling,
            debug=self.debug,
        )


__all__ = [
    "DataManagerBaselineDualSSD",
    "DataManagerDualSSD",
    "DataManagerLBDDualSSD",
    "DataManagerPatchDualSSD",
    "DataManagerRBDDualSSD",
    "DataManagerSourceDualSSD",
    "DataManagerWholeDualSSD",
    "DualSSDStagingMixin",
    "dual_ssd_manager_class",
]
