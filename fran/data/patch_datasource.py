# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import random

from utilz.helpers import find_matching_fn, info_from_filename, multiprocess_multiarg

from fran.data.datasource import Datasource
from fran.preprocessing.datasetanalyzers import case_analyzer_wrapper
from fran.preprocessing.helpers import import_h5py

# Assumes these exist in your codebase (as in your Datasource):
# - GetAttr
# - info_from_filename(fn.name, full_caseid=True) -> dict with "case_id"
# - find_matching_fn(img_fn, lms, tags=["all"]) -> (match, ...)
# - multiprocess_multiarg
# - case_analyzer_wrapper
# - headline

ImgLmPair = Tuple[Path, Path]


@dataclass(frozen=True)
class CasePatches:
    case_id: str
    patches: List[ImgLmPair]  # [(img_fn, lm_fn), ...]


class PatchDatasource(Datasource):
    """
    Patch-aware Datasource.

    Differences vs Datasource:
    - Allows multiple files per case_id.
    - Collates filepairs by case_id into self.case_patches.
    - For dataset property computation, samples ONE patch per case_id.
    """

    def __init__(
        self,
        folder: Union[str, Path],
        name: str = None,
        alias=None,
        bg_label: int = 0,
        test: bool = False,
        *,
        seed: int = 0,
        sample_strategy: str = "random",  # "random" | "first"
    ) -> None:
        self.seed = int(seed)
        self.sample_strategy = sample_strategy
        super().__init__(folder=folder, name=name, alias=alias, bg_label=bg_label, test=test)

    def integrity_check(self) -> None:
        images = list((self.folder / "images").glob("*"))
        lms = list((self.folder / "lms").glob("*"))
        assert len(images) == len(lms), (
            f"Different lengths of images {len(images)}, and lms {len(lms)}.\n"
            f"PatchDatasource still expects 1:1 image<->lm pairing at file level."
        )

        verified_pairs: List[ImgLmPair] = []
        for img_fn in images:
            lm_match = find_matching_fn(img_fn, lms, ["all"])[0]
            verified_pairs.append((img_fn, lm_match))

        # Collate by case_id (multiple patches per case_id allowed)
        by_case: Dict[str, List[ImgLmPair]] = {}
        for img_fn, lm_fn in verified_pairs:
            inf = info_from_filename(img_fn.name, full_caseid=True)
            case_id = inf["case_id"]
            by_case.setdefault(case_id, []).append((img_fn, lm_fn))

        self.verified_pairs = [[a, b] for (a, b) in verified_pairs]  # keep parent’s expected shape
        self.case_patches: List[CasePatches] = [
            CasePatches(case_id=k, patches=v) for k, v in sorted(by_case.items(), key=lambda x: x[0])
        ]

        print(
            f"{len(self.verified_pairs)} verified filepairs matched in folder {self.folder} "
            f"across {len(self.case_patches)} unique case_ids"
        )

    def _filter_unprocessed_cases(self) -> None:
        """
        Override to treat one H5 entry per case_id (not per patch file).
        """
        h5py = import_h5py()
        try:
            with h5py.File(self.h5_fname, "r") as h5f:
                prev_processed_cases = set(h5f.keys())
                print(f"Found {len(prev_processed_cases)} previously processed cases")
                self.raw_dataset_properties = []
                for case_id in prev_processed_cases:
                    case_data = {
                        "case_id": case_id,
                        "properties": {
                            "spacing": list(h5f[case_id].attrs["spacing"]),
                            "labels": list(h5f[case_id].attrs["labels"]),
                            "numel_fg": h5f[case_id].attrs["numel_fg"],
                            "mean_fg": h5f[case_id].attrs["mean_fg"],
                            "min_fg": h5f[case_id].attrs["min_fg"],
                            "max_fg": h5f[case_id].attrs["max_fg"],
                            "std_fg": h5f[case_id].attrs["std_fg"],
                        },
                    }
                    self.raw_dataset_properties.append(case_data)
        except FileNotFoundError:
            print(f"First time preprocessing dataset. Will create new file: {self.h5_fname}")
            self.raw_dataset_properties = []
            prev_processed_cases = set()

        all_case_ids = [cp.case_id for cp in self.case_patches]
        assert len(all_case_ids) == len(set(all_case_ids)), "Duplicate case_ids in case_patches (unexpected)."

        new_case_ids = set(all_case_ids).difference(prev_processed_cases)

        if not new_case_ids:
            print("No new cases found.")
            self.new_case_ids: List[str] = []
            self.new_case_patches: List[CasePatches] = []
        else:
            print(f"Found {len(new_case_ids)} new cases")
            self.new_case_ids = sorted(new_case_ids)
            self.new_case_patches = [cp for cp in self.case_patches if cp.case_id in new_case_ids]

    def _sample_one_patch_per_case(self, case_patches: Sequence[CasePatches]) -> List[ImgLmPair]:
        if self.sample_strategy not in {"random", "first"}:
            raise ValueError("sample_strategy must be 'random' or 'first'")

        rng = random.Random(self.seed)
        sampled: List[ImgLmPair] = []
        for cp in case_patches:
            if not cp.patches:
                continue
            if self.sample_strategy == "first":
                sampled.append(cp.patches[0])
            else:
                sampled.append(rng.choice(cp.patches))
        return sampled

    def process(
        self,
        return_voxels: bool = True,
        num_processes: int = 8,
        multiprocess: bool = True,
        debug: bool = False,
    ) -> None:
        """
        Computes dataset properties per case_id by sampling 1 patch per case_id.
        Stores one H5 dataset per case_id.
        """
        sampled_pairs = self._sample_one_patch_per_case(self.new_case_patches)

        # case_analyzer_wrapper expects [case_tuple, bg_label, return_voxels]
        # where case_tuple is [img_fn, lm_fn] (matching your base class)
        args_list = [[[img_fn, lm_fn], self.bg_label, return_voxels] for (img_fn, lm_fn) in sampled_pairs]

        self.outputs = multiprocess_multiarg(
            func=case_analyzer_wrapper,
            arguments=args_list,
            num_processes=num_processes,
            multiprocess=multiprocess,
            debug=debug,
            io=True,
        )

        # Force the case_id to be the collated case_id (not any patch-specific id)
        # Assumes outputs are in the same order as args_list.
        for (cp, output) in zip(self.new_case_patches, self.outputs):
            output["case"]["case_id"] = cp.case_id
            self.raw_dataset_properties.append(output["case"])

        self.dump_to_h5()

    def get_case_patch_filenames(self, case_id: str) -> List[ImgLmPair]:
        """
        Returns all (img,lm) pairs for a case_id.
        """
        for cp in self.case_patches:
            if cp.case_id == case_id:
                return list(cp.patches)
        return []

    @property
    def case_ids(self) -> List[str]:
        return [cp.case_id for cp in self.case_patches]

    def __len__(self) -> int:
        # one "case" per case_id
        return len(self.case_patches)

    @property
    def ds_type(self) -> str:        return "patch"


if __name__ == '__main__':
    bones_fldr = "/s/agent_rw/datasets/fully_annotated/ULS23_Radboudumc_Bone"
# %%
    ds = PatchDatasource(bones_fldr, "bones")
    # ds = Datasource(nodesthick_fldr, "nodesthick")
    ds.process()
    
