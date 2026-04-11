from pathlib import Path

import ray
import torch
from fran.localiser.preprocessing.data.nii2pt import (
    PreprocessorNII2PT,
    _PreprocessorNII2PTWorkerBase,
)
from utilz.stringz import strip_extension


class _PreprocessorNII2PTWorkerBase3D(_PreprocessorNII2PTWorkerBase):
    def worker_tfms_keys(self):
        return "L,E,O"

    def save_pt(self, tnsr, subfolder):
        fn = Path(tnsr.meta["filename_or_obj"])
        fn_name = strip_extension(fn.name) + ".pt"
        out_fn = self.output_folder / subfolder / fn_name
        torch.save(tnsr.contiguous(), out_fn)

    def image_suffixes(self):
        return [None]

    def _process_row(self, row):
        dici = {"image": row["image"], "lm": row["lm"]}
        dici = self.transforms(dici)
        self.save_pt(dici["image"], "images")
        self.save_pt(dici["lm"], "lms")
        return {"case_id": row["case_id"], "ok": True}


@ray.remote(num_cpus=1)
class PreprocessorNII2PTWorker3D(_PreprocessorNII2PTWorkerBase3D):
    pass


class PreprocessorNII2PTWorkerLocal3D(_PreprocessorNII2PTWorkerBase3D):
    pass


class PreprocessorNII2PT3D(PreprocessorNII2PT):
    def __init__(self, data_folder, output_folder):
        super().__init__(data_folder, output_folder)
        self.actor_cls = PreprocessorNII2PTWorker3D
        self.local_worker_cls = PreprocessorNII2PTWorkerLocal3D

    def register_existing_files(self):
        imgs = {p.name for p in self.output_fldr_imgs.glob("*.pt")}
        lms = {p.name for p in self.output_fldr_lms.glob("*.pt")}
        self.existing_output_fnames = imgs.intersection(lms)
        print("Output folder: ", self.output_folder)
        print(
            "Image files fully processed in a previous session: ",
            len(self.existing_output_fnames),
        )

    def remove_completed_cases(self):
        if not getattr(self, "existing_output_fnames", None):
            return
        n_before = len(self.df)
        keep_mask = self.df["image"].apply(
            lambda x: strip_extension(Path(x).name) + ".pt"
            not in self.existing_output_fnames
        )
        self.df = self.df[keep_mask]
        print("Image files remaining to process:", len(self.df), "/", n_before)
