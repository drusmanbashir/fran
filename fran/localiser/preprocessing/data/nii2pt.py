# %%
from pathlib import Path

import ipdb
import numpy as np
import pandas as pd
import ray
import torch
from fran.transforms.imageio import LoadSITKd
from fran.transforms.spatialtransforms import Project2D
from monai.transforms import Compose, Transform

from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.spatial.dictionary import Orientationd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToDeviced
from utilz.fileio import maybe_makedirs
from utilz.helpers import create_df_from_folder
from utilz.stringz import strip_extension

tr = ipdb.set_trace


def tfms_from_dict(keys, transforms_dict):
    keys = keys.replace(" ", "").split(",")
    tfms = [transforms_dict[key] for key in keys]
    return Compose(tfms)


class WindowTensor3Channeld(Transform):
    def __init__(self, image_key ):
        self.windows = {
            "b": [-450.0, 1050.0],
            "c": [-1350.0, 150.0],
            "a": [-150.0, 250.0],
        }
        self.image_key = image_key

    def __call__(self, data):
        image = data[self.image_key]

        outs = []
        for L, U in self.windows.values():
            img = torch.clamp(image, L, U)
            img = (img - L) / (U - L)
            outs.append(img)

        data[self.image_key] = torch.cat(outs, dim=0)
        return data


class _PreprocessorNII2PTWorkerBase:
    def __init__(self, output_folder, device="cpu", debug=False):
        self.output_folder = Path(output_folder)
        self.device = device
        self.debug = debug
        self.image_key = "image"
        self.lm_key = "lm"
        self.tfms_keys = self.worker_tfms_keys()
        self.create_transforms(device=device)
        self.transforms = tfms_from_dict(self.tfms_keys, self.transforms_dict)

    def worker_tfms_keys(self):
        return "L,E,O,Win,P1,P2"

    def create_transforms(self, device="cpu"):
        self.L = LoadSITKd(keys=[self.image_key, self.lm_key])
        self.E = EnsureChannelFirstd(
            keys=[self.image_key, self.lm_key], channel_dim="no_channel"
        )
        self.O = Orientationd(keys=[self.image_key, self.lm_key], axcodes="RAS")
        self.N = NormalizeIntensityd(keys=[self.image_key])
        self.Dev = ToDeviced(keys=[self.image_key, self.lm_key], device=device)
        self.P1 = Project2D(
            keys=[self.lm_key, self.image_key],
            operations=["mean", "sum"],
            dim=1,
            output_keys=["lm1", "image1"],
        )
        self.P2 = Project2D(
            keys=[self.lm_key, self.image_key],
            operations=["sum", "mean"],
            dim=2,
            output_keys=["lm2", "image2"],
        )
        self.P3 = Project2D(
            keys=[self.lm_key, self.image_key],
            operations=["sum", "mean"],
            dim=3,
            output_keys=["lm3", "image3"],
        )
        self.Win = WindowTensor3Channeld(image_key=self.image_key)
        # Axial XY projection is kept for reference but not used:
        # it is a top-down/bottom-up view after RAS orientation and is not informative for YOLO.
        self.transforms_dict = {
            "L": self.L,
            "E": self.E,
            "O": self.O,
            "N": self.N,
            "Dev": self.Dev,
            "P1": self.P1,
            "P2": self.P2,
            "P3": self.P3,
            "Win": self.Win,

        }

    def save_pt(self, tnsr, subfolder, suffix):
        fn = Path(tnsr.meta["filename_or_obj"])
        fn_name = strip_extension(fn.name) + "_" + str(suffix) + ".pt"
        out_fn = self.output_folder / subfolder / fn_name
        torch.save(tnsr.contiguous(), out_fn)

    def image_suffixes(self):
        if "Win" in self.tfms_keys:
            suffixes = []
            for window in self.Win.windows.keys():
                for projection in [1, 2]:
                    suffixes.append(f"{window}{projection}")
            return suffixes
        return [1, 2]

    def _process_row(self, row):
        dici = {"image": row["image"], "lm": row["lm"]}
        dici = self.transforms(dici)
        for projection in [1, 2]:
            image = dici["image" + str(projection)]
            lm = dici["lm" + str(projection)]
            if "Win" in self.tfms_keys:
                for window_ind, window in enumerate(self.Win.windows.keys()):
                    suffix = f"{window}{projection}"
                    self.save_pt(image[[window_ind]], "images", suffix)
                    self.save_pt(lm, "lms", suffix)
            else:
                self.save_pt(image, "images", projection)
                self.save_pt(lm, "lms", projection)
        return {"case_id": row["case_id"], "ok": True}

    def process(self, df):
        outputs = []
        for ind in range(len(df)):
            row = df.iloc[ind]
            outputs.append(self._process_row(row))
        return outputs


@ray.remote(num_cpus=1)
class PreprocessorNII2PTWorker(_PreprocessorNII2PTWorkerBase):
    pass


class PreprocessorNII2PTWorkerLocal(_PreprocessorNII2PTWorkerBase):
    pass


class PreprocessorNII2PT:
    def __init__(self, data_folder, output_folder):
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.fldr_imgs = self.data_folder / "images"
        self.fldr_lms = self.data_folder / "lms"
        self.output_fldr_imgs = self.output_folder / "images"
        self.output_fldr_lms = self.output_folder / "lms"
        self.actor_cls = PreprocessorNII2PTWorker
        self.local_worker_cls = PreprocessorNII2PTWorkerLocal
    def _df_from_folder(self):
        return create_df_from_folder(self.data_folder)

    def create_data_df(self):
        self.df = self._df_from_folder()
        assert len(self.df) > 0, "No valid case files found in {}".format(
            self.data_folder
        )
        self.case_ids = self.df["case_id"].tolist()
        print("Total number of cases: ", len(self.df))

    def create_output_folders(self):
        maybe_makedirs([self.output_fldr_imgs, self.output_fldr_lms])

    def register_existing_files(self):
        existing = []
        for suffix in self.local_worker_cls(self.output_folder).image_suffixes():
            suffix = str(suffix)
            imgs = {p.name for p in self.output_fldr_imgs.glob("*_" + suffix + ".pt")}
            lms = {p.name for p in self.output_fldr_lms.glob("*_" + suffix + ".pt")}
            existing.append(imgs.intersection(lms))
        self.existing_output_fnames = set.intersection(*existing) if existing else set()
        print("Output folder: ", self.output_folder)
        print(
            "Image files fully processed in a previous session: ",
            len(self.existing_output_fnames),
        )

    def remove_completed_cases(self):
        if not getattr(self, "existing_output_fnames", None):
            return
        n_before = len(self.df)
        suffix = self.local_worker_cls(self.output_folder).image_suffixes()[0]
        keep_mask = self.df["image"].apply(
            lambda x: (
                strip_extension(Path(x).name) + "_" + str(suffix) + ".pt"
                not in self.existing_output_fnames
            )
        )
        self.df = self.df[keep_mask]
        print("Image files remaining to process:", len(self.df), "/", n_before)

    def should_use_ray(self):
        debug = getattr(self, "debug", False)
        return (self.num_processes > 1) and (debug == False)

    def build_worker_kwargs(self, device, debug):
        return {
            "output_folder": self.output_folder,
            "device": device,
            "debug": debug,
        }

    def setup(self, overwrite=False, num_processes=8, device="cpu", debug=False):
        self.create_output_folders()
        self.num_processes = max(1, int(num_processes))
        self.debug = debug
        self.create_data_df()
        self.register_existing_files()
        print("Overwrite:", overwrite)
        if overwrite == False:
            self.remove_completed_cases()
        self.use_ray = self.should_use_ray()
        worker_kwargs = self.build_worker_kwargs(device=device, debug=debug)
        if self.use_ray:
            n = min(len(self.df), self.num_processes)
            self.mini_dfs = np.array_split(self.df, n)
            self.actors = [self.actor_cls.remote(**worker_kwargs) for _ in range(n)]
        else:
            self.mini_dfs = [self.df]
            self.local_worker = self.local_worker_cls(**worker_kwargs)

    def process(self):
        if len(self.df) == 0:
            return pd.DataFrame([])
        if self.use_ray:
            results = ray.get(
                [
                    actor.process.remote(mini_df)
                    for actor, mini_df in zip(self.actors, self.mini_dfs)
                ]
            )
        else:
            results = [self.local_worker.process(self.mini_dfs[0])]
        self.results = results
        self.results_df = pd.DataFrame(
            [item for sublist in self.results for item in sublist]
        )
        return self.results_df
