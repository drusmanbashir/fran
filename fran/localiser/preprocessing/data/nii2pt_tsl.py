# %%
from utilz.helpers import info_from_filename, set_autoreload

set_autoreload()
from fran.data.dataregistry import DS
from pathlib import Path
import ipdb
import ray
from fran.localiser.transforms import MultiRemapsTSL, TSLRegions
from fran.localiser.preprocessing.data.nii2pt import (
    PreprocessorNII2PT,
    _PreprocessorNII2PTWorkerBase,
)
from label_analysis.totalseg import TotalSegmenterLabels
from utilz.stringz import strip_extension
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 100)
tr = ipdb.set_trace


class _NII2PTTSLWorkerBase(_PreprocessorNII2PTWorkerBase):
    def __init__(
        self,
        output_folder,
        num_projections=2,
        device="cpu",
        exclude_regions=None,
        debug=False,
        merge_windows=False,
    ):
        self.exclude_regions = exclude_regions

        super().__init__(
            output_folder,
            num_projections=num_projections,
            device=device,
            debug=debug,
            merge_windows=merge_windows,
        )

    def worker_tfms_keys(self):
        return "L,E,O,Win,Remap,P1,P2"

    def create_transforms(self, device="cpu"):
        super().create_transforms(device=device)
        self.Remap = MultiRemapsTSL(lm_key=self.lm_key, exclude=self.exclude_regions)
        self.transforms_dict["Remap"] = self.Remap


@ray.remote(num_cpus=1)
class TSLWorker(_NII2PTTSLWorkerBase):
    pass


class TSLWorkerLocal(_NII2PTTSLWorkerBase):
    pass


class PreprocessorNII2PTTSL(PreprocessorNII2PT):
    def __init__(
        self,
        data_folder,
        output_folder,
        num_projections=2,
        exclude_regions=None,
        merge_windows=False,
    ):
        super().__init__(
            data_folder,
            output_folder,
            num_projections=num_projections,
            merge_windows=merge_windows,
        )
        self.exclude_regions = exclude_regions
        self.actor_cls = TSLWorker
        self.local_worker_cls = TSLWorkerLocal

    def build_worker_kwargs(self, device, debug):
        kwargs = super().build_worker_kwargs(device, debug)
        kwargs.update({"exclude_regions": self.exclude_regions})
        return kwargs

    def processing_args(self):
        args = super().processing_args()
        args.update({"exclude_regions": self.exclude_regions})
        return args

# %%
#SECTION:-------------------- setup--------------------------------------------------------------------------------------
if __name__ == '__main__':
        tsl = TotalSegmenterLabels()
        excluded_tags = ["misc", "background"]
        df = tsl.df[~tsl.df.name_region.isin(excluded_tags)]
        regions = df.name_region.unique()

        region = regions[0]
        remap = tsl.create_remapping("label_full", region, as_list=True)

        src_3d = DS["totalseg"].folder
        out_fldr = Path("/s/xnat_shadow/totalseg2d")
        out_2d = out_fldr / "pt"
        out_yolo = out_fldr / "jpg"
        P = PreprocessorNII2PTTSL(src_3d, out_2d, exclude_regions=["gut", "neck"])
        P.setup(device="cpu", num_processes=8, debug=True)
        P.process()
        len(P.df)
# %%

        existing_img = {p.name for p in (P.output_folder / "images").glob("*.pt")}
        existing_lm = {p.name for p in (P.output_folder / "lms").glob("*.pt")}
        existing_img
        existing_lm
        P.existing_output_fnames = list(existing_img.intersection(existing_lm))
        P.existing_output_fnames
        cids = [info_from_filename(fn, full_caseid=True)["case_id"] for fn in P.existing_output_fnames]
        case_ids


        case_ids = P.df.case_id.unique().tolist()
        cid = case_ids[0]
        cid in cids
        cids.count(cid)



# %%
        P.num_projections



        row = P.df.iloc[0]
        print(row)


# %%

        if getattr(P, "existing_output_fnames", None):
            n_before = len(P.df)
            keep_mask = P.df["image"].apply(
                lambda x: (
                    strip_extension(Path(x).name) + ".pt"
                    not in P.existing_output_fnames
                )
            )
            P.df = P.df[keep_mask]
            print("Image files remaining to process:", len(P.df), "/", n_before)
# %%
        aa = list(existing_img)
        aa[0]

# %%
        fn = "/s/xnat_shadow/totalseg2d/pt/lms/totalseg_s0009_a1.pt"
        lm = torch.load(fn, weights_only=False)
        lm.shape


# %%
