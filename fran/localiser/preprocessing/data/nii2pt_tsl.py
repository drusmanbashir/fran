# %%
import ipdb
import ray
from fran.localiser.transforms import MultiRemapsTSL, TSLRegions
from fran.localiser.preprocessing.data.nii2pt import (
    PreprocessorNII2PT,
    _PreprocessorNII2PTWorkerBase,
)
from label_analysis.totalseg import TotalSegmenterLabels

tr = ipdb.set_trace


class _NII2PTTSLWorkerBase(_PreprocessorNII2PTWorkerBase):
    def worker_tfms_keys(self):
        return "L,E,O,Win,Remap,P1,P2"

    def create_transforms(self, device="cpu"):
        super().create_transforms(device=device)
        self.Remap = MultiRemapsTSL(lm_key=self.lm_key)
        self.transforms_dict["Remap"] = self.Remap


@ray.remote(num_cpus=1)
class TSLWorker(_NII2PTTSLWorkerBase):
    pass


class TSLWorkerLocal(_NII2PTTSLWorkerBase):
    pass


class PreprocessorNII2PTTSL(PreprocessorNII2PT):
    def __init__(self, data_folder, output_folder):
        super().__init__(data_folder, output_folder)
        self.actor_cls = TSLWorker
        self.local_worker_cls = TSLWorkerLocal

if __name__ == '__main__':
        tsl = TotalSegmenterLabels()
        excluded_tags = ["misc", "background"]
        df = tsl.df[~tsl.df.name_region.isin(excluded_tags)]
        regions = df.name_region.unique()

        region = regions[0]
        remap = tsl.create_remapping("label_full", region, as_list=True)
