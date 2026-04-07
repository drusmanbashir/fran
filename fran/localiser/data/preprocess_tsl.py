# %%
import ipdb
import ray
import torch
from fran.localiser.data.preprocess import (
    PreprocessorNII2PT,
    _PreprocessorNII2PTWorkerBase,
)
from label_analysis.totalseg import TotalSegmenterLabels
from monai.transforms import Transform
from monai.transforms.utility.dictionary import MapLabelValued

tr = ipdb.set_trace


class MultiRemapsTSL(Transform):
    def __init__(self, lm_key):
        self.lm_key = lm_key
        tsl = TotalSegmenterLabels()
        excluded_tags = ["misc", "background"]
        df = tsl.df[~tsl.df.name_region.isin(excluded_tags)]
        self.regions = df.name_region.unique()
        self.remaps = {}
        for region in self.regions:
            remap = tsl.create_remapping("label_full", region, as_list=True)
            self.remaps[region] = MapLabelValued(
                keys=[self.lm_key], orig_labels=remap[0], target_labels=remap[1]
            )

    def __call__(self, data):
        lm = data[self.lm_key]
        remapped_lms = []
        for region in self.regions:
            remapped = self.remaps[region]({self.lm_key: lm.clone()})
            remapped_lms.append(remapped[self.lm_key])
        data[self.lm_key] = torch.concat(remapped_lms, dim=0)
        return data


class _TSLWorkerBase(_PreprocessorNII2PTWorkerBase):
    def worker_tfms_keys(self):
        return "L,E,O,N,Remap,P1,P2,P3"

    def create_transforms(self, device="cpu"):
        super().create_transforms(device=device)
        self.Remap = MultiRemapsTSL(lm_key=self.lm_key)
        self.transforms_dict["Remap"] = self.Remap


@ray.remote(num_cpus=1)
class TSLWorker(_TSLWorkerBase):
    pass


class TSLWorkerLocal(_TSLWorkerBase):
    pass


class PreprocessorNII2PTTSL(PreprocessorNII2PT):
    def __init__(self, data_folder, output_folder):
        super().__init__(data_folder, output_folder)
        self.actor_cls = TSLWorker
        self.local_worker_cls = TSLWorkerLocal


Preprocessor2DTSL = PreprocessorNII2PTTSL
