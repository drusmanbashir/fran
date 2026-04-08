# %%
import ipdb
import ray
import torch
from fran.localiser.preprocessing.data.nii2pt import (
    PreprocessorNII2PT,
    _PreprocessorNII2PTWorkerBase,
)
from label_analysis.totalseg import TotalSegmenterLabels
from monai.transforms import Transform
from monai.transforms.utility.dictionary import MapLabelValued

tr = ipdb.set_trace


class TSLRegions:
    def __init__(self):
        tsl = TotalSegmenterLabels()
        excluded_tags = ["misc", "background"]
        df = tsl.df[~tsl.df.name_region.isin(excluded_tags)]
        self.tsl = tsl
        self.regions = df.name_region.unique().tolist()
        self.data_yaml = "\n".join(self.data_yaml_lines())

    def data_yaml_lines(self):
        lines = ["names:"]
        for region in self.regions:
            lines.append("- " + region)
        lines.extend(
            [
                "nc: " + str(len(self.regions)),
                "test: ../test/images",
                "train: ../train/images",
                "val: ../valid/images",
                "",
            ]
        )
        return lines


class MultiRemapsTSLMonai(Transform):
    def __init__(self, lm_key):
        self.lm_key = lm_key
        self.tsl_regions = TSLRegions()
        self.regions = self.tsl_regions.regions
        self.remaps = {}
        for region in self.regions:
            remap = self.tsl_regions.tsl.create_remapping("label_full", region, as_list=True)
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


class MultiRemapsTSL(Transform):
    def __init__(self, lm_key):
        self.lm_key = lm_key
        self.tsl_regions = TSLRegions()
        self.regions = self.tsl_regions.regions
        self.luts = {}
        max_label = int(self.tsl_regions.tsl.df.label_full.max())
        for region in self.regions:
            remap = self.tsl_regions.tsl.create_remapping("label_full", region, as_list=True)
            lut = torch.zeros(max_label + 1, dtype=torch.float32)
            lut[torch.tensor(remap[0], dtype=torch.long)] = torch.tensor(
                remap[1], dtype=torch.float32
            )
            self.luts[region] = lut

    def __call__(self, data):
        lm = data[self.lm_key]
        lm_long = lm.long()
        remapped_lms = []
        for region in self.regions:
            lut = self.luts[region].to(lm.device)
            remapped_lms.append(lut[lm_long])
        data[self.lm_key] = torch.concat(remapped_lms, dim=0)
        return data


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
