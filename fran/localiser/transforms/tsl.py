# %%
import torch
from label_analysis.totalseg import TotalSegmenterLabels
from monai.transforms import Transform
from monai.transforms.utility.dictionary import MapLabelValued


class TSLRegions:
    def __init__(self, exclude:list[str]=None):
        tsl = TotalSegmenterLabels()
        excluded_tags = ["misc", "background"]
        if exclude is not None:
            excluded_tags.extend(exclude)
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
    def __init__(self, lm_key, exclude:list[str]=None):
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
    def __init__(self, lm_key, exclude:list[str]=None):
        self.lm_key = lm_key
        self.tsl_regions = TSLRegions(exclude=exclude)
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

# %%
if __name__ == '__main__':
    T = TSLRegions()
    print(T.data_yaml)
    T.tsl.create_remapping("label_full", "neck", as_list=True)
    T.regions

    M= MultiRemapsTSL("lm", ["gut"])
    M.regions

