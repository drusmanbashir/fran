from __future__ import annotations

import ray

from fran.preprocessing.fixed_spacing import (
    NiftiToTorchDataGenerator,
    _NiftiResamplerBase,
    generate_bboxes_from_lms_folder,
)
from fran.preprocessing.helpers import (
    create_dataset_stats_artifacts,
    infer_dataset_stats_window,
)
from fran.preprocessing.preprocessor import CPUS_PER_ACTOR, get_tensor_stats, store_label_count
from fran.transforms.fg_indices import FgBgToIndicesSubsampled


class _NiftiSubsampledBgResamplerBase(_NiftiResamplerBase):
    """Nifti resampler variant that writes fg and optionally subsampled bg indices."""

    def create_transforms(self):
        super().create_transforms()
        self.Indx = FgBgToIndicesSubsampled(
            keys=["lm"],
            ignore_labels=self.plan.get("fg_indices_exclude", []),
            subsample_bg=self.plan.get("subsample_bg", 5),
            image_key="image",
            image_threshold=-2600,
        )
        self.transforms_dict["Indx"] = self.Indx

    def _process_row(self, row):
        data = self._create_data_dict(row)
        data = self.apply_transforms(data)
        image = data["image"]
        lm = data["lm"]
        lm_fg_indices = data["lm_fg_indices"]
        lm_bg_indices = data["lm_bg_indices"]

        assert image.shape == lm.shape, "mismatch in shape"
        assert image.dim() == 4, "images should be cxhxwxd"

        inds = {
            "lm_fg_indices": lm_fg_indices,
            "lm_bg_indices": lm_bg_indices,
            "meta": image.meta,
        }
        self.save_indices(inds, self.indices_subfolder)
        self.save_pt(image[0], "images")
        self.save_pt(lm[0], "lms")
        return get_tensor_stats(image[0])

    @property
    def indices_subfolder(self):
        subsample_bg = self.plan.get("subsample_bg", 5)
        if subsample_bg is None:
            return self.output_folder / "indices"
        return self.output_folder / f"indices_bg_subsample_{subsample_bg}"


@ray.remote(num_cpus=CPUS_PER_ACTOR)
class NiftiSubsampledBgResampler(_NiftiSubsampledBgResamplerBase):
    pass


class NiftiSubsampledBgResamplerLocal(_NiftiSubsampledBgResamplerBase):
    pass


class NiftiToTorchSubsampledBgDataGenerator(NiftiToTorchDataGenerator):
    actor_cls = NiftiSubsampledBgResampler
    local_worker_cls = NiftiSubsampledBgResamplerLocal

    @property
    def indices_subfolder(self):
        subsample_bg = self.plan.get("subsample_bg", 5)
        if subsample_bg is None:
            return self.output_folder / "indices"
        return self.output_folder / f"indices_bg_subsample_{subsample_bg}"


