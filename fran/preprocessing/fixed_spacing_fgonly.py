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
from fran.preprocessing.preprocessor import get_tensor_stats, store_label_count
from fran.transforms.fg_indices import FgBgToIndicesSubsampled


class _NiftiSubsampledBgResamplerBase(_NiftiResamplerBase):
    """Nifti resampler variant that writes fg and optionally subsampled bg indices."""

    def create_transforms(self, device="cpu"):
        super().create_transforms(device=device)
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


@ray.remote(num_cpus=4)
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

    def postprocess_results(self, **process_kwargs):
        self._store_dataset_properties()
        generate_bboxes_from_lms_folder(
            self.output_folder / "lms",
            num_processes=getattr(self, "num_processes", 1),
        )
        store_label_count(
            self.output_folder, num_processes=getattr(self, "num_processes", 1)
        )
        create_dataset_stats_artifacts(
            lms_folder=self.output_folder / "lms",
            gif=self.store_gifs,
            label_stats=self.store_label_stats,
            gif_window=infer_dataset_stats_window(self.project),
        )

