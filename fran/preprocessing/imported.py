# %%
from fastcore.all import store_attr
from fran.data.collate import dict_list_collated
from fran.preprocessing.dataset import ImporterDataset
from fran.preprocessing.labelbounded import LabelBoundedDataGenerator

from torch.utils.data import DataLoader
from fran.utils.fileio import load_dict
from fran.utils.helpers import folder_name_from_list, resolve_device


import torch
from torch.utils.data import DataLoader

from fran.utils.helpers import folder_name_from_list, resolve_device


class LabelBoundedDataGeneratorImported(LabelBoundedDataGenerator):
    """
    works on fixed_spacing_folder, uses imported_folder which has lms with extra labels. Uses the extra labels to define image boundaries and crops each image accordingly. 
    A single image / lm  pair is generated per case.
    params: imported_labelsets: list of lists
    """

    _default = "project"

    def __init__(
        self,
        project,
        expand_by,
        spacing,
        lm_group,
        imported_folder,
        imported_labelsets,
        remapping,
        merge_imported=False,
        folder_suffix:str =None,
        fg_indices_exclude: list = None,
    ) -> None:

        super().__init__(project, expand_by, spacing, lm_group, fg_indices_exclude=fg_indices_exclude,folder_suffix=folder_suffix)
        store_attr('imported_folder,imported_labelsets,remapping,merge_imported')



    def set_folders_from_spacing(self, spacing):
        self.fixed_spacing_subfolder = folder_name_from_list(
            prefix="spc",
            parent_folder=self.fixed_spacing_folder,
            values_list=spacing,
        )
    def get_resampling_config(self, spacing):
        resamping_config_fn = self.project.fixed_spacing_folder / (
            "resampling_configs.json"
        )
        resampling_configs = load_dict(resamping_config_fn)
        for config in resampling_configs:
            if spacing == config["spacing"]:
                return config
        raise ValueError(
            "No resampling config found for this spacing: {0}. \nAll configs are:\n{1}".format(
                spacing, resampling_configs
            )
        )

    def setup(self, device="cpu", batch_size=4,num_workers=1, overwrite=True):
        device = resolve_device(device)
        print("Processing on ", device)
        self.register_existing_files()
        print("Overwrite:", overwrite)
        if overwrite == False:
            self.case_ids = self.remove_completed_cases()

        self.register_existing_files()
        self.ds = ImporterDataset(
            case_ids=self.case_ids,
            expand_by=self.expand_by,
            spacing=self.spacing,
            data_folder=self.fixed_spacing_subfolder,
            imported_folder=self.imported_folder,
            remapping_imported=self.remapping,
            merge_imported=self.merge_imported,
            device=device,
        )
        self.ds.setup()
        self.create_dl(batch_size=batch_size, num_workers=num_workers)


    def process_batch(self, batch):
        images, lms, fg_inds, bg_inds=(
            batch["image"],
            batch["lm"],
            batch["lm_fg_indices"],
            batch["lm_bg_indices"]
        )
        for (
            image,
            lm,
            fg_ind,
            bg_ind,
        ) in zip(
            images, lms, fg_inds, bg_inds,
        ):
            assert image.shape == lm.shape, "mismatch in shape".format(
                image.shape, lm.shape
            )
            assert image.dim() == 4, "images should be cxhxwxd"
            inds = {
                "lm_fg_indices": fg_ind,
                "lm_bg_indices": bg_ind,
                "meta": image.meta,
            }
            self.save_indices(inds, self.indices_subfolder)
            self.save_pt(image[0], "images")
            self.save_pt(lm[0], "lms")
            self.extract_image_props(image)

    def create_dl(self, batch_size=4, num_workers=4):
        self.dl = DataLoader(
            dataset=self.ds,
            num_workers=num_workers,
            collate_fn=dict_list_collated(
                keys=["image", "lm", "lm_imported","lm_fg_indices","lm_bg_indices", "bounding_box"]
            ),
            batch_size=batch_size,
        )

    def get_case_ids_lm_group(self, lm_group):
        dsrcs = self.global_properties[lm_group]["ds"]
        cids = []
        for dsrc in dsrcs:
            cds = self.df["case_id"][self.df["ds"] == dsrc].to_list()
            cids.extend(cds)
        return cids

    def create_properties_dict(self):
        resampled_dataset_properties = super().create_properties_dict()
        if self.remapping is None: labels = None
        else:
            labels = {
                k[0]: k[1] for k in self.remapping.items() if self.remapping[k[0]] != 0
            }
        additional_props = {
            "imported_folder": str(self.imported_folder),
            "imported_labels": labels,
            "merge_imported": self.merge_imported,
        }
        return resampled_dataset_properties | additional_props


class FixedSizeDataGeneratorImported(LabelBoundedDataGenerator):
    def __init__(self, project,  spatial_size, lm_group, folder_suffix: str, mask_label=None, fg_indices_exclude: list = None) -> None:
        super().__init__(project, expand_by, spacing, lm_group, folder_suffix, mask_label, fg_indices_exclude)
# %%

if __name__ == "__main__":
# %%
#SECTION:-------------------- SETUP--------------------------------------------------------------------------------------
    pass




