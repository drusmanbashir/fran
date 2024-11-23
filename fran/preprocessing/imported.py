# %%
from fastcore.all import store_attr
from pathlib import Path
from label_analysis.totalseg import TotalSegmenterLabels
from fran.data.collate import dict_list_collated
from fran.preprocessing.dataset import ImporterDataset

from fran.preprocessing.labelbounded import LabelBoundedDataGenerator

from torch.utils.data import DataLoader
from fran.utils.config_parsers import ConfigMaker, parse_excel_plan
from fran.utils.fileio import load_dict
from fran.utils.helpers import find_matching_fn, folder_name_from_list, resolve_device


from torch.utils.data import DataLoader

from fran.utils.helpers import folder_name_from_list, resolve_device
from fran.utils.string import info_from_filename


class LabelBoundedDataGeneratorImported(LabelBoundedDataGenerator):
    """
    works on fixed_spacing_folder, uses imported_folder which has lms with extra labels. Uses the extra labels to define image boundaries and crops each image accordingly.
    A single image / lm  pair is generated per case.
    """

    _default = "project"

    def __init__(
        self,
        project,
        plan,
        imported_folder,
        merge_imported_labels=False,
        folder_suffix: str = None,
        data_folder=None,
        output_folder=None,
        mask_label=None,
        remapping: dict = None,
    ) -> None:

        self.imported_folder = Path(imported_folder)
        store_attr("merge_imported_labels")
        super().__init__(
            project=project,
            plan=plan,
            folder_suffix=folder_suffix,
            data_folder=data_folder,
            output_folder=output_folder,
            mask_label=mask_label,
            remapping=remapping,
        )


    def create_data_df(self):
        super().create_data_df()
        self.df.imported=None
        imported_fns = list(self.imported_folder.glob("*"))
        for fn in imported_fns:
            case_id = info_from_filename(fn.name,full_caseid=True)['case_id']
            # Update the imported column for the matching case_id
            mask = self.df['case_id'] == case_id
            if mask.any():
                self.df.loc[mask, 'imported'] = fn
            else:
                print(f"No matching case_id found for {case_id}")
        # Check for NaN values in imported column
        nan_mask = self.df['imported'].isna()
        if nan_mask.any():
            print(f"Found {nan_mask.sum()} cases without matching imported files:")
            print(self.df[nan_mask]['case_id'].tolist())
        else:
            print("All case_ids matched with imported filenames. Good to go!")
            # self.merge_imported_files_to_df()

    def create_ds(self,device="cpu"):
        self.ds = ImporterDataset(
            project=self.project,
            plan=self.plan,
            df=self.df,
            data_folder=self.data_folder,
            imported_folder=self.imported_folder,
            merge_imported_labels=self.merge_imported_labels,
            remapping_imported=self.remapping,
            device=device,
        )

    def process_batch(self, batch):
        images, lms, fg_inds, bg_inds = (
            batch["image"],
            batch["lm"],
            batch["lm_fg_indices"],
            batch["lm_bg_indices"],
        )
        for (
            image,
            lm,
            fg_ind,
            bg_ind,
        ) in zip(
            images,
            lms,
            fg_inds,
            bg_inds,
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
                keys=[
                    "image",
                    "lm",
                    "lm_imported",
                    "lm_fg_indices",
                    "lm_bg_indices",
                    "bounding_box",
                ]
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
        if self.remapping is None:
            labels = None
        else:
            labels = {
                k[0]: k[1] for k in self.remapping.items() if self.remapping[k[0]] != 0
            }
        additional_props = {
            "imported_folder": str(self.imported_folder),
            "imported_labels": labels,
            "merge_imported_labels": self.merge_imported_labels,
        }
        return resampled_dataset_properties | additional_props


# %%

if __name__ == "__main__":
# %%
# SECTION:-------------------- SETUP-------------------------------------------------------------------------------------- <CR>

    from fran.utils.common import *
    from fran.managers import Project

    P = Project(project_title="litsmc")
    spacing = [0.8, 0.8, 1.5]
    P.maybe_store_projectwide_properties()

    conf = ConfigMaker(P, raytune=False, configuration_filename=None).config
    plan = conf["plan"]


# %%
# SECTION:-------------------- Imported labels-------------------------------------------------------------------------------------- <CR> <CR> <CR> <CR>

    if not "labels_all" in P.global_properties.keys():
        P.set_lm_groups(plan["lm_groups"])
        P.maybe_store_projectwide_properties(overwrite=False)
    lm_group = P.global_properties["lm_group1"]
    TSL = TotalSegmenterLabels()
    imported_folder = "/s/fran_storage/predictions/totalseg/LITS-1088"
    imported_labelsets = lm_group["imported_labelsets"]
    imported_labelsets = [TSL.get_labels("liver",localiser=True)]
    remapping = TSL.create_remapping(
        imported_labelsets,
        [
            1,
        ]
        * len(imported_labelsets),
        localiser=True,
    )
    merge_imported_labels = True
# %%

    L = LabelBoundedDataGeneratorImported(
        project=P,
        data_folder="/s/xnat_shadow/crc/tensors/fixed_spacing",
        output_folder="/s/xnat_shadow/crc/tensors/ldb_plan3",
        plan=plan,
        imported_folder=imported_folder,
        merge_imported_labels=merge_imported_labels,
        remapping=remapping,
        folder_suffix=None,
    )
# %%
    L.setup()
    L.process()
# %%
#SECTION:-------------------- TROUBLESHOOTING--------------------------------------------------------------------------------------
    dici = L.ds[30]
    dici['lm'].meta
    dici= L.ds.data[0]
    lm_fn = dici['lm']
    lm = torch.load(lm_fn)
    lm.meta
    dici['lm'].meta
# %%
    dl = L.dl
    iteri = iter(dl)
    batch  = next(iteri)
    batch['image'][1].meta
# %%
    row = L.df.iloc[10]
    img_fn = row['image']
    find_matching_fn(img_fn,Path(L.imported_folder))


# %%
# L.ds.set_transforms("R,LS,LT,D,Re,E,Rz,M,B,A")
    dici=L.ds.data[45]
    if L.ds.merge_imported_labels == True:
        L.ds.set_transforms("R,LS,LT,D,E,Rz,M,B,A,Ind")
    else:
        L.ds.set_transforms("R,LS,LT,D,E,Rz,B,A,Ind")
    dici = L.ds.transforms_dict['R'](dici)
    dici = L.ds.transforms_dict['LS'](dici)
    dici = L.ds.transforms_dict['LT'](dici)
    dici = L.ds.transforms_dict['D'](dici)
    dici = L.ds.transforms_dict['E'](dici)
    dici = L.ds.transforms_dict['Rz'](dici)
# %%
    dici = L.ds.transforms_dict['M'](dici)
    dici = L.ds.transforms_dict['B'](dici)
    dici = L.ds.transforms_dict['A'](dici)
    dici = L.ds.transforms_dict['Ind'](dici)
# %%
    dici['lm'].meta


# %%
# %%
    """Create data dictionaries from DataFrame."""
    data = []
    for index in range(len(L.ds.df)):
        row = L.ds.df.iloc[index]
        ds = row.get("ds")
        remapping = L.ds._get_ds_remapping(ds)
        dici= L.ds._dici_from_df_row(row,remapping)
        data.append(dici)
# %%
    imported_folder=L.ds.imported_folder
    masks_folder = L.ds.data_folder / "lms"
    images_folder = L.ds.data_folder / "images"
    lm_fns = list(masks_folder.glob("*.pt"))
    img_fns = list(images_folder.glob("*.pt"))
    imported_files = list(imported_folder.glob("*"))
    data = []
    for cid in L.ds.case_ids:
        lm_fn = L.ds.case_id_file_match(cid, lm_fns)
        img_fn = L.ds.case_id_file_match(cid, img_fns)
        imported_fn = L.ds.case_id_file_match(cid, imported_files)
        dici = {
            "lm": lm_fn,
            "image": img_fn,
            "lm_imported": imported_fn,
            "remapping_imported": L.ds.remapping_imported,
        }
        data.append(dici)
# %%
    
    # L.df.to_csv("/s/xnat_shadow/crc/tensors/ldb/info.csv")

# %%




# %%
