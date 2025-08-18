# %%
from fastcore.basics import listify
from monai.transforms.utils import is_positive
import pandas as pd
from monai.transforms import Compose
from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToDeviced
from pathlib import Path

from fran.managers.db import add_plan_to_db, find_matching_plan
from fran.transforms.imageio import LoadTorchd
from fran.transforms.misc_transforms import FgBgToIndicesd2
from fran.utils.config_parsers import ConfigMaker, is_excel_None
from utilz.string import info_from_filename
from pathlib import Path

from fastcore.basics import GetAttr, store_attr

from fran.preprocessing.preprocessor import Preprocessor, generate_bboxes_from_lms_folder
from utilz.fileio import *
from utilz.helpers import *
from utilz.imageviewers import *

MIN_SIZE = 32  # min size in a single dimension of any image


class LabelBoundedDataGenerator(Preprocessor, GetAttr):
#CODE: Preprocessor and downstream classes need thorough review and re-writing. E.g., where does expand_by go, in preprocessor or labelboudned?
    """
    Label-bounded data generator for preprocessing medical imaging data with automatic folder management.
    
    This class generates preprocessed image and label data by applying transformations and cropping
    based on label boundaries. It automatically manages input and output folder structures based
    on spacing parameters and configuration.
    
    **Algorithm for Setting Input/Output Folders:**
    
    1. **Input Data Folder Logic:**
       - If `data_folder` is provided: Uses the specified path directly
       - If `data_folder` is None: Automatically generates path using spacing parameters
         - Creates folder name: "{fixed_spacing_folder}/spc_{spacing[0]}_{spacing[1]}_{spacing[2]}"
         - Example: "/project/data/fixed_spacing/spc_1.5_1.5_1.5"
    
    2. **Output Folder Logic:**
       - If `output_folder` is provided: Uses the specified path directly  
       - If `output_folder` is None: Automatically generates path under project's lbd_folder
         - Base path: "{project.lbd_folder}/spc_{spacing[0]}_{spacing[1]}_{spacing[2]}"
         - If `folder_suffix` provided: Appends suffix to folder name
         - Example: "/project/lbd/spc_1.5_1.5_1.5_plan9" (with folder_suffix="plan9")
    
    3. **Folder Structure Created:**
       - `{output_folder}/images/` - Processed image files (.pt format)
       - `{output_folder}/lms/` - Processed label/mask files (.pt format)  
       - `{output_folder}/indices/` - Foreground/background indices (.pt format)
       - `{output_folder}/indices_fg_exclude_{labels}/` - When fg_indices_exclude is specified
    
    The automatic folder naming ensures consistent organization based on processing parameters
    and prevents conflicts between different spacing configurations or processing plans.
    """

    _default = "project"

    def __init__(
        self,
        project,
        plan,
        plan_name: str = None,
        data_folder=None,
        output_folder=None,
        mask_label=None,
        remapping: dict = None,
        expand_by=0,
    ) -> None:
        """
        Initialize the LabelBoundedDataGenerator.
        
        Args:
            project: Project instance containing paths and configuration
            plan: Processing plan dictionary containing spacing and other parameters
            folder_suffix: Optional suffix to append to output folder name
            data_folder: Path to input data folder. If None, auto-generated from spacing
            output_folder: Path to output folder. If None, auto-generated under project.lbd_folder
            mask_label: Specific label value to use for cropping. If None, uses all labels >0
            remapping: Dictionary for label value remapping (optional)
        """


        existing_fldr = find_matching_plan(project.db, plan)
        if existing_fldr is not None:
            print("Plan folder already exists in db: {}.\nWill use existing folder to add data".format( existing_fldr))
            output_folder=existing_fldr
        self.plan_name = plan_name
        self.plan = plan
        self.fg_indices_exclude = listify(plan.get("fg_indices_exclude"))
        self.expand_by = expand_by
        self.mask_label = mask_label
        self.lm_group = self.plan.get("lm_group")
        self.remapping = self.create_remapping_dict(remapping)
        if is_excel_None(self.lm_group):
            self.lm_group = "lm_group1"
        self.image_key = "image"
        self.lm_key = "lm"
        self.tnsr_keys = [self.image_key,self.lm_key]
        self.tfms_keys = "LT,E,D,C,Ind"
        Preprocessor.__init__(
            self,
            project=project,
            spacing=plan.get("spacing"),
            data_folder=data_folder,
            output_folder=output_folder,
        )

    def create_remapping_dict(self,remapping):
        return remapping

    def set_input_output_folders(self, data_folder, output_folder):
        if data_folder is None:
            self.set_folders_from_spacing(self.spacing)
        else:
            self.data_folder = Path(data_folder)
        if output_folder is None:
            self.set_output_folder(self.project.lbd_folder)
        else:
            self.output_folder = Path(output_folder)

    def set_folders_from_spacing(self, spacing):
        self.data_folder = folder_name_from_list(
            prefix="spc",
            parent_folder=self.fixed_spacing_folder,
            values_list=spacing,
        )

    def set_output_folder(self, parent_folder):
        self.output_folder = folder_name_from_list(
            prefix="spc",
            parent_folder=parent_folder,
            values_list=self.spacing,
        )
        if self.plan_name is not None:
            output_name = "_".join([self.output_folder.name, self.plan_name])
            self.output_folder = Path(
                self.output_folder.parent / output_name
            )

    def create_output_folders(self):
        maybe_makedirs(
            [
                self.output_folder / ("lms"),
                self.output_folder / ("images"),
                self.indices_subfolder,
            ]
        )


    def process(self):
        if not hasattr(self, "transforms"):
            print("No transforms created. No data to be processed. Run setup first")
            return 0
        assert len(self.df) > 0,"No new cases to process"
        self.create_output_folders()
        self.process_files()
        add_plan_to_db(self.plan,self.output_folder, db_path=self.project.db)
    

    def setup(self, device="cpu",  overwrite=True):
        device = resolve_device(device)
        print("Processing on ", device)
        self.register_existing_files()
        print("Overwrite:", overwrite)
        if overwrite == False:
            Preprocessor.remove_completed_cases(self)
        # Instead of creating dataset and dataloader, prepare file lists
        self.image_files = sorted(self.data_folder.glob("images/*.pt"))
        self.lm_files = sorted(self.data_folder.glob("lms/*.pt"))
        self.create_transforms(device=device)
        self.set_transforms(self.tfms_keys)

    def create_transforms(self, device):
        """Create transforms for processing images and labels"""
        keys = ["image", "lm"]
        transforms = [
            LoadTorchd(keys=keys),
            EnsureChannelFirstd(keys=keys),
            ToDeviced(device=device, keys=keys),
            FgBgToIndicesd2(keys = keys),
        ]
        if self.mask_label is not None:
            transforms.append(
                CropForegroundd(
                    keys=["image", "lm"],
                    source_key="lm",
                    select_fn=lambda x: x == self.mask_label
                )
            )
        return Compose(transforms)


    def create_transforms(self,device):
        margin = [int(self.expand_by / sp) for sp in self.spacing] 
        if self.mask_label is None:
            select_fn = is_positive
        else:
            select_fn = lambda lm: lm == self.mask_label
        # Transform attributes in alphabetical order
        self.C = CropForegroundd(
            keys=self.tnsr_keys,
            source_key=self.lm_key,
            select_fn=select_fn,
            margin=margin,
        )
        self.D = ToDeviced(
            device=device, keys=self.tnsr_keys
        )
        self.E = EnsureChannelFirstd(
            keys=self.tnsr_keys, channel_dim="no_channel"
        )
        self.Ind = FgBgToIndicesd2(
            keys=[self.lm_key],
            image_key=self.image_key,
            ignore_labels=self.fg_indices_exclude,
            image_threshold=-2600,
        )
        self.LT = LoadTorchd(keys=self.tnsr_keys)

        self.transforms_dict = {
            "C": self.C,
            "D": self.D,
            "E": self.E,
            "Ind": self.Ind,
            "LT": self.LT,
        }


    def process_files(self, force_store_props=False):
        """Process files without using DataLoader"""
        self.create_output_folders()
        self.results = []
        self.shapes = []

        # for img_file, lm_file in pbar(zip(self.image_files, self.lm_files), desc="Processing files", total=len(self.image_files)):
        for index, row in pbar(self.df.iterrows(), total=len(self.df)):
            try:
                # Load and process single case
                data = {
                    "image": row['image'],
                    "lm": row['lm'],
                }
                
                # Apply transforms
                data = self.transforms(data)
                
                # Get metadata and indices
                # Process the case
                self.process_single_case(
                    data["image"],
                    data["lm"],
                    data['lm_fg_indices'],
                    data['lm_bg_indices'],
                )
                
            except Exception as e:
                print(f"Error processing {row['image'].name}: {str(e)}")
                continue

        self.results_df = pd.DataFrame(self.results)
        # self.results= pd.DataFrame(self.results).values
        ts = self.results_df.shape
        if ts[-1] == 4 or force_store_props==True:  # only store if entire dset is processed
            self._store_dataset_properties()
            generate_bboxes_from_lms_folder(self.output_folder / ("lms"))
        else:
            print(
                "self.results  shape is {0}. Last element should be 4 , is {1}. therefore".format(
                    ts, ts[-1]
                )
            )
            print(
                "since some files skipped, dataset stats are not being stored. Either:\na) set force_store_props to True, or\nb) run self.get_tensor_folder_stats and generate_bboxes_from_lms_folder separately"
            )


    def create_properties_dict(self):
        resampled_dataset_properties = Preprocessor.create_properties_dict(self)
        ignore_keys = ["src_dest_labels", "mode", "spacing","samples_per_file", "src_dest_labels"]
        for key in self.plan.keys():
            if not key in ignore_keys:
                resampled_dataset_properties[key] = self.plan[key]
        return resampled_dataset_properties

    def process_single_case(self, image, lm, fg_inds, bg_inds):
        """Process a single case and save results"""
        assert image.shape == lm.shape, "mismatch in shape"
        assert image.dim() == 4, "images should be cxhxwxd"
        assert image.numel() > MIN_SIZE**3, f"image size is too small {image.shape}"
        
        inds = {
            "lm_fg_indices": fg_inds,
            "lm_bg_indices": bg_inds,
            "meta": image.meta,
        }
        
        self.save_indices(inds, self.indices_subfolder)
        self.save_pt(image[0], "images")
        self.save_pt(lm[0], "lms")
        self.extract_image_props(image)


    def get_case_ids_lm_group(self, lm_group):
        dsrcs = self.global_properties[lm_group]["ds"]
        cids = []
        for dsrc in dsrcs:
            cds = self.df["case_id"][self.df["ds"] == dsrc].to_list()
            cids.extend(cds)
        return cids


    def set_transforms(self, keys_tr: str):
        self.transforms = self.tfms_from_dict(keys_tr)

    def tfms_from_dict(self, keys: str):
        keys = keys.replace(" ", "").split(",")
        tfms = []
        for key in keys:
            tfm = self.transforms_dict[key]
            tfms.append(tfm)
        tfms = Compose(tfms)
        return tfms

    # def set_transforms(self, tfms: str = ""):
    #     tfms_final = []
    #     for tfm in tfms:
    #         tfms_final.append(getattr(self, tfm))
        self.transform = Compose(tfms_final)

    @property
    def indices_subfolder(self):
        if len(self.fg_indices_exclude) > 0:
            indices_subfolder = "indices_fg_exclude_{}".format(
                "".join([str(x) for x in self.fg_indices_exclude])
            )
        else:
            indices_subfolder = "indices"
        indices_subfolder = self.output_folder / indices_subfolder
        return indices_subfolder


class FGBGIndicesLBD(LabelBoundedDataGenerator):
    """
    Outputs FGBGIndices only. No images of lms are created.
    Use this generator when LBD images and lms are already created, but a new set of FG indices is required.
    """

    def __init__(self, project, data_folder, fg_indices_exclude: list = None) -> None:
        store_attr()
        self.data_folder = Path(data_folder)
        self.output_folder = Path(data_folder)

    def register_existing_files(self):
        self.existing_files = list(
            (self.output_folder / self.indices_subfolder).glob("*pt")
        )
        self.existing_case_ids = [
            info_from_filename(f.name, full_caseid=True)["case_id"]
            for f in self.existing_files
        ]
        self.existing_case_ids = set(self.existing_case_ids)
        print("Case ids processed in a previous session: ", len(self.existing_case_ids))

    def setup(self, device="cpu", batch_size=4, overwrite=False):
        device = resolve_device(device)
        print("Processing on ", device)
        self.register_existing_files()
        print("Overwrite:", overwrite)
        
        if not overwrite:
            self.remove_completed_cases()

        # Setup file lists and transforms
        if len(self.df) > 0:
            self.image_files = sorted(self.data_folder.glob("images/*.pt"))
            self.lm_files = sorted(self.data_folder.glob("lms/*.pt"))
            self.transforms = self.create_transforms(device)
        else:
            print("No cases to process.")

    def create_output_folders(self):
        maybe_makedirs(self.indices_subfolder)


if __name__ == "__main__":
# %%
    from fran.utils.common import *
    from fran.managers import Project

    project_title="litsmc"
    P = Project(project_title=project_title)
    P.maybe_store_projectwide_properties()
    # spacing = [1.5, 1.5, 1.5]

    conf = ConfigMaker(P, raytune=False, configuration_filename=None).config
    
    plan_str = "plan9"
    plan = conf[plan_str]
    spacing = plan['spacing']

    L = LabelBoundedDataGenerator(
        project=P,
        plan=plan,
        mask_label=None,
        plan_name=plan_str,
    )
    
    L.setup(device='cpu', overwrite=False)
# %%
    L.process()
# %%
# %%
#SECTION:-------------------- TS--------------------------------------------------------------------------------------

    # for img_file, lm_file in zip(I.L.image_files, I.L.lm_files):
    img_file, lm_file = L.image_files[0], L.lm_files[0]
    try:
            # Load and process single case
            data = {
                "image": img_file,
                "lm": lm_file,
            }
            
            # Apply transforms
            data = L.transforms(data)
            
            # Get metadata and indices
            fg_indices = L.get_foreground_indices(data["lm"])
            bg_indices = L.get_background_indices(data["lm"])
            coords = L.get_foreground_coords(data["lm"])
            
            # Process the case
            L.process_single_case(
                data["image"],
                data["lm"],
                fg_indices,
                bg_indices,
                coords["start"],
                coords["end"]
            )
            
    except Exception as e:
            print(f"Error processing {img_file.name}: {str(e)}")

# %%
    for index, row in L.df.iterrows():
        print(row)
        tr()
# %%
