from pathlib import Path
from typing import Optional, List, Union
import torch
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, EnsureChannelFirstd, Orientationd, Spacingd
)
from monai.transforms.utility.dictionary import SqueezeDimd

from fran.data.dataset import NormaliseClipd
from fran.transforms.imageio import LoadSITKd
import SimpleITK as sitk
import itk
from monai.data.itk_torch_bridge import itk_image_to_metatensor as itm

class DataManagerNifti:
    def __init__(
        self,
        config,
        batch_size: int = 1,
        num_workers: int = 0,
        safe_mode: bool = False
    ):
        """
        Args:
            config: Configuration dictionary with spacing and other parameters
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            safe_mode: If True, uses safer but slower settings
        """
        self.batch_size = 1 if safe_mode else batch_size
        self.num_workers = 0 if safe_mode else num_workers
        
        # Extract parameters from config
        self.dataset_params = config.get('dataset_params')
        self.create_transforms()

    def create_transforms(self, keys="all"):
        """Creates transformations used for data preprocessing."""
        self.transforms_dict = {
            'L': LoadSITKd(
                keys=["image"],
                image_only=True,
                ensure_channel_first=False,
                simple_keys=True,
            ),
            'E': EnsureChannelFirstd(
                keys=["image"], 
                channel_dim="no_channel"
            ),
            'S': Spacingd(
                keys=["image"], 
                pixdim=self.spacing
            ),
            'N': NormaliseClipd(
                keys=["image"],
                clip_range=self.intensity_clip_range,
                mean=self.mean_fg,
                std=self.std_fg,
            ),
            'O': Orientationd(
                keys=["image"], 
                axcodes="RPS"
            )
        }



    def tfms_from_dict(self, keys: str):
        """Create transform composition from sequence string"""
        if not keys:
            return None
            
        keys = keys.split(",")
        tfms = []
        for key in keys:
            if key in self.transforms_dict:
                tfm = self.transforms_dict[key]
                if isinstance(tfm, list):
                    tfms.extend(tfm)
                else:
                    tfms.append(tfm)
        return Compose(tfms) if tfms else None

    def parse_input(self, data: Union[str, Path, sitk.Image, List]) -> List[dict]:
        """Parse various input types into standardized format"""
        if not isinstance(data, list):
            data = [data]
            
        processed_data = []
        for item in data:
            if isinstance(item, (str, Path)):
                item = Path(item)
                if item.is_dir():
                    processed_data.extend([{"image": img} for img in item.glob("*")])
                else:
                    processed_data.append({"image": item})
            elif isinstance(item, sitk.Image):
                processed_data.append({"image": item})
            elif isinstance(item, itk.Image):
                processed_data.append({"image": itm(item)})
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
                
        return processed_data

    def setup(self, stage: str = None) -> None:
        """Setup datasets and dataloaders"""
        if not hasattr(self, '_data'):
            raise ValueError("Please call prepare_data() with input data first")
            
        transforms = self.tfms_from_dict(self._keys_tfms)
        
        self.dataset = Dataset(
            data=self._data,
            transform=transforms
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def prepare_data(self, data: List, keys_tfms: str = "L,E,S,N,O") -> None:
        """Prepare data for inference"""
        self._data = self.parse_input(data)
        self._keys_tfms = keys_tfms

    def test_dataloader(self):
        """Return inference dataloader"""
        return self.dataloader

    @property
    def output_folder(self):
        """Get output folder from project"""
        return self.project.predictions_folder

