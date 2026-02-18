# %%
from typing import Dict
import itertools as il
import ipdb

from utilz.stringz import ast_literal_eval
tr = ipdb.set_trace

from fran.managers.unet import UNetManager
from pathlib import Path
from fran.trainers import checkpoint_from_model_id

import itk
import numpy as np
import SimpleITK as sitk
import torch
from fastcore.all import List, Optional, listify, store_attr
from fastcore.foundation import GetAttr
from lightning.fabric import Fabric
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.itk_torch_bridge import itk_image_to_metatensor as itm
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import SaveImaged
from monai.transforms.post.dictionary import Activationsd, AsDiscreted
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import (EnsureChannelFirstd,
                                                 SqueezeDimd)

from fran.data.dataset import NormaliseClipd
from fran.transforms.imageio import LoadSITKd
from fran.transforms.inferencetransforms import KeepLargestConnectedComponentWithMetad, SaveMultiChanneld, ToCPUd
from fran.transforms.spatialtransforms import ResizeToMetaSpatialShaped
from utilz.dictopts import DictToAttr, fix_ast
from utilz.helpers import slice_list
from utilz.imageviewers import ImageMaskViewer

class InferenceDataModule:
    """Handles data preparation for inference following Lightning's DataModule pattern"""
    
    def __init__(
        self,
        project,
        config: Dict,
        batch_size: int = 1,
        num_workers: int = 0,
        safe_mode: bool = False,
        transform_sequence: str = "ESN"
    ):
        """
        Args:
            project: Project object containing configuration
            config: Configuration dictionary with dataset_params and plan
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            safe_mode: If True, uses safer but slower settings
            transform_sequence: String specifying transform sequence (e.g., "ESN")
        """
        self.project = project
        self.config = config
        self.batch_size = 1 if safe_mode else batch_size
        self.num_workers = 0 if safe_mode else num_workers
        self.transform_sequence = transform_sequence
        
        # Extract parameters from config
        self.dataset_params = config.get('dataset_params', {})
        self.plan = config.get('plan', {})
        
        self.create_transforms()
        
    def create_transforms(self):
        """Initialize all possible transforms"""
        self.transforms = {}
        
        # Load transform
        self.transforms['L'] = LoadSITKd(
            keys=["image"],
            image_only=True,
            ensure_channel_first=False,
            simple_keys=True,
        )
        
        # Ensure channel first transform
        self.transforms['E'] = EnsureChannelFirstd(
            keys=["image"], 
            channel_dim="no_channel"
        )
        
        # Spacing transform
        spacing = self.plan.get("spacing")
        if spacing:
            self.transforms['S'] = Spacingd(
                keys=["image"], 
                pixdim=spacing
            )
            
        # Normalization transform
        self.transforms['N'] = NormaliseClipd(
            keys=["image"],
            clip_range=self.dataset_params.get("intensity_clip_range"),
            mean=self.dataset_params.get("mean_fg"),
            std=self.dataset_params.get("std_fg"),
        )
        
        # Orientation transform
        self.transforms['O'] = Orientationd(
            keys=["image"], 
            axcodes="RPS"
        )

    def set_transform_sequence(self, sequence: str = "") -> Compose:
        """Create transform composition from sequence string"""
        transform_list = []
        for key in sequence:
            if key in self.transforms:
                transform_list.append(self.transforms[key])
        return Compose(transform_list)

    def parse_input(self, imgs_inp) -> List[Dict]:
        """Parse various input types into standardized format"""
        if not isinstance(imgs_inp, list):
            imgs_inp = [imgs_inp]
            
        imgs_out = []
        for dat in imgs_inp:
            if any([isinstance(dat, str), isinstance(dat, Path)]):
                self.input_type = "files"
                dat = Path(dat)
                if dat.is_dir():
                    dat = list(dat.glob("*"))
                else:
                    dat = [dat]
            else:
                self.input_type = "itk"
                if isinstance(dat, sitk.Image):
                    pass
                elif isinstance(dat, itk.Image):
                    dat = itm(dat)
                else:
                    raise ValueError(f"Unsupported input type: {type(dat)}")
                dat = [dat]
            imgs_out.extend(dat)
            
        return [{"image": img} for img in imgs_out]

    def load_images(self, data):
        """Load images using appropriate loader"""
        data = self.parse_input(data)
        return [self.transforms['L'](d) for d in data]

    def setup(self, data, transform_sequence: Optional[str] = None) -> DataLoader:
        """Setup dataset and dataloader"""
        sequence = transform_sequence or self.transform_sequence
        data = self.load_images(data)
        transforms = self.set_transform_sequence(sequence)
        
        self.ds = Dataset(
            data=data,
            transform=transforms
        )
        
        self.dataloader = DataLoader(
            self.ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
        
        return self.dataloader

    @property
    def output_folder(self):
        """Get output folder from project"""
        return self.project.predictions_folder
