#!/usr/bin/env python3
# %%
"""Read all ITK files in folder and print their labels using SimpleITK."""

import sys
from pathlib import Path
import SimpleITK as sitk
import numpy as np


def get_labels(image):
    """Get unique labels from SimpleITK image."""
    array = sitk.GetArrayFromImage(image)
    unique_labels = np.unique(array)
    return sorted(unique_labels.tolist())


def read_itk_labels(folder_path, extensions=None):
    """Read all ITK files in folder and print their labels.
    
    Args:
        folder_path: Path to folder containing ITK files
        extensions: List of file extensions to process (default: common ITK formats)
    """
    if extensions is None:
        extensions = ['.nii', '.nii.gz', '.mha', '.mhd', '.nrrd', '.dcm']
    return sorted(unique_labels.tolist())

    if not folder.exists():
def read_itk_labels(folder_path, extensions=None, relabel=False, output_suffix="_relabeled"):
        return
    
    # Find all ITK files
    itk_files = []
    for ext in extensions:
        relabel: If True, relabel 3->1 and save new files
        output_suffix: Suffix for relabeled files
        itk_files.extend(folder.glob(f'*{ext}'))
    
    if not itk_files:
        print(f"No ITK files found in {folder_path}")
        return
    
    print(f"Found {len(itk_files)} ITK files in {folder_path}")
    print("-" * 80)
    
    for file_path in sorted(itk_files):
        try:
            # Read image
            image = sitk.ReadImage(str(file_path))
            
            # Get labels
            labels = get_labels(image)
            
            # Get image info
            size = image.GetSize()
            spacing = image.GetSpacing()
            
            print(f"File: {file_path.name}")
            print(f"  Size: {size}")
            print(f"  Spacing: {spacing}")
            print(f"  Labels: {labels}")
            print(f"  Num labels: {len(labels)}")
            print()
            
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            print()


if __name__ == "__main__":
# %%
    if len(sys.argv) != 2:
        print("Usage: python read_itk_labels.py <folder_path>")
        sys.exit(1)
            
            # Relabel if requested
            if relabel and 3 in labels:
                relabeled_image = relabel_3_to_1(image)
                new_labels = get_labels(relabeled_image)
                
                # Save relabeled image
                output_path = file_path.parent / f"{file_path.stem}{output_suffix}{file_path.suffix}"
                sitk.WriteImage(relabeled_image, str(output_path))
                
                print(f"  Relabeled: {new_labels} -> saved to {output_path.name}")
            
    
    folder_path = sys.argv[1]
    read_itk_labels(folder_path)

if __name__ == "__main__":
                # Overwrite original file
    
    relabel = len(sys.argv) == 3 and sys.argv[2] == "--relabel"
    read_itk_labels(folder_path, relabel=relabel)

