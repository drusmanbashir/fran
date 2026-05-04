from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[3]
LOCAL_REPOS = [
    "fran",
    "localiser",
    "utilz",
    "label_analysis",
    "dicom_utils",
    "xnat",
]

repo_roots = [CODE_ROOT / repo_name for repo_name in LOCAL_REPOS]
sys.path[:] = [
    path
    for path in sys.path
    if Path(path or ".").resolve() != CODE_ROOT
]
sys.path[:0] = [str(repo_root) for repo_root in repo_roots if repo_root.exists()]

fran_conf = Path("/s/fran_storage/conf")
if fran_conf.exists():
    os.environ.setdefault("FRAN_CONF", str(fran_conf))


THIRD_PARTY_MODULES = [
    "numpy",
    "pandas",
    "torch",
    "torchvision",
    "torchaudio",
    "monai",
    "lightning",
    "itk",
    "SimpleITK",
    "h5py",
    "scipy",
    "skimage",
    "ultralytics",
    "ray",
    "nnunet",
    "nnunetv2",
    "torchio",
    "wandb",
    "plotly.express",
    "seaborn",
    "vtk",
    "pydicom",
    "pyxnat",
    "roboflow",
    "supervision",
]


INTERNAL_MODULES = [
    # utilz / label_analysis surface
    "utilz.fileio",
    "utilz.helpers",
    "utilz.imageviewers",
    "utilz.itk_sitk",
    "label_analysis.geometry",
    "label_analysis.overlap",
    "label_analysis.visualizaton",
    # dicom / xnat surface
    "dicom_utils.helpers",
    "dicom_utils.dcm_to_sitk",
    "dicom_utils.sitk_to_dcm",
    "xnat.helpers",
    "xnat.object_oriented",
    "xnat.totalseg",
    # localiser surface
    "localiser.data.dataset",
    "localiser.inference.base",
    "localiser.inference.localiserinferer",
    "localiser.train.training",
    "localiser.transforms.transforms",
    "localiser.utils.bbox_helpers",
    # fran surface
    "fran.api.dcm_inference",
    "fran.architectures.dynunet",
    "fran.callback.case_recorder",
    "fran.callback.wandb.wandb",
    "fran.data.dataset",
    "fran.inference.base",
    "fran.inference.cascade_yolo",
    "fran.managers.unet",
    "fran.preprocessing.imported",
    "fran.preprocessing.regionbounded",
    "fran.run.preproc.analyze_resample",
    "fran.transforms.imageio",
]


def test_repo_integrity_import_surface():
    for module_name in THIRD_PARTY_MODULES + INTERNAL_MODULES:
        importlib.import_module(module_name)
