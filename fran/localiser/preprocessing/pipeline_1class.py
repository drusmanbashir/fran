from fran.data.dataregistry import DS
from pathlib import Path

from fran.localiser.preprocessing.data.nii2pt import PreprocessorNII2PT
from fran.localiser.preprocessing.data.pt2jpg import DetectDataModule


if __name__ == "__main__":
    src_3d = DS["lidc"].folder
    out_2d = Path("/s/xnat_shadow/lidc2d")
    out_yolo = Path("/s/xnat_shadow/lidc2d_yolo")
    merge_windows = False
    outputsize = [512, 512]

    P = PreprocessorNII2PT(src_3d, out_2d, merge_windows=merge_windows)
    P.setup(device="cpu", num_processes=8, debug=False)
    P.process()

    dm = DetectDataModule(
        data_dir=out_2d,
        merge_windows=merge_windows,
        outputsize=outputsize,
    )
    dm.export_yolo_dataset(out_yolo)
