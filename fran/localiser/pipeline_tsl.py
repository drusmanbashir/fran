from fran.data.dataregistry import DS
from pathlib import Path

from fran.localiser.data import DetectDataModuleTSL, PreprocessorNII2PTTSL


if __name__ == "__main__":
    src_3d = DS["totalseg"].folder
    out_2d = Path("/s/xnat_shadow/totalseg2d")
    out_yolo = Path("/s/xnat_shadow/totalseg2d_yolo")

    P = PreprocessorNII2PTTSL(src_3d, out_2d)
    P.setup(device="cpu", num_processes=8, debug=False)
    P.process()

    dm = DetectDataModuleTSL(data_dir=out_2d)
    dm.export_yolo_dataset(out_yolo)
