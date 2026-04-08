from fran.data.dataregistry import DS
from pathlib import Path

from fran.localiser.preprocessing.data.nii2pt_tsl import PreprocessorNII2PTTSL
from fran.localiser.preprocessing.data.pt2jpg_tsl import PreprocessorPT2JPG_TSL



if __name__ == "__main__":
    src_3d = DS["totalseg"].folder
    out_fldr = Path("/s/xnat_shadow/totalseg2d")
    out_2d = out_fldr / "pt"
    out_yolo = out_fldr / "jpg"

# %%
    P = PreprocessorNII2PTTSL(src_3d, out_2d)
    P.setup(device="cpu", num_processes=8, debug=False)
    P.process()

    dm = PreprocessorPT2JPG_TSL(data_dir=out_2d)
    dm.export_yolo_dataset(out_yolo)
# %%
