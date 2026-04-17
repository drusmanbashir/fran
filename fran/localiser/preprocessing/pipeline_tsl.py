# %%
from fran.data.dataregistry import DS
from pathlib import Path

from fran.localiser.preprocessing.data.nii2pt_tsl import PreprocessorNII2PTTSL
from fran.localiser.preprocessing.data.pt2jpg_tsl import PreprocessorPT2JPG_TSL



# %%
if __name__ == "__main__":
    src_3d = DS["totalseg"].folder
    merge_windows = False
    letterbox = True
    outputsize = [512, 512]
    if merge_windows:
        out_fldr = Path("/s/xnat_shadow/totalseg2d_merged")
    else:
        out_fldr = Path("/s/xnat_shadow/totalseg2d")
    out_2d = out_fldr / "pt"
    out_yolo = out_fldr / "jpg"

# %%
    P = PreprocessorNII2PTTSL(src_3d, out_2d, merge_windows=merge_windows, exclude_regions=["gut", "neck"])
    P.setup(device="cpu", num_processes=16, debug=False)
    P.process()

# %%
    dm = PreprocessorPT2JPG_TSL(
        data_dir=out_2d,
        exclude_regions=["gut", "neck"],
        merge_windows=merge_windows,
        letterbox=letterbox,
        outputsize=outputsize,
    )
# %%
    dm.export_yolo_dataset(out_yolo)
# %%
