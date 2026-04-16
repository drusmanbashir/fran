# Localiser Inference Context

Working folder: `/home/ub/code/fran/fran`

Main files touched:
- `localiser/inference/base.py`
- `localiser/helpers.py`

Current localiser inference path:
- `LocaliserInferer` loads NIfTI images with local lightweight `load_images_nifti`.
- Default device is CUDA 0: `device=0`.
- Default chunk size is `64` volumes.
- Preprocess sequence: `E,O,W,P1,P2,PF,R,Rep,E2,N`.
- `P1` is lateral projection: mean over dim `1`.
- `P2` is AP projection: mean over dim `2`.
- `PF` is `PermuteFlip2Dd`: `permute(0, 2, 1)` then vertical flip on each 2D projection.
- `R` is letterbox mode by default:
  - `Resized(..., size_mode="longest")`
  - `SpatialPadd(..., method="symmetric")`
- Collate flattens projections for bulk YOLO inference:
  - input batch of `B` 3D volumes becomes `[2B, 3, H, W]`
  - ordering: `case0 image1`, `case0 image2`, `case1 image1`, `case1 image2`, ...
- `predict()` calls `self.model(image, verbose=False)`.

Output image writing:
- `LocaliserInferer(..., out_folder=...)` stores annotated jpgs.
- Folder creation happens inside `run()` via `maybe_makedirs`.
- `draw_tensor_boxes` in `localiser/helpers.py` now accepts:
  - `filename=None`
  - `show=True`
- When `filename` is set, it saves jpg and returns the drawn image array.

Projection metadata:
- `collate_projections` builds `projection_meta` list.
- Each meta includes original `MetaTensor.meta` plus:
  - `case_index`
  - `projection_index`
  - `projection_key`
  - `letterbox_padded`
  - `letterbox_orig_size`
  - `letterbox_resized_size`
- MONAI letterbox padding is read from `MetaTensor.applied_operations`.
- Example padding from smoke:
  - `letterbox_padded = ((0, 0), (113, 113), (0, 0))`
  - `letterbox_orig_size = (61, 512)`
  - `letterbox_resized_size = (30, 256)`

BBox to slices:
- Added `yolo_bbox_to_slices`.
- Input bbox is YOLO normalized `cx, cy, w, h`.
- Output is 2D tensor slices in `(row_slice, col_slice)` order.
- It can undo letterbox padding using:
  - `padding`
  - `padded_shape`
  - `resized_shape`
- Added `yolo_bbox_to_3d_slices`.
- Input `spatial_shape_3d = (x, y, z)`.
- For `image1` / `p1` / `lat`, output is:
  - `(slice(None), y_slice, z_slice)`
- For `image2` / `p2` / `ap`, output is:
  - `(x_slice, slice(None), z_slice)`
- It accounts for YOLO `xywh`, tensor `row,col`, letterbox unpadding, and the vertical flip from `PF`.

Smoke tests done:
- Compile:
  - `python -m py_compile localiser/inference/base.py localiser/helpers.py`
- Import after removing heavy `fran.inference.helpers` dependency:
  - `from fran.localiser.inference.base import LocaliserInferer`
- CUDA check:
  - `cuda_available True`
  - `current_device 0`
  - `model_device cuda:0`
- One real KITS23 NIfTI smoke:
  - input: `/media/UB/datasets/kits23/images/kits23_00007.nii.gz`
  - output folder: `/tmp/fran_loc2d_run_smoke`
  - `n_batches 1`
  - `preds 2`
  - `image_shape (2, 3, 256, 256)`
  - `n_jpg 2`
- Full KITS23 folder run was started before final CUDA/default patch, then stopped after 120 jpgs.
  - target folder: `/media/UB/datasets/kits23/loc2d`
  - folder has partial old smoke outputs.

Potential next work:
- Re-run full KITS23 smoke after final patches:
  - `data_folder = Path("/media/UB/datasets/kits23/")`
  - `imgs_fldr = data_folder / "images"`
  - `out_fldr = data_folder / "loc2d"`
  - `model = YOLO("/s/fran_storage/yolo_output/totalseg_localiser/train32/weights/best.pt")`
  - `M = LocaliserInferer(model, imsize=256, window="a", projection_dim=(1, 2), out_folder=out_fldr)`
  - `out = M.run(list(imgs_fldr.glob("*")))`
- Verify slice mapping against real YOLO `Results.boxes.xywhn` and crop original post-orientation/resampling 3D tensor.
