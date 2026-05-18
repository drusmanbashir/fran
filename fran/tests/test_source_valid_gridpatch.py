import numpy as np
import torch
from monai.data import MetaTensor

from fran.managers.data.main import DataManagerSource, PadLmOutsideOriginald
from fran.managers.data.valid_patch_stream import ValidPatchStreamDataset
from fran.managers import unet as unet_module
from fran.utils.common import PAD_VALUE


def test_pad_lm_outside_original_marks_only_out_of_bounds_voxels():
    transform = PadLmOutsideOriginald(keys=["lm"])
    lm = MetaTensor(torch.zeros(1, 4, 4, 4, dtype=torch.int16), meta={})
    data = {
        "lm": lm,
        "patch_coords": np.asarray([[0, 1], [-1, 3], [1, 5], [0, 4]]),
        "original_spatial_shape": (4, 4, 4),
    }

    out = transform(data)

    assert out["is_padded"] is True
    assert torch.all(out["lm"][:, 0, :, :] == PAD_VALUE)
    assert torch.all(out["lm"][:, :, 3, :] == PAD_VALUE)
    assert torch.all(out["lm"][:, 1:, :3, :] == 0)


def test_source_valid_dataset_uses_patch_stream_dataset():
    manager = DataManagerSource.__new__(DataManagerSource)
    manager.data = [
        {
            "image": MetaTensor(torch.ones(1, 4, 4, 5), meta={"filename_or_obj": "case_001.pt"}),
            "lm": MetaTensor(torch.zeros(1, 4, 4, 5, dtype=torch.int16), meta={"filename_or_obj": "case_001.pt"}),
            "case_id": "case_001",
        }
    ]
    manager.transforms = lambda x: x
    manager.ds_type = None
    manager.cache_rate = 0.0
    manager.plan = {"patch_size": (4, 4, 4)}
    manager.split = "valid"

    manager.create_dataset()

    assert isinstance(manager.ds, ValidPatchStreamDataset)
    patches = list(iter(manager.ds))
    assert len(patches) == 2
    assert patches[1]["image"].shape == (1, 4, 4, 4)
    assert torch.all(patches[1]["lm"][:, :, :, 1:] == PAD_VALUE)


def test_source_valid_split_keeps_effective_batch_size_for_patch_stream():
    manager = DataManagerSource.__new__(DataManagerSource)
    manager.collate_fn = None
    manager.batch_size = 4
    manager.effective_batch_size = 4

    manager.override_batch_size_valid_split(split="valid")

    assert manager.batch_size == 4
    assert manager.effective_batch_size == 4


def test_validation_step_uses_masked_direct_loss_for_grid_batches():
    class LossSpy:
        def __init__(self):
            self.use_mask = None
            self.loss_dict = {}

        def __call__(self, pred, target, use_mask=False):
            self.use_mask = use_mask
            self.loss_dict = {
                "loss": torch.tensor(1.0),
                "loss_ce": torch.tensor(0.25),
                "loss_dice": torch.tensor(0.75),
            }
            return torch.tensor(1.0)

    manager = unet_module.UNetManager.__new__(unet_module.UNetManager)
    manager.loss_fnc = LossSpy()
    manager.forward = lambda image: image + 1
    manager.log_losses = lambda loss_dict, prefix: None
    manager.maybe_store_preds = lambda pred: None
    manager.swi_on_val_batch = lambda batch, batch_idx: (_ for _ in ()).throw(
        AssertionError("grid validation should not use SWI")
    )
    manager.batch_size = 1

    batch = {
        "image": torch.zeros(1, 1, 4, 4, 4),
        "lm": torch.zeros(1, 1, 4, 4, 4, dtype=torch.int16),
        "patch_coords": [np.asarray([[0, 1], [0, 4], [0, 4], [0, 4]])],
    }

    loss, loss_dict = unet_module.UNetManager.validation_step(manager, batch, 0)

    assert manager.loss_fnc.use_mask is True
    assert torch.equal(batch["pred"], batch["image"] + 1)
    assert loss.item() == 1.0
    assert loss_dict["loss_dice"].item() == 0.75
