import torch
from torch.utils.data import DataLoader
from monai.data import MetaTensor

from fran.managers.data.valid_patch_stream import (
    ValidPatchStreamDataset,
    valid_patch_stream_collated,
)
from fran.managers import unet as unet_module
from fran.utils.common import PAD_VALUE


def _case(case_id, z_dim):
    image = MetaTensor(
        torch.ones(1, 4, 4, z_dim, dtype=torch.float32),
        meta={"filename_or_obj": f"{case_id}.pt"},
    )
    lm = MetaTensor(
        torch.zeros(1, 4, 4, z_dim, dtype=torch.int16),
        meta={"filename_or_obj": f"{case_id}.pt"},
    )
    return {"image": image, "lm": lm, "case_id": case_id}


def test_valid_patch_stream_preserves_single_case_per_patch_and_masks_padding():
    ds = ValidPatchStreamDataset(
        case_dataset=[_case("case1", 9), _case("case2", 2)],
        patch_size=(4, 4, 4),
    )

    patches = list(ds)

    assert [patch["case_id"] for patch in patches] == [
        "case1",
        "case1",
        "case1",
        "case2",
    ]
    assert patches[2]["patch_index"] == 2
    assert patches[2]["patches_in_case"] == 3
    assert patches[2]["is_padded"] is True
    assert torch.all(patches[2]["lm"][:, :, :, 1:] == PAD_VALUE)
    assert patches[3]["is_padded"] is True
    assert torch.all(patches[3]["lm"][:, :, :, 2:] == PAD_VALUE)


def test_valid_patch_stream_batching_can_mix_case_ids_across_patch_slots():
    ds = ValidPatchStreamDataset(
        case_dataset=[_case("case1", 9), _case("case2", 2)],
        patch_size=(4, 4, 4),
    )
    dl = DataLoader(ds, batch_size=2, collate_fn=valid_patch_stream_collated, num_workers=0)

    iter_dl = iter(dl)
    batch1 = next(iter_dl)
    batch2 = next(iter_dl)

    assert batch1["case_id"] == ["case1", "case1"]
    assert batch2["case_id"] == ["case1", "case2"]
    assert batch2["image"].shape == (2, 1, 4, 4, 4)
    assert batch2["validation_impl"] == "patch_stream"


def test_validation_step_uses_masked_loss_for_patch_stream_batches():
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
        AssertionError("patch_stream validation should not use SWI")
    )
    manager.batch_size = 2

    batch = {
        "image": torch.zeros(2, 1, 4, 4, 4),
        "lm": torch.zeros(2, 1, 4, 4, 4, dtype=torch.int16),
        "validation_impl": "patch_stream",
        "case_id": ["case1", "case2"],
    }

    loss, loss_dict = unet_module.UNetManager.validation_step(manager, batch, 0)

    assert manager.loss_fnc.use_mask is True
    assert torch.equal(batch["pred"], batch["image"] + 1)
    assert loss.item() == 1.0
    assert loss_dict["loss_dice"].item() == 0.75
