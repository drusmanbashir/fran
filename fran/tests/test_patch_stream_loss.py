import torch
from monai.data.meta_tensor import MetaTensor

from fran.evaluation.losses import CombinedLoss
from fran.evaluation.patch_stream_loss import PatchStreamValidationLoss
from fran.utils.common import PAD_VALUE


def _target(vals):
    tens = MetaTensor(torch.tensor(vals, dtype=torch.long).view(1, 1, 1, 1, -1))
    tens.meta["filename_or_obj"] = "kits23_99999.pt"
    return tens


def test_patch_stream_validation_loss_matches_full_case_dice_with_padding_mask():
    base = CombinedLoss(fg_classes=2, include_background=False, softmax=True)
    wrapped = PatchStreamValidationLoss(base)

    full_logits = torch.tensor(
        [
            [
                [[[-0.3, -0.1, 1.7, -0.2, 1.4]]],
                [[[2.4, 2.0, -0.6, 2.3, -0.5]]],
                [[[-0.8, -0.7, -0.4, -0.8, 2.8]]],
            ]
        ],
        dtype=torch.float32,
    )
    full_target = _target([1, 1, 0, 1, 2])

    _ = base(full_logits, full_target, use_mask=True)
    expected = {
        "loss_dice": float(base.loss_dict["loss_dice"]),
        "loss_dice_label1": float(base.loss_dict["loss_dice_label1"]),
        "loss_dice_label2": float(base.loss_dict["loss_dice_label2"]),
    }

    patch1_logits = full_logits[..., :3]
    patch1_target = _target([1, 1, 0])
    patch2_logits = torch.zeros((1, 3, 1, 1, 3), dtype=torch.float32)
    patch2_logits[..., :2] = full_logits[..., 3:]
    patch2_target = _target([1, 2, PAD_VALUE])

    batch1 = {"case_id": ["kits23_99999"], "patch_index": [0], "patches_in_case": [2]}
    batch2 = {"case_id": ["kits23_99999"], "patch_index": [1], "patches_in_case": [2]}

    _ = wrapped(patch1_logits, patch1_target, batch1, use_mask=True)
    assert "loss_dice_case" not in wrapped.loss_dict

    _ = wrapped(patch2_logits, patch2_target, batch2, use_mask=True)
    assert torch.isclose(wrapped.loss_dict["loss_dice_case"], torch.tensor(expected["loss_dice"]), atol=1e-6)
    assert torch.isclose(
        wrapped.loss_dict["loss_dice_case_label1"], torch.tensor(expected["loss_dice_label1"]), atol=1e-6
    )
    assert torch.isclose(
        wrapped.loss_dict["loss_dice_case_label2"], torch.tensor(expected["loss_dice_label2"]), atol=1e-6
    )


def test_patch_stream_validation_loss_flushes_completed_case_in_mixed_batch():
    base = CombinedLoss(fg_classes=1, include_background=False, softmax=True)
    wrapped = PatchStreamValidationLoss(base)

    first_patch_logits = torch.tensor([[[[[0.0, 0.0]]], [[[3.0, 3.0]]]]], dtype=torch.float32)
    first_patch_target = _target([1, 1])
    full_case_logits = torch.tensor([[[[[0.0, 0.0, 0.0, 0.0]]], [[[3.0, 3.0, 3.0, 3.0]]]]], dtype=torch.float32)
    full_case_target = _target([1, 1, 1, 1])
    _ = base(full_case_logits, full_case_target, use_mask=True)
    expected_case_loss = torch.tensor(float(base.loss_dict["loss_dice"]), dtype=torch.float32)

    _ = wrapped(
        first_patch_logits,
        first_patch_target,
        {"case_id": ["caseA"], "patch_index": [0], "patches_in_case": [2]},
        use_mask=True,
    )

    mixed_logits = torch.tensor(
        [
            [[[[0.0, 0.0]]], [[[3.0, 3.0]]]],
            [[[[0.0, 0.0]]], [[[3.0, 3.0]]]],
        ],
        dtype=torch.float32,
    )
    mixed_target = MetaTensor(torch.tensor([[[[[1, 1]]]], [[[[1, 1]]]]], dtype=torch.long))
    mixed_target.meta["filename_or_obj"] = ["caseA.pt", "caseB.pt"]
    _ = wrapped(
        mixed_logits,
        mixed_target,
        {
            "case_id": ["caseA", "caseB"],
            "patch_index": [1, 0],
            "patches_in_case": [2, 2],
        },
        use_mask=True,
    )

    assert wrapped.completed_cases[-1]["case_id"] == "caseA"
    assert "caseB" in wrapped._case_state
    assert torch.isclose(wrapped.loss_dict["loss_dice_case"], expected_case_loss, atol=1e-6)
