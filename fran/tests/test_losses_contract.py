import torch
from monai.data import MetaTensor

from fran.evaluation.losses import CombinedLoss, DeepSupervisionLoss
from fran.utils.common import PAD_VALUE


def _target_with_meta(values, filenames):
    return MetaTensor(
        torch.tensor(values, dtype=torch.long),
        meta={"filename_or_obj": filenames},
    )


def test_combined_loss_preserves_case_recorder_keys():
    loss_fnc = CombinedLoss(
        fg_classes=1,
        include_background=False,
        softmax=True,
    )
    logits = torch.randn(2, 2, 2, 2, 2)
    target = _target_with_meta(
        [
            [[[[0, 1], [1, 0]], [[0, 1], [1, 0]]]],
            [[[[1, 1], [0, 0]], [[1, 0], [0, 1]]]],
        ],
        ["kits23_001.nii.gz", "kits23_002.nii.gz"],
    )

    loss = loss_fnc(logits, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss_fnc.loss_dict["batch0_filename"] == "kits23_001.nii.gz"
    assert loss_fnc.loss_dict["batch1_case_id"] == "kits23_002"
    assert "loss_dice_label1" in loss_fnc.loss_dict
    assert "loss_dice_batch0_label1" in loss_fnc.loss_dict


def test_deep_supervision_loss_supports_target_sanitizer_and_pad_mask():
    loss_fnc = DeepSupervisionLoss(
        levels=2,
        deep_supervision_scales=[[1, 1, 1], [0.5, 0.5, 0.5]],
        fg_classes=1,
        include_background=False,
    )
    loss_fnc.set_target_label_sanitizer(lambda target: (target > 0).long())
    preds = [
        torch.randn(1, 2, 4, 4, 4),
        torch.randn(1, 2, 2, 2, 2),
    ]
    target = _target_with_meta(
        [
            [
                [
                    [
                        [0, 2, 0, PAD_VALUE],
                        [2, 0, 0, 0],
                        [0, 0, 2, 0],
                        [0, 0, 0, 2],
                    ],
                    [
                        [0, 2, 0, 0],
                        [2, 0, 0, 0],
                        [0, 0, 2, 0],
                        [0, 0, 0, 2],
                    ],
                    [
                        [0, 2, 0, 0],
                        [2, 0, 0, 0],
                        [0, 0, 2, 0],
                        [0, 0, 0, 2],
                    ],
                    [
                        [0, 2, 0, 0],
                        [2, 0, 0, 0],
                        [0, 0, 2, 0],
                        [0, 0, 0, 2],
                    ],
                ]
            ]
        ],
        ["kits23_003.nii.gz"],
    )

    loss = loss_fnc(preds, target, use_mask=True)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss_fnc.loss_dict["batch0_case_id"] == "kits23_003"
    assert "loss_dice_label1" in loss_fnc.loss_dict
    assert "loss_dice_label2" not in loss_fnc.loss_dict
