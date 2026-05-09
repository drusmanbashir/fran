from types import SimpleNamespace

import pandas as pd

from fran.callback.case_recorder import (
    CaseIDRecorder,
    selected_case_recorder_plot_labels,
)
from fran.callback.wandb.wandb import WandbImageGridCallback


class DoneFuture:
    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def result(self):
        return self._result


class FakeLogger:
    def __init__(self):
        self.logged_images = []
        self.logged_runs = []
        self.experiment = SimpleNamespace(log=self.logged_runs.append)

    def log_image(self, key, images):
        self.logged_images.append((key, images))


def test_case_id_recorder_drains_completed_exports_during_train_batch_start(tmp_path):
    callback = CaseIDRecorder(freq=5, local_folder=tmp_path)
    callback.on_fit_start(None, None)
    callback._pending_plot_exports = [
        DoneFuture(
            {
                "artifacts": [{"key": "valid_plot", "path": str(tmp_path / "plot.png")}],
                "plotly_fallback_message": None,
            }
        )
    ]
    trainer = SimpleNamespace(current_epoch=0, logger=FakeLogger())

    callback.on_train_batch_start(trainer, None, {}, 0)

    assert trainer.logger.logged_images == [
        ("valid_plot", [str(tmp_path / "plot.png")])
    ]
    assert callback._pending_plot_exports == []


def test_case_id_recorder_limits_plot_labels_to_requested_subset(tmp_path):
    callback = CaseIDRecorder(
        freq=5, local_folder=tmp_path, labels_for_plots=[1, 2, 3, 4]
    )
    callback.on_fit_start(None, None)
    df_long = pd.DataFrame(
        {
            "case_id": ["a", "a", "b", "b", "a", "b"],
            "label": [
                "loss_dice_label1",
                "loss_dice_label2",
                "loss_dice_label1",
                "loss_dice_label2",
                "loss_dice_label9",
                "loss_dice_label9",
            ],
            "loss_dice": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    callback.worst_case_ids["train"] = ["a", "b"]

    figs = callback.create_plotly(df_long, "train", chunk_size=10)

    assert list(figs) == ["loss_dice_label1", "loss_dice_label2"]


def test_selected_case_recorder_plot_labels_intersects_with_available_labels():
    labels = selected_case_recorder_plot_labels(
        ["loss_dice_label2", "loss_dice_label9", "shape", "loss_dice_label1"],
        [1, 2, 3, 4],
    )

    assert labels == ["loss_dice_label2", "loss_dice_label1"]


def test_wandb_grid_callback_drains_completed_renders_during_validation_batch_start(
    monkeypatch, tmp_path
):
    callback = WandbImageGridCallback(
        classes=2,
        patch_size=(8, 8, 8),
        epoch_freq=5,
        local_folder=tmp_path,
    )
    image_path = tmp_path / "grid.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    callback._pending_grid_renders = [
        DoneFuture({"image_path": str(image_path), "key": "images/grid"})
    ]
    logged = []
    trainer = SimpleNamespace(
        current_epoch=0,
        logger=SimpleNamespace(experiment=SimpleNamespace(log=logged.append)),
    )
    monkeypatch.setattr(
        "fran.callback.wandb.wandb.wandb.Image", lambda image: ("wandb-image", image.size)
    )

    callback.on_validation_batch_start(trainer, None, {}, 0)

    assert len(logged) == 1
    assert list(logged[0]) == ["images/grid"]
    assert callback._pending_grid_renders == []
    assert image_path.exists() is False
