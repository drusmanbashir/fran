from types import SimpleNamespace

import pandas as pd
import torch

from fran.callback.case_recorder import (
    CaseIDRecorder,
    selected_case_recorder_plot_labels,
)
from fran.callback.wandb.wandb import WandbImageGridCallback


class FakeLogger:
    def __init__(self):
        self.logged_images = []
        self.logged_runs = []
        self.experiment = SimpleNamespace(log=self.logged_runs.append)

    def log_image(self, key, images):
        self.logged_images.append((key, images))


def test_case_id_recorder_train_batch_start_collects_without_async_state(tmp_path):
    callback = CaseIDRecorder(freq=5, local_folder=tmp_path)
    callback.on_fit_start(None, None)
    trainer = SimpleNamespace(current_epoch=4, logger=FakeLogger())
    batch = {"image": SimpleNamespace(meta={"filename_or_obj": "case.pt"})}

    callback.on_train_batch_start(trainer, None, batch, 0)

    assert callback.files_this_batch == "case.pt"


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


def test_wandb_grid_callback_logs_sync_grid_image_on_scheduled_epoch(monkeypatch):
    callback = WandbImageGridCallback(
        classes=2,
        patch_size=(8, 8, 8),
        epoch_freq=1,
    )
    callback.grid_imgs = [torch.zeros((4, 3, 4, 4), dtype=torch.uint8)]
    callback.grid_preds = [torch.zeros((4, 3, 4, 4), dtype=torch.uint8)]
    callback.grid_labels = [torch.zeros((4, 3, 4, 4), dtype=torch.uint8)]
    callback.grid_case_ids = [["a", "b", "c", "d"]]
    logged = []
    trainer = SimpleNamespace(
        current_epoch=0,
        logger=SimpleNamespace(experiment=SimpleNamespace(log=logged.append)),
    )
    monkeypatch.setattr(
        "fran.callback.wandb.wandb.wandb.Image",
        lambda image: ("wandb-image", image.shape),
    )

    callback.on_train_epoch_end(trainer, None)

    assert len(logged) == 1
    assert list(logged[0]) == ["images/grid"]
