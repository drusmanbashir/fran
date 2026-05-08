import pandas as pd

from fran.managers.wandb.wandb import WandbManager


class FakeRun:
    def __init__(self, rows):
        self.rows = rows

    def scan_history(self, keys):
        for row in self.rows:
            yield {key: row.get(key) for key in keys}


def test_df_columns_returns_requested_chart_columns():
    manager = object.__new__(WandbManager)
    manager._remote_run_for = lambda run_id=None: FakeRun(
        [
            {"epoch": 1, "val0_loss": 0.5, "val0_loss_dice": 0.3},
            {"epoch": 2, "val0_loss": 0.4, "val0_loss_dice": 0.29},
            {"epoch": None, "val0_loss": None, "val0_loss_dice": None},
        ]
    )

    df = manager.df_columns(["epoch", "val0_loss", "val0_loss_dice"])

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["epoch", "val0_loss", "val0_loss_dice"]
    assert len(df) == 2
    assert df.iloc[-1].to_dict() == {
        "epoch": 2.0,
        "val0_loss": 0.4,
        "val0_loss_dice": 0.29,
    }


def test_df_columns_keeps_all_nan_rows_when_requested():
    manager = object.__new__(WandbManager)
    manager._remote_run_for = lambda run_id=None: FakeRun(
        [{"epoch": 1, "val0_loss": 0.5}, {"epoch": None, "val0_loss": None}]
    )

    df = manager.df_columns(["epoch", "val0_loss"], drop_all_nan=False)

    assert len(df) == 2
