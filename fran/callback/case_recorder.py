import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch
from fran.callback.base import *
from lightning.pytorch.utilities.types import STEP_OUTPUT
from tqdm.auto import tqdm
from utilz.cprint import cprint
from utilz.fileio import load_json, save_json
from utilz.stringz import headline

try:
    from fran.callback.case_ids_json_tracker import update_case_ids_json
except Exception:
    update_case_ids_json = None

# %%


def chunked_case_ids(case_ids, chunk_size=25):
    for i in range(0, len(case_ids), chunk_size):
        yield case_ids[i : i + chunk_size]


def check_continuous_ints(labels_all):
    u = np.unique(labels_all)
    return np.array_equal(u, np.arange(u.min(), u.max() + 1))


def infer_labels_and_update_out_channels(dm, configs: dict):

    labels_all_chunks = []

    headline("Infer labels and update out_channels")

    def _scan(dl):
        for batch in tqdm(dl):
            labels_all_chunks.append(torch.unique(batch["lm"]))

        # after scanning all loaders

    try:
        dm.prepare_data()
        labels_all = load_json(dm.data_folder / ("labels_all.json"))
        labels_all = set(labels_all)
    except FileNotFoundError:
        cprint("No labels_all.json found. Scanning data loaders.", color="yellow")

        dm.setup(stage="fit")
        _scan(dm.train_dataloader())
        val_dl = dm.val_dataloader()
        if isinstance(val_dl, (list, tuple)):
            for dl in val_dl:
                _scan(dl)
        else:
            _scan(val_dl)

        labels_all = set(torch.unique(torch.cat(labels_all_chunks)).cpu().tolist())
        labels_all = set(sorted(int(v) for v in labels_all))

        # Store with same format as Preprocessor.store_labels_info:
        # sorted JSON list of integer label values.
        labels_all_sorted = sorted(int(v) for v in labels_all)
        save_json(labels_all_sorted, dm.train_manager.data_folder / "labels_all.json")
    max_label = max(labels_all)
    out_channels = int(max_label) + 1

    if labels_all != set(range(out_channels)):
        warnings.warn(f"Label values have gaps: {labels_all}", stacklevel=2)

    configs["plan_train"]["labels_all"] = sorted(labels_all)
    if "plan_valid" in configs:
        configs["plan_valid"]["labels_all"] = sorted(labels_all)
    configs["model_params"]["out_channels"] = out_channels
    cprint(
        f"Labels: {labels_all}, out_channels: {out_channels}",
        color="magenta",
        bold=True,
    )

    return labels_all, out_channels


class CaseIDRecorder(Callback):
    def __init__(self, vip_label=1, freq=5, local_folder="/tmp", dpi=300, plot_x=50):
        """

        :param freq:
        :param local_folder:
        :param dpi:
        :param plot_x: number of cases to plot in each subplot, fewer the cases , the more the plots wil be genrated
        """
        self.set_figure_params(dpi)
        self.freq = freq
        self.local_folder = Path(local_folder)
        self.local_folder.mkdir(parents=True, exist_ok=True)
        self.plot_x = plot_x
        self.width = 80 * self.plot_x + 200
        self._warned_plotly_export = False
        self.vip_label = vip_label

    def on_fit_start(self, trainer, pl_module):
        self.incrementing = False
        self.loss_dicts_train = []
        self.loss_dicts_valid = []
        self.loss_dicts_train2 = []
        self.dfs = {}
        self.worst_case_ids = {}

    def _before_batch(self, batch):
        self.files_this_batch = batch["image"].meta["filename_or_obj"]

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._before_batch(batch)

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._before_batch(batch)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss_dict_full = pl_module.loss_dict_full
        self.loss_dicts_train.append(loss_dict_full)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        loss_dict_full = pl_module.loss_dict_full
        if self.incrementing == False:
            self.loss_dicts_valid.append(loss_dict_full)
        else:
            self.loss_dicts_train2.append(loss_dict_full)

    def reset(self):
        self.loss_dicts_valid = []
        self.loss_dicts_train = []
        self.loss_dicts_train2 = []
        self.incrementing = False

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        epoch = trainer.current_epoch + 1
        if (epoch > 0 and epoch % self.freq == 0) or self.incrementing == True:
            self.store_results(trainer)
            self.reset()

    def _store(self, trainer, stage, loss_dicts, epoch):
        mini_df = self.create_limited_df(loss_dicts)
        df_final = self.pivot_batch_cols(mini_df)
        val_vars = [
            var for var in df_final.columns if "dice" in var
        ]  # or "shape" in var] shape creates issues
        df_long = df_final.melt(
            id_vars="case_id",
            value_vars=val_vars,
            var_name="label",
            value_name="loss_dice",
        )
        df_long.dropna(inplace=True)
        self.dfs[stage] = df_long
        self._log_df_to_wandb(
            trainer=trainer, df_long=df_long, stage=stage, epoch=epoch
        )
        figs_labels_caseidchunks = self.create_plotly(
            df_long, stage, chunk_size=self.plot_x
        )

        self._save_and_log_plotly(
            figs=figs_labels_caseidchunks,
            trainer=trainer,
            stage=stage,
            epoch=epoch,
        )
        if update_case_ids_json is not None:
            update_case_ids_json(
                trainer=trainer,
                pl_module=trainer.lightning_module,
                stage=stage,
                epoch=epoch,
                df_long=df_long,
            )

    def _save_and_log_plotly(self, figs, trainer, stage, epoch):
        for label in figs.keys():
            figsi = figs[label]
            for ind, (fig_casegroup, df_chunk, cases_chunk) in enumerate(figsi):
                fig_fname = (
                    f"{self.local_folder}/{stage}_{epoch}_{label}_casesgp{ind}.png"
                )
                if not self._write_plot(
                    fig_casegroup, df_chunk, cases_chunk, label, fig_fname
                ):
                    continue
                try:
                    # trainer.logger.log_image(key = f"{stage}_boxplots", images=[fig_fname])
                    trainer.logger.log_image(
                        key=f"{stage}_{label}_casesgp{ind}_boxplots", images=[fig_fname]
                    )
                except AttributeError as e:
                    cprint(e, color="red")


    def _is_wandb_logger(self, trainer) -> bool:
        logger = getattr(trainer, "logger", None)
        if logger is None:
            return False
        logger_name = logger.__class__.__name__
        return logger_name in {"WandbLogger", "WandbManager"}

    def _log_df_to_wandb(
        self, trainer, df_long: pd.DataFrame, stage: str, epoch: int
    ) -> None:
        if not self._is_wandb_logger(trainer):
            return
        import wandb
        try:
            run = trainer.logger.experiment
            table = wandb.Table(dataframe=df_long.reset_index(drop=True))
            key = f"case_recorder/{stage}/df_epoch_{epoch}"
            run.log({key: table}, step=trainer.global_step)
        except Exception as e:
            cprint(f"W&B dataframe logging failed: {e}", color="yellow")

    def store_results(self, trainer):
        epoch = trainer.current_epoch + 1
        self.dfs["epoch"] = epoch
        # cprint("CaseIDRecorder: Storing results", color = "green", italic=True)
        if self.incrementing == False:
            for stage, loss_dicts in zip(
                ["train", "valid"], [self.loss_dicts_train, self.loss_dicts_valid]
            ):
                self._store(trainer, stage, loss_dicts, epoch)
        else:
            self._store(trainer, "train2", self.loss_dicts_train2, epoch)
        trainer.dfs = self.dfs

    def set_figure_params(self, dpi):
        self.rcs = [
            {"figure.dpi": dpi, "figure.figsize": (15, 10)},  # valid
            {"figure.dpi": dpi, "figure.figsize": (25, 10)},
        ]  # train

    def create_limited_df(self, dicts):
        df_train = pd.DataFrame(dicts)
        bad_cols = ["loss", "loss_ce", "loss_dice"]
        others = [col for col in df_train.columns if "filename" in col]
        others2 = [col for col in df_train.columns if not "batch" in col]
        bad_cols_all = bad_cols + others + others2
        cols = df_train.columns
        cols_to_remove = set(cols).intersection(set(bad_cols_all))
        dft = df_train.drop(columns=cols_to_remove)
        return dft

    def pivot_batch_cols(self, dft):

        batch_vars = [var for var in dft.columns if re.search(r"batch.*id", var)]
        dfs = []
        num_batches = len(batch_vars)
        for n in range(num_batches):
            batch_var = "batch" + str(n) + "_"
            df1 = dft.loc[:, dft.columns.str.contains(batch_var)]
            df1.columns = df1.columns.str.replace(batch_var, "")
            dfs.append(df1)
        df_final = pd.concat(dfs, axis=0)
        df_final.dropna(inplace=True)
        return df_final

    def get_worst_case_ids(self, df_long, stage):
        case_ids = self.worst_case_ids.get(stage)
        if case_ids is None:
            self._set_worst_case_ids(df_long, stage)
            case_ids = self.worst_case_ids.get(stage)
        return case_ids

    def _set_worst_case_ids(self, df_long, stage):
        vip_label_str = "loss_dice_label" + str(self.vip_label)
        df_label = df_long[df_long["label"] == vip_label_str].copy()
        df_label["case_id"] = df_label["case_id"].astype(str)
        case_order = (
            df_label.groupby("case_id")["loss_dice"]
            .var()
            .fillna(0)
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )
        self.worst_case_ids[stage] = case_order

    def create_plotly(self, df_long, stage, chunk_size=25) -> dict:
        figs = {}
        labels = df_long["label"].unique()
        labels = [
            lab for lab in labels if "loss_dice_label" in lab
        ]  # remove shape from the group
        case_ids = self.get_worst_case_ids(df_long, stage)
        if len(case_ids) == 0:
            raise ValueError("No worst case ids found, thers a bug in the code")
        for label in labels:
            figs_this_label = []
            df_label = df_long[df_long["label"] == label].copy()
            for cases_chunk in chunked_case_ids(case_ids, chunk_size=chunk_size):
                df_chunk = df_label[
                    df_label["case_id"].astype(str).isin(cases_chunk)
                ].copy()
                fig = px.violin(
                    df_chunk,
                    x="case_id",
                    y="loss_dice",
                    points="all",
                    box=True,
                    category_orders={"case_id": cases_chunk},
                    title=f"{label}",
                )
                fig.update_traces(jitter=0.2, pointpos=0, spanmode="hard")
                fig.update_layout(width=self.width)
                fig.update_xaxes(tickangle=90, tickfont={"size": 24})
                fig.update_yaxes(range=[0, 1])
                figs_this_label.append((fig, df_chunk, list(cases_chunk)))
            figs[label] = figs_this_label
        return figs

    def _write_plot(self, fig, df_chunk, cases_chunk, label, fig_fname) -> bool:
        try:
            fig.write_image(fig_fname, scale=2)
            return True
        except Exception as e:
            if not self._warned_plotly_export:
                cprint(
                    f"Plotly static export unavailable ({e}). Falling back to seaborn/matplotlib.",
                    color="yellow",
                )
                self._warned_plotly_export = True
            return self._write_plot_matplotlib(df_chunk, cases_chunk, label, fig_fname)

    def _write_plot_matplotlib(self, df_chunk, cases_chunk, label, fig_fname) -> bool:
        try:
            plt.ioff()
            df_mpl = df_chunk.copy()
            df_mpl["case_id"] = df_mpl["case_id"].astype(str)
            order = [str(c) for c in cases_chunk]
            fig_w = max(12, min(80, int(len(order) * 0.75)))
            fig, ax = plt.subplots(figsize=(fig_w, 8), dpi=150)
            sns.violinplot(
                data=df_mpl,
                x="case_id",
                y="loss_dice",
                order=order,
                inner="box",
                cut=0,
                ax=ax,
            )
            sns.stripplot(
                data=df_mpl,
                x="case_id",
                y="loss_dice",
                order=order,
                color="black",
                size=2,
                alpha=0.35,
                ax=ax,
            )
            ax.set_title(str(label))
            ax.tick_params(axis="x", rotation=90, labelsize=8)
            fig.tight_layout()
            fig.savefig(fig_fname)
            plt.close(fig)
            return True
        except Exception as e:
            cprint(f"Fallback plot write failed: {e}", color="red")
            return False


#


# %%
if __name__ == "__main__":
    # dfd = pd.read_html('~/Downloads/valid.html')
# %%
# SECTION:-------------------- download tables-------------------------------------------------------------------------------------- <CR>
    from pathlib import Path

    import pandas as pd
    import wandb

    ENTITY = "drubashir"
    PROJECT = "kits2"
    RUN_ID = "KITS-0018"  # wandb run id
    OUT = Path("wandb_tables")
    OUT.mkdir(exist_ok=True)

    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

    epoch = 130

    key = f"case_recorder/fit/df_epoch_{epoch}"
# _store(self,trainer, stage, loss_dict,epoch):
