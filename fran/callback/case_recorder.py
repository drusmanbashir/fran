import plotly.express as px
import itertools as il
import re
import sys
from pathlib import Path
from typing import Any

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from lightning.pytorch.utilities.types import STEP_OUTPUT
from utilz.cprint import cprint
from utilz.stringz import cleanup_fname

from fran.callback.base import *

# %%




class CaseIDRecorder(Callback):
    def __init__(self, freq=5, local_folder="/tmp", dpi=300):
        """

        :param freq:
        :param local_folder:
        :param dpi:
        """
        self.nep_field = "metrics/case_id_dices"
        self.set_figure_params(dpi)
        self.freq = freq
        self.local_folder = Path(local_folder)


    def on_fit_start(self, trainer, pl_module):
        self.incrementing=False
        self.loss_dicts_train = []
        self.loss_dicts_valid = []
        self.loss_dicts_train2=[]
        self.dfs = {}


    def _before_batch(self, batch):
        self.files_this_batch = batch["image"].meta["filename_or_obj"]

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        self._before_batch(batch)


    def on_validation_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        self._before_batch(batch)



    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        loss_dict_full = pl_module.loss_dict_full
        self.loss_dicts_train.append(loss_dict_full)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        loss_dict_full = pl_module.loss_dict_full
        if self.incrementing == False:
            self.loss_dicts_valid.append(loss_dict_full)
        else:
            self.loss_dicts_train2.append(loss_dict_full)

    def reset(self):
            self.loss_dicts_valid = []
            self.loss_dicts_train = []
            self.loss_dicts_train2=[]
            self.incrementing=False
    

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch >0 and trainer.current_epoch % self.freq == 0 or self.incrementing==True:
            self.store_results(trainer)
            self.reset()


    def _store(self,trainer, stage, loss_dict,epoch):
            mini_df = self.create_limited_df(loss_dict)
            df_final = self.pivot_batch_cols(mini_df)
            val_vars  = [var for var in df_final.columns if "dice" in var]
            df_long = df_final.melt(
                  id_vars="case_id",
                  value_vars=val_vars,
                  var_name="label",
                  value_name="loss_dice"
              )
            df_long.dropna(inplace=True)
            self.dfs[stage] = df_long
            fig = self.create_plotly(df_long)
            fig_fname = f"{self.local_folder}/{stage}_{epoch}.png"
            fig.write_image(fig_fname, scale=2)
            try:
                # trainer.logger.log_image(key = f"{stage}_boxplots", images=[fig_fname])
                trainer.logger.log_image(key = f"{stage}_boxplots", images=[fig_fname])
            except AttributeError as e:
                cprint(e, color = "red")

    def store_results(self, trainer):
        epoch = trainer.current_epoch
        self.dfs["epoch"]=epoch
        cprint("CaseIDRecorder: Storing results", color = "green", italic=True)
        if self.incrementing==False:
            for stage ,loss_dict in zip (["train", "valid"], [self.loss_dicts_train, self.loss_dicts_valid]):
                self._store(trainer, stage, loss_dict,epoch)
        else:
            self._store(trainer, "train2", self.loss_dicts_train2,epoch)
        trainer.dfs = self.dfs


    def set_figure_params(self, dpi):
        self.rcs = [
            {"figure.dpi": dpi, "figure.figsize": (15, 10)},  # valid
            {"figure.dpi": dpi, "figure.figsize": (25, 10)},
        ]  # train


    def create_limited_df(self,dicts):
        df_train = pd.DataFrame(dicts)
        bad_cols  = ["loss", "loss_ce", "loss_dice"]
        others = [col for col in df_train.columns if "filename" in col]
        others2 = [col for col in df_train.columns if not "batch" in col]
        bad_cols_all = bad_cols+others+ others2
        dft = df_train.drop(columns=bad_cols_all)
        return dft


    def pivot_batch_cols(self,dft):
        batch_vars  = [var for var in dft.columns if re.search(r"batch.*id", var)] 
        num_batches = len(batch_vars)
        dfs = []
        for n in range(num_batches):
            batch_var = "batch"+str(n)
            df1 = dft.loc[:,dft.columns.str.contains(batch_var)]
            df1.columns= df1.columns.str.replace(batch_var+"_", "")
            dfs.append(df1)
        df_final = pd.concat(dfs, axis=0)
        return df_final

    def create_plotly(self, df_long):
            fig = px.box(
                  df_long,
                  x="case_id",
                  y="loss_dice",
                  facet_col="label",      # separate box plot per label
                  color="label",
                  points="outliers"
              )
            return fig



# %%
if __name__ == "__main__":
    # dfd = pd.read_html('~/Downloads/valid.html')
    # rn = 45000
    # df1 = dfd[0]
    #
    # df2 = df1[-45000::]
    # %%
    df2 = pd.read_csv("/tmp/small2.csv")
    # %%
    plt.ioff()
    df2 = df2.melt(id_vars=["case_id", "filename"])
    df2 = df2[df2.variable.str.contains("Unnamed") == False]
    df2.variable = df2.variable.astype("category")

    figure = px.box(df2, x="case_id", y="value", color="variable")
    figure.write_image(file="/tmp/valid.png", width=1000, height=700, scale=2)
    # figure.show()
    # %%
    # %%
    np.random.seed(1234)
    df = pd.DataFrame(np.random.randn(10, 4), columns=["Col1", "Col2", "Col3", "Col4"])
    # %%
    box = df2.boxplot(column=["variable"])
    # %%
    ax = sns.boxplot(x="case_id", y="value", hue="variable", data=df2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    figure = ax.figure
    figure.tight_layout()
    figure.savefig("/tmp/tt.png")
    plt.show()
    # %%
    ax = sns.boxplot(x="case_id", y="value", hue="variable", data=df2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    figure = ax.figure
    figure.tight_layout()
    plt.show()
# %%
    df_train = pd.DataFrame(cir.loss_dicts_train)
    batch_size = 2
    batch_idx = range(2)
    batch_strings = []
    for id in batch_idx:
        batch_strings.append(
           "batch"+str(id) 
        )

    batch_str = batch_strings[0]

    small_df = df_train
    dfd = small_df

# %%

    cid_vars =  [var for var in small_df.columns if "case" in var]

# %%

    dfs = dfd.loc[:,dfd.columns.str.contains(batch_str)]
    val_cols = [col for col in dfs.columns if "loss_dice" in col]
    cid_vars =  [var for var in dfs.columns if "case" in var]
    df2 = dfs.melt(id_vars=cid_vars, value_vars=val_cols)

    import plotly.express as px
    figure = px.box(df2, x=cid_vars[0], y="value", color="variable")
    figure.savefig(fname_plot)
    df= df2
# %%
#
#       out = (
#           df2.assign(label=df2["variable"].str.extract(r"(label\d+)$", expand=False))
#              .pivot_table(
#                  index="batch0_caseid",
#                  columns="label",
#                  values="value",
#                  aggfunc="mean"   # or "first"
#              )
#              .reset_index()
#              .rename_axis(None, axis=1)
#              .rename(columns={"batch0_caseid": "case_id"})
#       )
# # %%
#     df2.to_csv("tmp2.csv")
#
#     df2 = df2[df2.variable.str.contains("Unnamed") == False]
#     df2.variable = df2.variable.astype("category")
# # %%
# %%
    # cprint("storing results", color = "yellow")
    # for stage ,loss_dict in zip (["train", "valid"], [cir.loss_dicts_train, cir.loss_dicts_valid]):
    #     mini_df = cir.create_limited_df(loss_dict)
    #     df_final = cir.pivot_batch_cols(mini_df)
    #     val_vars  = [var for var in df_final.columns if "dice" in var]
    #     df_long = df_final.melt(
    #           id_vars="caseid",
    #           value_vars=val_vars,
    #           var_name="label",
    #           value_name="loss_dice"
    #       )
    #     fig = cir.create_plotly(df_long)
    #
# %%
# %%
