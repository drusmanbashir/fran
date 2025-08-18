import re
import plotly.express as px

import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fran.callback.base import *
import itertools as il

from pathlib import Path
from utilz.string import cleanup_fname

# %%


class DropBBox(Callback):
    order = 2
    def before_batch(self):
        self.learn.yb = [self.learn.xb[1]]
        self.learn.xb = [self.learn.xb[0]]


class CaseIDRecorder(Callback):
    def __init__(self, freq=50, local_folder="/tmp", dpi=300):
        """

        :param freq:
        :param local_folder:
        :param dpi:
        """
        self.nep_field = "metrics/case_id_dices"
        self.set_figure_params(dpi)
        store_attr()

    def before_fit(self):
        self.losses_dice = [], []
        self.df_titles = ["valid", "train"]
        self.dfs = {"valid": None, "train": None}

    def before_batch(self):
        self.files_this_batch = []
        for i, fn in enumerate(self.learn.yb[0]):
            self.files_this_batch.append(fn)

    def after_batch(self):
        dict_this_batch = {}
        full_dicts = []
        for k, v in self.learn.loss_func.loss_dict.items():
            if "batch" in k:
                dict_this_batch.update({k: v})
        for ind, name in enumerate(self.files_this_batch):
            pat = r"label\d"
            dici = {"filename": name}
            relevant_losses = {
                re.search(pat, key).group(): dict_this_batch[key]
                for key in dict_this_batch.keys()
                if "batch{}".format(ind) in key
            }
            dici.update(relevant_losses)
            full_dicts.append(dici)
        self.losses_dice[self.training].append(full_dicts)

    def after_epoch(self):
        for i, label in enumerate(self.df_titles):
            df = pd.DataFrame.from_dict(il.chain.from_iterable(self.losses_dice[i]))
            df["case_id"] = case_id_from_series(df.filename)
            self.append_to_running_df(label, df)
        if all([self.epoch >= self.freq, self.epoch % self.freq == 0]):
            if not hasattr(self, "rows_per_plot"):
                self.compute_rows_per_plot()
            self.store_results()

    def set_figure_params(self, dpi):
        self.rcs = [
            {"figure.dpi": dpi, "figure.figsize": (15, 10)},  # valid
            {"figure.dpi": dpi, "figure.figsize": (25, 10)},
        ]  # train

    def append_to_running_df(self, label, df):
        if isinstance(self.dfs[label], pd.DataFrame):
            self.dfs[label] = pd.concat([self.dfs[label], df], axis=0)
        else:
            self.dfs[label]: self.dfs[label] = df

    def store_results(self):
        for label in self.df_titles:
            small_df = self.create_limited_df(self.dfs[label])
            figure = self.create_plotly(small_df)
            fname_df = Path(self.local_folder) / ("{}.csv".format(label))
            self.dfs[label].to_csv(fname_df, index=False)
            # figure.savefig(fname_plot)
            if hasattr(self.learn, "nep_run"):
                # self.nep_run[self.nep_field+"_dataframes/{}".format(storage_string_df)].upload(File.as_html(self.dfs[label]))  TOO LARGE
                # self.nep_run["_".self.nep_field+"_plots/{}".format(storage_string_plot)].upload(fname_plot)
                # self.nep_run["_".join([self.nep_field,label])].log(File.as_image(figure))

                field_name = "/".join(
                    [
                        self.nep_field,
                        label,
                        "epoch_{}".format(self.epoch),
                        "interactive_img",
                    ]
                )
                self.nep_run[field_name].upload(figure)

    def compute_rows_per_plot(self):
        self.rows_per_plot = [len(self.dfs[label]) for label in self.df_titles]

    def create_limited_df(self, dfd):
        dfd = dfd[-self.rows_per_plot[self.training] : :]
        return dfd

    def df_plottable(self, dfd):

        df2 = dfd.melt(id_vars=["case_id", "filename"])
        df2 = df2[df2.variable.str.contains("Unnamed") == False]
        df2.variable = df2.variable.astype("category")
        return df2

    def create_plotly(self, dfd):

        df2 = self.df_plottable(dfd)
        figure = px.box(df2, x="case_id", y="value", color="variable")
        return figure

    def create_plot_sns(self, dfd):
        sns.set(rc=self.rcs[self.training])
        plt.ioff()
        df2 = self.df_plottable(dfd)
        ax = sns.boxplot(x="case_id", y="value", hue="variable", data=df2)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        figure = ax.figure
        figure.tight_layout()
        return figure


def case_id_from_series(series):
    output = [cleanup_fname(Path(y).name) for y in series]
    return output


# %%
if __name__ == "__main__":
    D = DropBBox()
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
