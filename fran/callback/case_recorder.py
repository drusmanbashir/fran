import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fran.callback.base import *
from fastai.callback.tracker import Callback
from fran.utils.helpers import get_case_id_from_filename
import itertools as il

from neptune.types import File
# %%

class CaseIDRecorder(Callback):
    def __init__(self,freq=25, local_folder='/tmp',dpi=300):
        '''

        :param freq:
        :param local_folder:
        :param dpi:
        '''
        self.nep_field = "metrics/case_id_dices"
        self.set_figure_params(dpi)
        store_attr()


    def before_fit(self):
        self.losses_dice=[],[]
        self.df_titles = ['valid','train']
        self.dfs ={'valid':None, 'train':None}
    
    def before_batch(self):
        self.files_this_batch=[]
        for i, fn in enumerate (self.learn.yb[0]):
            self.files_this_batch.append(fn)
        self.learn.yb =  [self.learn.xb[1]]
        self.learn.xb =  [self.learn.xb[0]]
    def after_batch(self):
        dict_this_batch={}
        full_dicts=[]
        for k , v in self.learn.loss_func.loss_dict.items():
            if 'batch' in k:
                dict_this_batch.update({k:v})
        for ind,name in enumerate(self.files_this_batch):
            pat = "label\d"
            dici={ 'filename':name}
            relevant_losses = {re.search(pat,key).group():dict_this_batch[key] for key in dict_this_batch.keys() if 'batch{}'.format(ind) in key}
            dici.update(relevant_losses)
            full_dicts.append(dici)
        self.losses_dice[self.training].append(full_dicts)

    def after_epoch(self):
        for i,label in enumerate(self.df_titles):
            df = pd.DataFrame.from_dict(il.chain.from_iterable(self.losses_dice[i]))
            df['case_id']= case_id_from_series(df.filename)
            self.append_to_running_df(label, df)
        if all([self.epoch>=self.freq , self.epoch%self.freq==0]):
            if not hasattr(self,'rows_per_plot'): self.compute_rows_per_plot()
            self.store_results()

    def set_figure_params(self,dpi):
        self.rcs = [ {'figure.dpi': dpi, 'figure.figsize':(15,10)},  # valid
                 {'figure.dpi': dpi, 'figure.figsize':(25,10)}]     # train

    def append_to_running_df(self,label,df):
        if isinstance(self.dfs[label],pd.DataFrame):
            self.dfs[label] = pd.concat([self.dfs[label],df],axis=0)
        else:
             self.dfs[label]: self.dfs[label] = df

    def store_results(self):
                for label in self.df_titles:
                    storage_string_plot = label+"_plot_epoch{}".format(self.epoch)
                    storage_string_df = label
                    small_df = self.create_limited_df(self.dfs[label])
                    figure = self.create_plot(small_df)
                    if hasattr(self.learn,'nep_run') :
                        self.nep_run[self.nep_field+"_dataframes/{}".format(storage_string_df)].upload(File.as_html(self.dfs[label]))
                        self.nep_run[self.nep_field+"_plots/{}".format(storage_string_plot)].upload(figure)
                    else:
                        fname_df = Path(self.local_folder)/("{}.csv".format(label))
                        fname_plot = fname_df.str_replace(".csv,.jpg")
                        self.dfs[label].to_csv(fname_df, index=False)
    def compute_rows_per_plot(self):
        self.rows_per_plot =[len(self.dfs[label]) for label in self.df_titles]

    def create_limited_df(self,dfd):
        dfd = dfd[-self.rows_per_plot[self.training]::]
        return dfd

                
    def create_plot(self, dfd):
        sns.set(rc=self.rcs[self.training])
        plt.ioff()
        df2 = dfd.melt(id_vars=['case_id','filename'])
        ax= sns.boxplot(x='case_id',y='value',hue='variable' ,data=df2)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        figure = ax.figure
        figure.tight_layout()
        return figure


def case_id_from_series(series):
    output = [get_case_id_from_filename(None , Path(y)) for y in series]
    return output

if __name__ == "__main__":
    dfd = pd.read_csv('fran/managers/dsd.csv')


